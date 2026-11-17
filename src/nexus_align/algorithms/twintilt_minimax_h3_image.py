"""
TwinTilt on MiniMax-H3-Image: group rollouts on the native six-step PDD sampler.

The projection, the targets, and the loss stay those of TwinTilt. Only the model calls
change: H3 packs the text and image rows into one sequence, needs its own layout per
prompt, and trains without autocast.

Each prompt group lives on one rank, because a packed sequence holds one prompt. The
sampler gives whole groups, and one training batch is one group.

References:
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private)
            - src/nexus_align/twintilt_h3/backend.py: TwinTiltH3Backend.sample, forward
            - src/nexus_align/twintilt_h3/sampling.py: initial_group_rows, rollout_batch
            - src/nexus_align/twintilt_h3/trainer.py: TwinTiltH3Trainer.sample_epoch
            - src/nexus_align/twintilt_h3/settings.py: validate_sampling
"""

import torch

from nexus_align.algorithms.twintilt import TwinTilt, churn_step
from nexus_align.algorithms.minimax_h3_image_group import MiniMaxH3ImageGroup


class TwinTiltMiniMaxH3Image(MiniMaxH3ImageGroup, TwinTilt):
    """TwinTilt training algorithm on the MiniMax-H3-Image DiT."""

    def __init__(self, cfg_algorithm):
        super().__init__(cfg_algorithm)
        if self.kl_cfg_scale != 1.0:
            raise ValueError("❌ H3-Image has no CFG, so the base KL uses none either")
        self.init_groups(cfg_algorithm)

    def prepare(self, model, device, pipeline) -> None:
        """Bind the DiT and the model wrapper, and take the native PDD schedule."""
        super().prepare(model, device, pipeline)
        self.bind_schedule(pipeline)

    def forward(self, data):
        """
        Regress the velocity of one chunk of members at one PDD step onto its target.

        The trainer counts num_steps * num_chunks micro steps per batch, and it divides
        each loss by their number. Equal chunks then give the mean over all members.
        """
        j, chunk = divmod(data["step"], self.num_chunks)
        members = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)
        embeds, layout = self.group_layout(data)
        args = (data["xs"][members, j].float(), embeds, layout, j)
        pred = self.pipeline.velocity(self.model, *args).float()
        with torch.no_grad(), self.model.disable_adapter():
            base_pred = self.pipeline.velocity(self.model, *args).float()

        # Every member and timestep of a group moves by the same correction.
        target = data["velocities"][members, j].float() - data["displacement"][members]
        dims = tuple(range(1, pred.ndim))
        policy_loss = (pred - target).square().mean(dims).mean()
        kl_loss = (pred - base_pred).square().mean(dims).mean()
        loss = policy_loss + self.kl_weight * kl_loss
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Integrate the rows of one group with the native six-step Euler sampler.

        Return the final rows, and the inputs and velocities of the model calls.
        """
        net = data.get("net", self.model)
        churn = data.get("churn_weight", self.churn_weight)
        embeds, layout = data["embeds"], data["layout"]
        sigmas = self.sigmas.tolist()
        out = []
        for chunk in data["rows"].split(self.chunk_size):
            rows, xs, velocities = chunk, [], []
            for j in range(self.num_steps):
                velocity = self.pipeline.velocity(net, rows, embeds, layout, j).float()
                xs.append(rows)
                velocities.append(velocity)
                rows = churn_step(rows, velocity, sigmas[j], sigmas[j + 1], churn)
            out.append((rows, torch.stack(xs, dim=1), torch.stack(velocities, dim=1)))
        return {
            "rows": torch.cat([x[0] for x in out]),
            "xs": torch.cat([x[1] for x in out]),  # (K, num_steps, rows, channels)
            "velocities": torch.cat([x[2] for x in out]),
        }

    def group_noise(self, seed: int, latent_size: int, num_images: int) -> torch.Tensor:
        """
        Draw the initial rows of one group.

        At the trajectory level, one noise seeds the whole group, and the churn prior
        then gives each image its own copy. At the distribution level, each image draws
        its own noise.
        """
        if self.optimization_level != "trajectory":
            noise = [
                self.pipeline.initial_noise(seed * self.group_size + i, latent_size)
                for i in range(num_images)
            ]
            return torch.stack(noise).float()
        rows = self.pipeline.initial_noise(seed, latent_size)
        rows = rows.expand(num_images, -1, -1).float()
        if self.churn_weight <= 0:
            return rows.clone()
        gamma = torch.tensor(self.churn_weight)  # fp32 coefficients, as in FDFO
        scale = 1 / (gamma + 1)
        noise_scale = ((gamma + 1).square() - 1).sqrt() / (gamma + 1)
        return scale * rows + noise_scale * torch.randn_like(rows)

    @torch.no_grad()
    def rollout(self, prompts: list[str], group_ids: list[int]) -> dict:
        """
        Roll out one complete prompt group with the EMA, which tracks the policy.

        The trainer gives the images of one group, so all prompts and IDs are the same.
        """
        if len(set(prompts)) != 1 or len(set(group_ids)) != 1:
            raise ValueError("❌ One H3-Image batch holds one prompt group")
        if len(prompts) != self.group_size:
            raise ValueError("❌ H3-Image rolls out complete groups")
        embeds, tags = self.pipeline.encode_prompt(prompts[0], self.max_sequence_length)
        layout = self.pipeline.layout(tags, self.resolution)
        latent_size = layout[0]
        rows = self.group_noise(group_ids[0], latent_size, len(prompts))

        net = self.pipeline.ema if self.pipeline.ema is not None else self.model
        out = self.sample({
            "rows": rows, "embeds": embeds, "layout": layout, "net": net,
        })
        slot = len(self.groups)
        self.groups.append((embeds, layout))
        samples = {
            "xs": out["xs"],
            "velocities": out["velocities"],
            "latents": out["rows"],  # the endpoints, for the projection
            "group_ids": torch.tensor(group_ids, device=self.device),
            "group_slots": torch.full((len(prompts),), slot, device=self.device),
        }
        responses = self.pipeline.decode(out["rows"], latent_size)
        return {"samples": samples, "responses": responses}

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int, guidance_scale: float):
        """Make evaluation images with the native sampler, without churn."""
        if guidance_scale != 1.0:
            raise ValueError("❌ H3-Image samples without CFG")
        images = []
        for prompt in prompts:
            embeds, tags = self.pipeline.encode_prompt(prompt, self.max_sequence_length)
            layout = self.pipeline.layout(tags, self.resolution)
            seed = int(torch.randint(2**31, (1,)))
            out = self.sample({
                # initial_noise gives one image's rows, and sample takes a batch.
                "rows": self.pipeline.initial_noise(seed, layout[0]).float()[None],
                "embeds": embeds,
                "layout": layout,
                "churn_weight": 0.0,
            })
            images.append(self.pipeline.decode(out["rows"], layout[0]))
        return torch.cat(images)
