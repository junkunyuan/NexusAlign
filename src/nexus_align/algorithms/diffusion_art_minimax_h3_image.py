"""
DiffusionART on MiniMax-H3-Image: group rollouts on the native six-step PDD sampler.

The guidance of each group, the frozen target, the sigma weight, and the trust-region
mask stay those of DiffusionART. Only the model calls change: H3 packs the text and
image rows into one sequence, needs its own layout per prompt, and conditions the DiT
on the index of the PDD step.

Each prompt group lives on one rank, because a packed sequence holds one prompt. The
sampler gives whole groups, and the two disjoint updates split the members of each
group.

References:
    - DiffusionART:
        - Official code: https://github.com/vanzll/DiffusionART
            - src/visual_rl/algorithms/art.py: art_loss, for the frozen target, the
              sigma weight, and the trust-region mask
            - src/visual_rl/training/rollout.py: collect_rollout, for the Euler ODE
              rollouts and their cached velocities
    - MiniMax-H3-Image:
        - Weights: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image
"""

import torch

from nexus_align.algorithms.diffusion_art import DiffusionART, prompt_ids
from nexus_align.algorithms.minimax_h3_image_group import MiniMaxH3ImageGroup


class DiffusionARTMiniMaxH3Image(MiniMaxH3ImageGroup, DiffusionART):
    """DiffusionART training algorithm on the MiniMax-H3-Image DiT."""

    def __init__(self, cfg_algorithm):
        super().__init__(cfg_algorithm)
        self.init_groups(cfg_algorithm)

    def prepare(self, model, device, pipeline) -> None:
        """Bind the DiT and the model wrapper, and take the native PDD schedule."""
        super(DiffusionART, self).prepare(model, device)
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler
        self.bind_schedule(pipeline)

    def forward(self, data):
        """
        Calculate the ART loss of one chunk of members at one PDD step; see paper Eq. 6.

        The trainer counts num_steps * num_chunks micro steps per batch, and it divides
        each loss by their number. Equal chunks then give the mean over all members.
        """
        j, chunk = divmod(data["step"], self.num_chunks)
        members = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)
        embeds, layout = self.group_layout(data)

        # 1. Predict the velocity at the cached rollout state of this timestep.
        xt = data["latents"][members, j].float()
        old = data["velocities"][members, j].float()
        sigma, dt = self.sigmas[j], self.sigmas[j + 1] - self.sigmas[j]
        pred = self.pipeline.velocity(self.model, xt, embeds, layout, j).float()

        # 2. Regress it to the frozen target. The velocity points from the clean
        # latents to the noise, so the target subtracts the guidance.
        guidance = data["guidance"][members]
        target = old - guidance
        dims = tuple(range(1, xt.ndim))
        mse = ((pred - target) ** 2).mean(dim=dims)  # see paper Eq. 6

        # 3. Mask the outward moves whose drift from the old velocity is too large.
        # The drift is the KL of the reference SDE steps with the current and the old
        # velocities, per latent dimension; see paper Eq. 7 and 8.
        with torch.no_grad():
            drift = pred - old
            nu = self.mask_noise_level
            if sigma == 1:  # the first step uses the endpoint convention
                coef, var = dt, nu**2
            else:
                coef, var = dt * (1 + nu**2 / 2), nu**2 * sigma / (1 - sigma) * -dt
            kl = coef**2 * drift.square().mean(dim=dims) / (2 * var)
            outward = (drift * (target - pred)).mean(dim=dims) > 0
            masked = (kl >= self.mask_threshold) & outward
            # A group with equal rewards has zero guidance and no loss.
            active = guidance.flatten(1).abs().amax(dim=1) > 0
            keep = active & ~masked

        # 4. Weight each loss by sigma, and average over the members of the chunk.
        loss = (sigma * mse * keep).mean()
        metrics = {
            "masked_ratio": masked.float().mean(),
            "active_ratio": active.float().mean(),
            "drift_kl": kl.mean(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Integrate the rows of one group with the native six-step Euler sampler.

        Return the states before and after each step, and the velocity at each state.
        """
        embeds, layout = data["embeds"], data["layout"]
        sigmas = self.sigmas.tolist()
        out = []
        for chunk in data["rows"].split(self.chunk_size):
            rows, all_rows, velocities = chunk, [chunk], []
            for j in range(self.num_steps):
                velocity = self.pipeline.velocity(
                    self.model, rows, embeds, layout, j
                ).float()
                rows = rows + velocity * (sigmas[j + 1] - sigmas[j])
                all_rows.append(rows)
                velocities.append(velocity)
            out.append((torch.stack(all_rows, dim=1), torch.stack(velocities, dim=1)))
        return {
            # (K, num_steps + 1, rows, channels) and (K, num_steps, rows, channels)
            "latents": torch.cat([x[0] for x in out]),
            "velocities": torch.cat([x[1] for x in out]),
        }

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """
        Roll out one complete prompt group. Return its images, states, and velocities.

        The trainer gives the images of one group, so all prompts are the same. Every
        member draws its own noise from a seed that is unique in the run.
        """
        if len(set(prompts)) != 1:
            raise ValueError("❌ One H3-Image batch holds one prompt group")
        if len(prompts) != self.group_size:
            raise ValueError("❌ H3-Image rolls out complete groups")
        embeds, tags = self.pipeline.encode_prompt(prompts[0], self.max_sequence_length)
        layout = self.pipeline.layout(tags, self.resolution)
        latent_size = layout[0]
        slot = len(self.groups)
        seed = self.group_seed(epoch, slot)
        rows = self.group_noise(seed, latent_size, len(prompts))

        out = self.sample({"rows": rows, "embeds": embeds, "layout": layout})
        self.groups.append((embeds, layout))
        samples = {
            "prompt_ids": prompt_ids(prompts).to(self.device),  # prompt groups
            "latents": out["latents"],
            "velocities": out["velocities"],
            "group_slots": torch.full((len(prompts),), slot, device=self.device),
        }
        responses = self.pipeline.decode(out["latents"][:, -1], latent_size)
        return {"samples": samples, "responses": responses}

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int, guidance_scale: float):
        """Make evaluation images with the native sampler."""
        if guidance_scale != 1.0:
            raise ValueError("❌ H3-Image samples without CFG")
        if num_steps != self.num_steps:
            raise ValueError(f"❌ H3-Image samples {self.num_steps} PDD steps")
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
            })
            images.append(self.pipeline.decode(out["latents"][:, -1], layout[0]))
        return torch.cat(images)
