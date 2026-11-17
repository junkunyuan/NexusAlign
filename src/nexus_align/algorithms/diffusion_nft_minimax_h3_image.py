"""
DiffusionNFT on MiniMax-H3-Image: group rollouts on the native six-step PDD sampler.

The NFT loss, the optimality weights, the soft-updated old policy, and the KL to the
reference policy stay those of DiffusionNFT. Only the model calls change: H3 packs the
text and image rows into one sequence, needs its own layout per prompt, and conditions
the DiT on the index of the PDD step.

Each prompt group lives on one rank, because a packed sequence holds one prompt. The
sampler gives whole groups, and one training batch is one group.

References:
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
            - scripts/train_nft_sd3.py: the NFT loss, the adaptive x0 weight, and the
              velocity KL of main; return_decay, for the soft update
    - MiniMax-H3-Image:
        - Weights: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image
"""

import torch

from nexus_align.algorithms.diffusion_nft import DiffusionNFT, adaptive_x0_loss
from nexus_align.algorithms.minimax_h3_image_group import MiniMaxH3ImageGroup


class DiffusionNFTMiniMaxH3Image(MiniMaxH3ImageGroup, DiffusionNFT):
    """DiffusionNFT training algorithm on the MiniMax-H3-Image DiT."""

    def __init__(self, cfg_algorithm):
        super().__init__(cfg_algorithm)
        if self.solver != "flow":
            raise ValueError("❌ H3-Image samples with its native Euler PDD sampler")
        if self.timestep_fraction != 1.0:
            raise ValueError("❌ H3-Image trains every PDD step; timestep_fraction 1")
        self.init_groups(cfg_algorithm)

    def prepare(self, model, device, pipeline) -> None:
        """Bind the DiT and the model wrapper, and take the native PDD schedule."""
        super().prepare(model, device, pipeline)
        self.bind_schedule(pipeline)

    def forward(self, data):
        """
        Calculate the NFT loss of one chunk of members at one PDD step; see paper Eq. 5.

        Every member of a forward pass shares the PDD step, because the DiT takes one
        step index. The trainer counts num_steps * num_chunks micro steps per batch,
        and equal chunks then give the mean over all members.
        """
        j, chunk = divmod(data["step"], self.num_chunks)
        members = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)
        embeds, layout = self.group_layout(data)

        # 1. Add noise to the clean rows at the noise level of this PDD step.
        x0 = data["latents"][members].float()
        sigma = self.sigmas[j]
        noise = torch.randn_like(x0)
        xt = (1 - sigma) * x0 + sigma * noise

        # 2. Predict the old, trained, and reference (LoRA disabled) velocities.
        args = (xt, embeds, layout, j)
        with torch.no_grad(), self.old_policy():
            old_pred = self.pipeline.velocity(self.model, *args).float()
        pred = self.pipeline.velocity(self.model, *args).float()
        with torch.no_grad(), self.model.disable_adapter():
            ref_pred = self.pipeline.velocity(self.model, *args).float()

        # 3. Weight the losses of the implicit positive and negative policies with the
        # optimality r.
        adv_clip_max = self.adv_clip_max
        advantages = data["advantages"][members]
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
        # Normalize the advantages to [0, 1].
        r = torch.clamp((advantages / adv_clip_max) / 2.0 + 0.5, 0, 1)
        beta = self.beta
        positive_pred = beta * pred + (1 - beta) * old_pred
        negative_pred = (1.0 + beta) * old_pred - beta * pred
        positive_loss = adaptive_x0_loss(xt - sigma * positive_pred, x0)
        negative_loss = adaptive_x0_loss(xt - sigma * negative_pred, x0)
        policy_loss = r * positive_loss / beta + (1.0 - r) * negative_loss / beta
        policy_loss = (policy_loss * adv_clip_max).mean()

        # 4. Use the squared velocity difference as the KL to the reference policy.
        dims = tuple(range(1, x0.ndim))
        kl_loss = torch.mean(((pred - ref_pred) ** 2).mean(dim=dims))
        loss = policy_loss + self.kl_beta * kl_loss
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "old_deviate": torch.mean((pred - old_pred) ** 2).detach(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Integrate the rows of one group with the native six-step Euler sampler.

        Return the final rows. The NFT loss needs only them, because it trains on the
        forward process of the clean rows.
        """
        net = data.get("net", self.model)
        embeds, layout = data["embeds"], data["layout"]
        sigmas = self.sigmas.tolist()
        out = []
        for chunk in data["rows"].split(self.chunk_size):
            rows = chunk
            for j in range(self.num_steps):
                velocity = self.pipeline.velocity(net, rows, embeds, layout, j).float()
                rows = rows + (sigmas[j + 1] - sigmas[j]) * velocity
            out.append(rows)
        return {"rows": torch.cat(out)}

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """
        Roll out one complete prompt group with the old policy.

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

        with self.old_policy():
            out = self.sample({"rows": rows, "embeds": embeds, "layout": layout})
        self.groups.append((embeds, layout))
        samples = {
            "latents": out["rows"],  # the clean rows, which the NFT loss noises again
            "group_slots": torch.full((len(prompts),), slot, device=self.device),
        }
        responses = self.pipeline.decode(out["rows"], latent_size)
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
            images.append(self.pipeline.decode(out["rows"], layout[0]))
        return torch.cat(images)
