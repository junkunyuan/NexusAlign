"""
AWM on MiniMax-H3-Image: group rollouts on the native six-step PDD sampler.

The advantage-weighted flow matching loss, the generalized Huber log-likelihood, and
the two KLs stay those of AWM. Only the model calls change: H3 packs the text and image
rows into one sequence, needs its own layout per prompt, and conditions the DiT on the
index of the PDD step. The six PDD steps are the trained timesteps, and all images of
one prompt share the noise of each step, as the official code does.

Each prompt group lives on one rank, because a packed sequence holds one prompt. The
sampler gives whole groups, and one training batch is one group.

References:
    - AWM:
        - Official code: https://github.com/scxue/advantage_weighted_matching
            - advantage_weighted_matching/scripts/train_sd3_awm.py:
              compute_log_prob_awm, for the generalized Huber log-likelihood; the
              training loop of main, for the policy loss and the two KLs
            - advantage_weighted_matching/flow_grpo/prompt_noise.py:
              prompt_training_seed, shared_prompt_training_inputs
    - MiniMax-H3-Image:
        - Weights: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image
"""

import torch

from nexus_align.algorithms.awm import AWM, prompt_seed
from nexus_align.algorithms.minimax_h3_image_group import MiniMaxH3ImageGroup


class AWMMiniMaxH3Image(MiniMaxH3ImageGroup, AWM):
    """AWM training algorithm on the MiniMax-H3-Image DiT."""

    def __init__(self, cfg_algorithm):
        super().__init__(cfg_algorithm)
        if self.kl_cfg_scale != 1.0:
            raise ValueError("❌ H3-Image has no CFG, so the base KL uses none either")
        if self.num_train_timesteps != self.num_steps:
            raise ValueError("❌ H3-Image trains every PDD step; T must be num_steps")
        if self.noise_level:
            raise ValueError("❌ The native H3-Image PDD sampler is the ODE")
        self.init_groups(cfg_algorithm)

    def prepare(self, model, device, pipeline) -> None:
        """Bind the DiT and the model wrapper, and take the native PDD schedule."""
        super().prepare(model, device, pipeline)
        self.bind_schedule(pipeline)

    def forward(self, data):
        """
        Calculate the AWM loss of one chunk of members at one PDD step.

        The trainer counts num_steps * num_chunks micro steps per batch, and it divides
        each loss by their number. Equal chunks then give the mean over all members.
        """
        j, chunk = divmod(data["step"], self.num_chunks)
        members = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)
        embeds, layout, group_noise = self.group_layout(data)

        # 1. Add the noise of this step to the clean rows at its noise level. All
        # members of the group share it.
        x0 = data["latents"][members].float()
        sigma, noise = self.sigmas[j], group_noise[j].float()
        xt = (1 - sigma) * x0 + sigma * noise

        # 2. Predict the velocities of the EMA reference, the policy, and the base
        # model (LoRA disabled). H3 has no CFG, so the base model runs without one.
        args = (xt, embeds, layout, j)
        with torch.no_grad(), self.reference():
            ref_pred = self.pipeline.velocity(self.model, *args).float()
        pred = self.pipeline.velocity(self.model, *args).float()
        with torch.no_grad(), self.model.disable_adapter():
            base_pred = self.pipeline.velocity(self.model, *args).float()

        # 3. Weight the gradient of each image's log-likelihood surrogate by its
        # advantage. On policy, the ratio is 1, so the PPO clip has no effect; see paper
        # Eq. 15 and 17.
        dims = tuple(range(1, x0.ndim))
        mse = ((pred - (noise - x0)) ** 2).mean(dim=dims)
        p = self.ghuber_power
        log_prob = -((mse + 1e-10) ** p - 1e-10**p) * sigma / p
        ratio = torch.exp(log_prob - log_prob.detach())
        adv_clip_max = self.adv_clip_max
        advantages = data["advantages"][members]
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
        advantages = advantages / adv_clip_max * self.advantage_max
        policy_loss = torch.mean(-advantages * ratio)

        # 4. Use the squared velocity differences as the KLs to the base model and to
        # the EMA reference; see paper Eq. 16.
        kl_loss = torch.mean(((pred - base_pred) ** 2).mean(dim=dims))
        ref_kl_loss = torch.mean(((pred - ref_pred) ** 2).mean(dim=dims))
        loss = policy_loss + self.kl_beta * kl_loss + self.ref_kl_beta * ref_kl_loss
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "ref_kl_loss": ref_kl_loss.detach(),
            "fm_loss": mse.mean().detach(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Integrate the rows of one group with the native six-step Euler sampler.

        Return the final rows. The AWM loss needs only them, because it trains on the
        forward process of the clean rows.
        """
        embeds, layout = data["embeds"], data["layout"]
        sigmas = self.sigmas.tolist()
        out = []
        for chunk in data["rows"].split(self.chunk_size):
            rows = chunk
            for j in range(self.num_steps):
                velocity = self.pipeline.velocity(
                    self.model, rows, embeds, layout, j
                ).float()
                rows = rows + (sigmas[j + 1] - sigmas[j]) * velocity
            out.append(rows)
        return {"rows": torch.cat(out)}

    @torch.no_grad()
    def training_noise(self, prompt: str, epoch: int, rows: torch.Tensor):
        """Draw the shared training noise of each PDD step from the prompt and epoch."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(prompt_seed(epoch, prompt))
        return torch.randn(
            (self.num_steps, *rows.shape[1:]),
            generator=generator, device=self.device, dtype=rows.dtype,
        )

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """
        Roll out one complete prompt group. Return its images and training noise.

        The trainer gives the images of one group, so all prompts are the same. Every
        member draws its own initial noise from a seed that is unique in the run.
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
        noise = self.training_noise(prompts[0], epoch, out["rows"])
        self.groups.append((embeds, layout, noise))
        samples = {
            "latents": out["rows"],  # the clean rows, which the AWM loss noises again
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
