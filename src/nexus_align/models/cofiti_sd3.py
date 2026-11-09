"""SD3.5 wrapper preserving the local FDFO LoRA/DDP initialization order."""

import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel

from nexus_align.models.sd3 import SD3Model


class CoFiTiSD3Model(SD3Model):
    """Reuse SD3 conditioning with BF16 frozen weights and FP32 student LoRA.

    The only EMA is the trainer's small LoRA shadow, not a second transformer.
    """

    def __init__(self, cfg_model, build_keys=("model",)):
        if tuple(build_keys) != ("model",):
            raise ValueError("CoFiTi evaluation uses model_key=model")
        self.cfg_model = cfg_model
        self.device = torch.device("cuda", torch.cuda.current_device())
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            cfg_model.path, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        self.scheduler = pipeline.scheduler
        self.image_processor = pipeline.image_processor
        self.vae = pipeline.vae.to(self.device).requires_grad_(False)
        self.tokenizers = [
            pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3,
        ]
        self.text_encoders = [
            pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3,
        ]
        for encoder in self.text_encoders:
            encoder.to(self.device).requires_grad_(False).eval()
        transformer = pipeline.transformer.to(self.device).requires_grad_(False)
        lora = cfg_model.lora
        transformer = get_peft_model(transformer, LoraConfig(
            r=lora.rank, lora_alpha=lora.alpha,
            init_lora_weights=lora.init, target_modules=list(lora.target_modules),
        ))
        for parameter in transformer.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if cfg_model.get("activation_ckpt", False):
            transformer.enable_gradient_checkpointing()
        self.model = transformer
        if dist.is_initialized():
            self.model = DistributedDataParallel(
                transformer, device_ids=[self.device.index],
                broadcast_buffers=False, find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        self.ema = None
        self._eval_pipeline = None
        self.save_trainable_only = True
        print("📦 Prepared CoFiTi SD3.5 with DDP and FP32 student LoRA")

    @property
    def transformer(self):
        """Return the unwrapped PEFT transformer for frozen rollout/base forwards."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def text_encoders_to(self, device):
        """Move the frozen encoders between prompt preparation and rollout."""
        for encoder in self.text_encoders:
            encoder.to(device)

    @torch.no_grad()
    def generate_evaluation(
        self, prompts, num_steps, guidance_scale, seeds,
        resolution=512, max_sequence_length=256,
    ):
        """Match FDFO's eval/models.py run_pipe using the loaded student weights."""
        if len(seeds) != len(prompts):
            raise ValueError("Evaluation requires one seed per image")
        if self._eval_pipeline is None:
            # Share weights, but keep evaluation scheduler state outside training.
            self._eval_pipeline = StableDiffusion3Pipeline(
                transformer=self.transformer,
                scheduler=FlowMatchEulerDiscreteScheduler.from_config(
                    self.scheduler.config
                ),
                vae=self.vae,
                text_encoder=self.text_encoders[0],
                text_encoder_2=self.text_encoders[1],
                text_encoder_3=self.text_encoders[2],
                tokenizer=self.tokenizers[0],
                tokenizer_2=self.tokenizers[1],
                tokenizer_3=self.tokenizers[2],
            )
            self._eval_pipeline.set_progress_bar_config(disable=True)
        generators = [
            torch.Generator(device="cpu").manual_seed(int(seed) % (1 << 31))
            for seed in seeds
        ]
        self.text_encoders_to(self.device)
        # The source evaluates BF16 weights with FP32 LoRA, without training AMP.
        with torch.autocast(self.device.type, enabled=False):
            return self._eval_pipeline(
                prompt=prompts, generator=generators, output_type="pt",
                num_inference_steps=num_steps, guidance_scale=guidance_scale,
                height=resolution, width=resolution,
                max_sequence_length=max_sequence_length,
            ).images

    @torch.no_grad()
    def decode(self, latents):
        """Decode without an extra BF16 rounding of the FP32 solver endpoint."""
        config = self.vae.config
        latents = latents / config.scaling_factor + config.shift_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(images, output_type="pt")
