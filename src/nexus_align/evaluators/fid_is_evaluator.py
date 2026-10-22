"""FID-IS evaluator: distributed sampling followed by FID/IS on the generated images."""

import os
import json
import math
import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL

import torch
import torch.distributed as dist

from nexus_align.utils.seed import set_seed
from nexus_align.utils.progress import TqdmBar
from nexus_align.models.base_model import BaseModel
from nexus_align.models.components.embedding import to_2tuple
from nexus_align.evaluators.base_evaluator import BaseEvaluator
from nexus_align.algorithms.base_algorithm import BaseAlgorithm
from nexus_align.eval.metrics import build_feature_extractor, features_to_metrics


# Level 1 encodes about 3x faster than the default 6 and grows the files by only ~4%.
PNG_COMPRESS_LEVEL = 1


def save_png(image: np.ndarray, path: str) -> None:
    """Save one uint8 HWC image as a PNG file."""
    Image.fromarray(image).save(path, compress_level=PNG_COMPRESS_LEVEL)


def load_png(path: str) -> np.ndarray:
    """Load one PNG file as a uint8 HWC image."""
    return np.asarray(Image.open(path).convert("RGB"))


class FIDISEvaluator(BaseEvaluator):
    """Generate images and score them with FID and Inception Score."""

    def __init__(
        self,
        cfg_evaluator,
        model: BaseModel,
        algorithm: BaseAlgorithm,
        log_dir: str = "logs/temp",
    ) -> None:
        super().__init__(
            cfg_evaluator=cfg_evaluator,
            model=model,
            algorithm=algorithm,
            log_dir=log_dir
        )
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = model.device

        # Sampling noise must differ across ranks, so seed each rank differently.
        set_seed(self.global_seed + self.rank)

        # Sample with the configured weights; the algorithm owns the sampling loop.
        self.eval_model = getattr(model, self.model_key)
        assert self.eval_model is not None, (
            f"❌ Model has no <{self.model_key}> weights to evaluate"
        )
        self.eval_model.eval()
        if self.compile:
            self.eval_model = torch.compile(self.eval_model)
        self.algorithm.prepare(self.eval_model, self.device)

        # Get VAE decoder and latent normalization tensors.
        self.vae = AutoencoderKL.from_pretrained(self.vae_path).to(self.device)
        if self.compile:
            self.vae.decoder = torch.compile(self.vae.decoder)
        self.latents_scale = torch.full((1, 4, 1, 1), self.vae_scale, device=self.device)
        self.latents_bias = torch.full((1, 4, 1, 1), self.vae_bias, device=self.device)

        # Round the sample count up to a whole distributed batch.
        self.global_batch_size = self.per_proc_batch_size * self.world_size
        self.total_samples = int(
            math.ceil(self.num_fid_samples / self.global_batch_size) * self.global_batch_size
        )
        self.iterations = self.total_samples // self.global_batch_size

        # Check the metric requirements once, before any checkpoint is sampled.
        if self.resolution != 256:
            raise NotImplementedError(f"❌ Unsupported evaluation resolution: {self.resolution}")
        assert os.path.exists(self.fid_stats), f"❌ FID statistics not found: {self.fid_stats}"

        # Every rank scores its own shard of the samples.
        self.feature_extractor = build_feature_extractor(self.device, self.inception_weights)

    def metrics_path(self, ckpt: str) -> str:
        return os.path.join(self.eval_dir, os.path.basename(ckpt), "metrics.json")

    def reuse_saved_images(self, img_dir: str) -> bool:
        """Return True when the folder already holds exactly the expected images."""
        sample_ready = [False]
        if self.rank == 0:
            img_paths = glob.glob(os.path.join(img_dir, "*.png"))
            sample_ready[0] = len(img_paths) == self.num_fid_samples
            if sample_ready[0]:
                print(f"✅ Found {len(img_paths)} images in <{img_dir}>, skipping sampling")
            elif img_paths:
                print(
                    f"⚠️ WARNING: found {len(img_paths)}/{self.num_fid_samples} images, "
                    "removing the stale ones"
                )
                for path in img_paths:
                    os.remove(path)
        dist.broadcast_object_list(sample_ready, src=0)
        dist.barrier()
        return sample_ready[0]

    @torch.no_grad()
    def sample_images(self, img_dir: str) -> tuple[torch.Tensor, ...] | None:
        """Generate images in parallel, each rank handling its own shard of indices.

        With save_images the shard is written as PNG files by a thread pool and
        nothing is returned; otherwise its Inception features are returned instead.
        """
        # Skip sampling only when the folder holds exactly the expected images.
        if self.save_images and self.reuse_saved_images(img_dir):
            return None

        target = f"<{img_dir}>" if self.save_images else "memory"
        print(f"⏳ Sampling {self.num_fid_samples} images to {target}")
        latent_h, latent_w = to_2tuple(self.model.cfg_model.input_size)
        offsets = torch.arange(self.per_proc_batch_size, device=self.device) * self.world_size
        pbar = TqdmBar(total=self.iterations, desc="Sampling", unit="batch")
        feats_2048, feats_logits, indices, futures = [], [], [], []
        total = 0
        with ThreadPoolExecutor(self.num_workers) as pool:
            for _ in range(self.iterations):
                z = torch.randn(
                    self.per_proc_batch_size, self.eval_model.in_channels,
                    latent_h, latent_w, device=self.device,
                )
                y = torch.randint(
                    0, self.eval_model.num_classes, (self.per_proc_batch_size,), device=self.device
                )
                samples = self.algorithm.sample({
                    "latents": z,
                    "labels": y,
                    "cfg_scale": self.cfg_scale,
                    "cfg_min_t": self.cfg_min_t,
                    "cfg_max_t": self.cfg_max_t,
                    "num_steps": self.num_steps,
                }).to(torch.float32)
                images = self.vae.decode((samples - self.latents_bias) / self.latents_scale).sample
                images = torch.clamp(127.5 * images + 128.0, 0, 255)  # +128 so uint8 truncation rounds
                if self.save_images:
                    # Encoding runs in the pool, overlapping with the next sampled batch.
                    array = images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                    for i, image in enumerate(array):
                        index = i * self.world_size + self.rank + total
                        if index < self.num_fid_samples:
                            path = os.path.join(img_dir, f"{index:06d}.png")
                            futures.append(pool.submit(save_png, image, path))
                else:
                    f2048, flogits = self.feature_extractor(images.to(torch.uint8))
                    feats_2048.append(f2048)
                    feats_logits.append(flogits)
                    indices.append(offsets + self.rank + total)
                total += self.global_batch_size
                pbar.update(1)
            for future in futures:  # surface any error raised while writing the files
                future.result()
        pbar.close()

        if self.save_images:
            return None
        return torch.cat(feats_2048), torch.cat(feats_logits), torch.cat(indices)

    @torch.no_grad()
    def extract_saved_features(self, img_dir: str) -> tuple[torch.Tensor, ...]:
        """Read the saved images, one strided shard per rank, and score them with Inception."""
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert len(img_paths) == self.num_fid_samples, (
            f"❌ Found {len(img_paths)}/{self.num_fid_samples} images in <{img_dir}>"
        )

        # Pad the shortest shards so every rank gathers the same number of features;
        # the padding indices are out of range and get dropped after the gather.
        shard_paths = img_paths[self.rank::self.world_size]
        shard_indices = list(range(self.rank, len(img_paths), self.world_size))
        padding = math.ceil(len(img_paths) / self.world_size) - len(shard_paths)
        shard_paths = shard_paths + shard_paths[-1:] * padding
        shard_indices = shard_indices + [self.num_fid_samples] * padding

        batches = math.ceil(len(shard_paths) / self.per_proc_batch_size)
        pbar = TqdmBar(total=batches, desc="Inception features", unit="batch")
        feats_2048, feats_logits = [], []
        with ThreadPoolExecutor(self.num_workers) as pool:
            for start in range(0, len(shard_paths), self.per_proc_batch_size):
                chunk = shard_paths[start: start + self.per_proc_batch_size]
                array = np.stack(list(pool.map(load_png, chunk)))
                images = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
                f2048, flogits = self.feature_extractor(images.to(self.device))
                feats_2048.append(f2048)
                feats_logits.append(flogits)
                pbar.update(1)
        pbar.close()

        indices = torch.tensor(shard_indices, device=self.device)
        return torch.cat(feats_2048), torch.cat(feats_logits), indices

    def save_metrics(self, metrics: dict, metrics_file: str) -> None:
        """Save and report the metrics of one checkpoint."""
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        fid = metrics.get("frechet_inception_distance")
        is_mean = metrics.get("inception_score_mean")
        is_std = metrics.get("inception_score_std")
        print(f"📊 FID: {fid:.2f}  Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        print(f"💾 Saved metrics to <{metrics_file}>")

    def compute_metrics(self, features: tuple, metrics_file: str) -> None:
        """Gather the Inception features scored by every rank and reduce them on rank 0."""
        gathered = []
        for tensor in features:
            buffer = torch.empty(
                (self.world_size * tensor.shape[0], *tensor.shape[1:]),
                dtype=tensor.dtype, device=self.device,
            )
            dist.all_gather_into_tensor(buffer, tensor.contiguous())
            gathered.append(buffer)
        if self.rank != 0:
            return

        # Drop the padding samples and restore the order the saved PNG files would have.
        feats_2048, feats_logits, indices = (buffer.cpu() for buffer in gathered)
        order = torch.argsort(indices)
        order = order[indices[order] < self.num_fid_samples]
        metrics = features_to_metrics(feats_2048[order], feats_logits[order], self.fid_stats)
        self.save_metrics(metrics, metrics_file)

    def evaluate(self, ckpt: str) -> None:
        """Sample images from one checkpoint, then compute the metrics."""
        sample_dir = os.path.join(self.eval_dir, os.path.basename(ckpt))
        img_dir = os.path.join(sample_dir, "img_dir")
        if self.rank == 0:
            os.makedirs(img_dir if self.save_images else sample_dir, exist_ok=True)
        dist.barrier()

        features = self.sample_images(img_dir)
        dist.barrier()

        print("⏳ Computing evaluation metrics")
        if features is None:
            features = self.extract_saved_features(img_dir)
        self.compute_metrics(features, self.metrics_path(ckpt))
        dist.barrier()
