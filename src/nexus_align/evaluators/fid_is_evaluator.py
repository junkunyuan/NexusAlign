"""FID-IS evaluator: distributed sampling, then the FID and IS of the images."""

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
from nexus_align.engine.distributed import broadcast_from_rank0, gather_tensor
from nexus_align.eval.metrics import build_feature_extractor, features_to_metrics


# Level 1 encodes about 3x faster than the default 6, with files only about 4% larger.
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
        # Sampling noise must differ across ranks, so seed each rank differently.
        set_seed(self.global_seed + self.rank)

        if self.compile:
            self.eval_model = torch.compile(self.eval_model)
        self.algorithm.prepare(self.eval_model, self.device)

        # Get the VAE decoder and the latent normalization tensors.
        self.vae = AutoencoderKL.from_pretrained(self.vae_path).to(self.device)
        if self.compile:
            self.vae.decoder = torch.compile(self.vae.decoder)
        self.latents_scale = torch.full((1, 4, 1, 1), self.vae_scale, device=self.device)
        self.latents_mean = torch.tensor(self.vae_mean).view(-1, 1, 1).to(self.device)
        self.latents_std = torch.tensor(self.vae_std).view(-1, 1, 1).to(self.device)

        # Round the sample count up to a whole distributed batch.
        self.global_batch_size = self.per_proc_batch_size * self.world_size
        self.iterations = math.ceil(self.num_fid_samples / self.global_batch_size)

        # Check the metric requirements once, before sampling any checkpoint.
        if self.resolution != 256:
            raise NotImplementedError(f"❌ Unsupported evaluation resolution: {self.resolution}")
        assert os.path.exists(self.fid_stats), f"❌ FID statistics not found: {self.fid_stats}"

        # Every rank scores its own shard of the samples.
        self.feature_extractor = build_feature_extractor(self.device, self.inception_weights)

    def reuse_saved_images(self, img_dir: str) -> bool:
        """Return True if the directory already holds exactly the expected images."""
        sample_ready = False
        if self.rank == 0:
            img_paths = glob.glob(os.path.join(img_dir, "*.png"))
            sample_ready = len(img_paths) == self.num_fid_samples
            if sample_ready:
                print(f"✅ Found {len(img_paths)} images in <{img_dir}>, skipping sampling")
            elif img_paths:
                print(
                    f"⚠️ WARNING: found {len(img_paths)}/{self.num_fid_samples} images, "
                    "removing the stale ones"
                )
                for path in img_paths:
                    os.remove(path)
        sample_ready = broadcast_from_rank0(sample_ready)
        dist.barrier()
        return sample_ready

    @torch.no_grad()
    def sample_images(self, img_dir: str) -> tuple[torch.Tensor, ...] | None:
        """
        Generate images in parallel. Each rank samples its own shard of indices.

        If save_images is true, a thread pool writes the shard as PNG files, and this
        method returns None. If not, it returns the Inception features of the shard.
        """
        # Skip sampling only if the directory holds exactly the expected images.
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
                index = offsets + self.rank + total  # global sample indices
                z = torch.randn(
                    self.per_proc_batch_size, self.eval_model.in_channels,
                    latent_h, latent_w, device=self.device,
                )
                if self.balanced_labels:
                    # 50 per class for 50k samples, as in iMeanFlow's evaluate.py.
                    y = index % self.eval_model.num_classes
                else:
                    y = torch.randint(
                        0, self.eval_model.num_classes, (self.per_proc_batch_size,),
                        device=self.device,
                    )
                samples = self.algorithm.sample({
                    "latents": z,
                    "labels": y,
                    "cfg_scale": self.cfg_scale,
                    "cfg_min_t": self.cfg_min_t,
                    "cfg_max_t": self.cfg_max_t,
                    "num_steps": self.num_steps,
                }).to(torch.float32)
                latents = samples * self.latents_std + self.latents_mean
                images = self.vae.decode(latents / self.latents_scale).sample
                images = torch.clamp(127.5 * images + 128.0, 0, 255)  # +128 so uint8 truncation rounds
                if self.save_images:
                    # The pool encodes the images during the sampling of the next batch.
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
                    indices.append(index)
                total += self.global_batch_size
                pbar.update(1)
            for future in futures:  # raise any error of the PNG writer threads
                future.result()
        pbar.close()

        if self.save_images:
            return None
        return torch.cat(feats_2048), torch.cat(feats_logits), torch.cat(indices)

    @torch.no_grad()
    def extract_saved_features(self, img_dir: str) -> tuple[torch.Tensor, ...]:
        """Read one strided shard of saved images per rank and get their Inception features."""
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert len(img_paths) == self.num_fid_samples, (
            f"❌ Found {len(img_paths)}/{self.num_fid_samples} images in <{img_dir}>"
        )

        # Pad the shortest shards so every rank gathers the same number of features.
        # The padding indices are out of range, so compute_metrics drops them.
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

        fid = metrics["frechet_inception_distance"]
        is_mean = metrics["inception_score_mean"]
        is_std = metrics["inception_score_std"]
        print(f"📊 FID: {fid:.2f}  Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        print(f"💾 Saved metrics to <{metrics_file}>")

    def compute_metrics(self, features: tuple, metrics_file: str) -> None:
        """Gather the Inception features of all ranks and get the metrics on rank 0."""
        gathered = [gather_tensor(tensor) for tensor in features]
        if self.rank != 0:
            return

        # Drop the padding and sort the rest in the order of the saved PNG files.
        feats_2048, feats_logits, indices = (buffer.cpu() for buffer in gathered)
        order = torch.argsort(indices)
        order = order[indices[order] < self.num_fid_samples]
        metrics = features_to_metrics(feats_2048[order], feats_logits[order], self.fid_stats)
        self.save_metrics(metrics, metrics_file)

    def evaluate(self, ckpt: str) -> None:
        """Sample images from one checkpoint, then calculate the metrics."""
        sample_dir = self.ckpt_eval_dir(ckpt)
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
