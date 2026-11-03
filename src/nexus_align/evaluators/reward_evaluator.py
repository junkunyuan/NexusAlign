"""Reward evaluator: sample the test prompts once and average the rewards."""

import os
import json
import math
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from nexus_align.registry import registry
from nexus_align.utils.seed import set_seed
from nexus_align.utils.progress import TqdmBar
from nexus_align.eval.summary import refresh_summary
from nexus_align.models.base_model import BaseModel
from nexus_align.datasets.text_dataset import TextDataset
from nexus_align.algorithms.base_algorithm import BaseAlgorithm
from nexus_align.engine.distributed import broadcast_from_rank0, gather_tensor
from nexus_align.evaluators.base_evaluator import BaseEvaluator, parse_train_state
from nexus_align.evaluators.fid_is_evaluator import save_png, load_png


# Configs that decide the generated images. Reuse saved images only if they match.
SAMPLE_KEYS = ("split", "global_seed", "num_steps", "guidance_scale", "model_key")


class RewardEvaluator(BaseEvaluator):
    """
    Generate one response per test prompt with the algorithm and score it with rewards.

    - Every rank samples a strided shard of the prompts. The evaluator drops the
      padding later.
    - The evaluator saves the images as PNG files with their sampling configs. A later
      run with other rewards then scores the saved images and does not sample again.
    - The evaluator scores only the rewards with missing metrics. It merges their
      metrics into the metrics file of each checkpoint.
    - Each metric is the mean over the prompts that it applies to. The value -10
      marks the other prompts.
    """

    def __init__(
        self,
        cfg_evaluator,
        model: BaseModel,
        algorithm: BaseAlgorithm,
        rewards: dict,
        log_dir: str = "logs/temp",
    ) -> None:
        super().__init__(
            cfg_evaluator=cfg_evaluator,
            model=model,
            algorithm=algorithm,
            log_dir=log_dir
        )
        self.rewards = rewards
        self.algorithm.prepare(self.eval_model, self.device, model)

        # Get the test prompts. Indices past the end pad every rank to the same count.
        dataset = TextDataset(self.data, split=self.split)
        self.prompts, self.metadatas = dataset.texts, dataset.metadatas
        self.num_prompts = len(self.prompts)
        total = math.ceil(self.num_prompts / self.world_size) * self.world_size
        self.indices = list(range(self.rank, total, self.world_size))
        size = self.per_proc_batch_size
        self.batches = [
            self.indices[i: i + size] for i in range(0, len(self.indices), size)
        ]

        # The saved images must match the prompts and the sampling configs.
        config = OmegaConf.to_container(cfg_evaluator, resolve=True)
        prompts_sha256 = hashlib.sha256("\n".join(self.prompts).encode()).hexdigest()
        self.sample_config = {
            "data": self.data.name,
            "num_prompts": self.num_prompts,
            "prompts_sha256": prompts_sha256,
            **{key: config[key] for key in SAMPLE_KEYS},
        }

    def check_config(self, saved_config: dict, config: dict) -> None:
        """Fail if a reused evaluation directory used other sampling configs."""
        super().check_config(saved_config, config)
        changed = [k for k in SAMPLE_KEYS if saved_config.get(k) != config[k]]
        if changed:
            raise ValueError(
                f"❌ <{self.eval_dir}> was sampled with other configs {changed}; "
                "set evaluator.new_run=true to evaluate in a new dir"
            )

    def unpad(self, indices: list[int]) -> list[int]:
        """Map the padding indices to the last prompt."""
        return [min(i, self.num_prompts - 1) for i in indices]

    def samples_path(self, ckpt: str) -> str:
        """Return the sampling configs file, which sample() writes after the images."""
        return os.path.join(self.ckpt_eval_dir(ckpt), "samples.json")

    def image_path(self, ckpt: str, index: int) -> str:
        return os.path.join(self.ckpt_eval_dir(ckpt), "images", f"{index:05d}.png")

    def load_metrics(self, ckpt: str) -> dict:
        if not os.path.exists(self.metrics_path(ckpt)):
            return {}
        with open(self.metrics_path(ckpt), "r", encoding="utf-8") as f:
            return json.load(f)

    def pending_rewards(self, ckpt: str) -> dict:
        """Return the rewards with a headline metric missing from the metrics file."""
        metrics = self.load_metrics(ckpt)
        return {
            name: reward for name, reward in self.rewards.items()
            if any(m not in metrics for m in reward.metrics)
        }

    def metric_names(self, ckpts: list[str]) -> list[str]:
        """Return the headline metrics of all scored rewards, in the order of the runs."""
        names = []
        for ckpt in ckpts:  # find the rewards of earlier runs by their names in the keys
            for key in self.load_metrics(ckpt):
                try:
                    metrics = registry.get("reward", key).metrics
                except KeyError:  # not a reward name, for example a sub-metric
                    continue
                names += [m for m in metrics if m not in names]
        current = [m for reward in self.rewards.values() for m in reward.metrics]
        return names + [m for m in current if m not in names]

    def refresh_summary(self, ckpts: list[str]) -> None:
        """Aggregate the reward metrics of all evaluated checkpoints."""
        if self.rank == 0:
            metric_plots = [
                {"key": k, "std_key": None, "name": k, "label": f"{k} ↑",
                 "lower_is_better": False}
                for k in self.metric_names(ckpts)
            ]
            refresh_summary(
                ckpt_metrics=[(parse_train_state(c), c, self.metrics_path(c)) for c in ckpts],
                out_dir=self.eval_dir,
                parse=lambda metrics: metrics,
                metric_plots=metric_plots,
            )
        dist.barrier()

    def is_evaluated(self, ckpt: str) -> bool:
        return not self.pending_rewards(ckpt)

    def find_samples(self, ckpt: str) -> bool:
        """Return True if all images of the checkpoint exist with these configs."""
        found, changed = False, []
        sample_dir = self.ckpt_eval_dir(ckpt)
        if self.rank == 0 and os.path.exists(self.samples_path(ckpt)):
            with open(self.samples_path(ckpt), "r", encoding="utf-8") as f:
                saved = json.load(f)
            changed = [k for k, v in self.sample_config.items() if saved.get(k) != v]
            missing = [
                i for i in range(self.num_prompts)
                if not os.path.exists(self.image_path(ckpt, i))
            ]
            if changed:
                print(f"❌ Images in <{sample_dir}> were sampled with other configs")
            elif missing:
                print(f"⚠️ WARNING: {len(missing)}/{self.num_prompts} images missing "
                      f"in <{sample_dir}>, sampling again")
                os.remove(self.samples_path(ckpt))
            else:
                found = True
                print(f"✅ Found {self.num_prompts} images in <{sample_dir}>, "
                      "skipping sampling")
        found, changed = broadcast_from_rank0((found, changed))
        if changed:
            raise ValueError(
                f"❌ Saved images do not match the configs {changed}; "
                "set evaluator.new_run=true to evaluate in a new dir"
            )
        return found

    @torch.no_grad()
    def sample(self, ckpt: str) -> list:
        """Sample the test prompts. Return uint8 images and save them as PNG files."""
        image_dir = os.path.dirname(self.image_path(ckpt, 0))
        os.makedirs(self.ckpt_eval_dir(ckpt), exist_ok=True)

        # Every checkpoint starts from the same noise.
        set_seed(self.global_seed + self.rank)
        print(f"⏳ Sampling {self.num_prompts} test prompts")
        pbar = TqdmBar(total=len(self.batches), desc="Sampling", unit="batch")
        responses, futures = [], []
        with ThreadPoolExecutor(self.num_workers) as pool:
            for indices in self.batches:
                prompts = [self.prompts[i] for i in self.unpad(indices)]
                response = self.algorithm.generate(
                    prompts, self.num_steps, self.guidance_scale
                )
                if torch.is_tensor(response):
                    # Score the same uint8 pixels as the saved images, so reruns match.
                    response = (response * 255).round().clamp(0, 255)
                    response = response.to("cpu", torch.uint8)
                    os.makedirs(image_dir, exist_ok=True)
                    array = response.permute(0, 2, 3, 1).numpy()  # NCHW -> NHWC
                    for index, image in zip(indices, array):
                        if index < self.num_prompts:
                            path = self.image_path(ckpt, index)
                            futures.append(pool.submit(save_png, image, path))
                else:  # for example, videos with audio, which sample() does not save
                    response = {
                        k: v.cpu() if torch.is_tensor(v) else v
                        for k, v in response.items()
                    }
                responses.append(response)
                pbar.update(1)
            for future in futures:  # raise any error of the PNG writer threads
                future.result()
        pbar.close()

        # Mark the images as complete only after every rank writes its shard.
        dist.barrier()
        if self.rank == 0 and os.path.isdir(image_dir):
            with open(self.samples_path(ckpt), "w", encoding="utf-8") as f:
                json.dump(self.sample_config, f, indent=4)
            print(f"💾 Saved {self.num_prompts} images to <{image_dir}>")
        return responses

    def load_samples(self, ckpt: str) -> list:
        """Read the saved images of this rank, in the same batches as sample()."""
        responses = []
        with ThreadPoolExecutor(self.num_workers) as pool:
            for indices in self.batches:
                paths = [self.image_path(ckpt, i) for i in self.unpad(indices)]
                array = np.stack(list(pool.map(load_png, paths)))
                response = torch.from_numpy(array).permute(0, 3, 1, 2)  # NHWC -> NCHW
                responses.append(response.contiguous())
        return responses

    @torch.no_grad()
    def evaluate(self, ckpt: str) -> None:
        """Sample the test prompts with one checkpoint if needed, then score them."""
        rewards = self.pending_rewards(ckpt)
        if self.find_samples(ckpt):
            responses = self.load_samples(ckpt)
        else:
            responses = self.sample(ckpt)
            # Release the cached sampling memory. The reward servers share these GPUs.
            torch.cuda.empty_cache()

        print(f"⏳ Scoring with {', '.join(rewards)}")
        pbar = TqdmBar(total=len(responses), desc="Scoring", unit="batch")
        values = defaultdict(list)
        for indices, response in zip(self.batches, responses):
            prompts = [self.prompts[i] for i in self.unpad(indices)]
            metadatas = [self.metadatas[i] for i in self.unpad(indices)]
            if torch.is_tensor(response):
                response = response.to(self.device, torch.float32) / 255
            for reward in rewards.values():
                scores = reward(response, prompts, metadatas, only_strict=False)
                for key, value in scores.items():
                    values[key] += list(value)
            values["index"] += indices
            pbar.update(1)
        pbar.close()

        # Gather all ranks, drop the padding, and average each metric over its prompts.
        gathered = {}
        for key, value in values.items():
            tensor = torch.as_tensor(value, dtype=torch.float64, device=self.device)
            gathered[key] = gather_tensor(tensor).cpu()
        if self.rank == 0:
            valid = gathered.pop("index") < self.num_prompts
            metrics = self.load_metrics(ckpt)
            for key, value in gathered.items():
                value = value[valid]
                value = value[value != -10]
                metrics[key] = value.mean().item() if len(value) else None
            with open(self.metrics_path(ckpt), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            names = [m for reward in rewards.values() for m in reward.metrics]
            print("📊 " + "  ".join(f"{k}: {metrics[k]:.4f}" for k in names))
            print(f"💾 Saved metrics to <{self.metrics_path(ckpt)}>")
        dist.barrier()
