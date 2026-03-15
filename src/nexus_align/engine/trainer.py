"""Generic training loop."""

import os

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from nexus_align.engine.logger import TrainingLogger
from nexus_align.core.base_algorithm import BaseAlgorithm
from nexus_align.engine.checkpoint import CheckpointManager


class Trainer:
    """
    Runs the training loop.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        algorithm: BaseAlgorithm,
        cfg: dict,
        val_dataset: Dataset | None = None,
    ) -> None:
        self.algorithm = algorithm

        self.checkpoint_manager = CheckpointManager(
            kwargs=cfg["ckpt_manager"],
            fsdp=cfg["model"].get("fsdp", None) is not None
        )

        self.train_dataloader = self._build_dataloader(
            dataset=train_dataset, 
            shuffle=cfg["data"]["load"]["shuffle"],
            seed=cfg["common"]["seed"],
            batch_size=cfg["algorithm"]["train"]["train_batch_size"],
            num_workers=cfg["data"]["load"]["num_workers"],
            drop_last=cfg["data"]["load"]["drop_last"],
            pin_memory=cfg["data"]["load"]["pin_memory"]
        )

        self.logger = TrainingLogger(
            log_dir=cfg["log"]["log_dir"],
            wandb_init=cfg["log"]["wandb"]["wandb_init"],
        )

        # Validation setup
        val_cfg = cfg.get("validation", {})
        self.val_enabled = val_dataset is not None
        if self.val_enabled:
            sample_batch_size = cfg["algorithm"]["run"]["sample_batch_size"]
            self.val_dataloader = self._build_dataloader(
                dataset=val_dataset,
                shuffle=False,
                seed=cfg["common"]["seed"],
                batch_size=sample_batch_size,
                num_workers=cfg["data"]["load"]["num_workers"],
                drop_last=True,
                pin_memory=cfg["data"]["load"]["pin_memory"],
            )
            self.val_every_n_steps = val_cfg.get("val_every_n_steps", 50)

    def _build_dataloader(
        self, 
        dataset: Dataset, 
        shuffle: bool,
        seed: int,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
        pin_memory: bool
    ) -> DataLoader:
        """Build distributed sampler and train dataloader."""
        sampler = DistributedSampler(
            dataset=dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=shuffle,
            seed=seed,
        )
        loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=getattr(dataset, "collate_fn", None),
            pin_memory=pin_memory,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return loader

    @torch.no_grad()
    def validate(self) -> None:
        """
        Run validation: generate images on val set and score with reward model.

        Reuses the training pipeline (algo.pipeline) to generate images with current
        model weights, then scores them with the training reward model (algo.reward_model).
        """
        algo = self.algorithm
        pipeline = algo.pipeline
        total_steps = algo.meters.total_steps

        print(f"\n{'~' * 80}\n🔍 Validation at total step {total_steps}\n{'~' * 80}")

        # Redirect sample saves to a validation subdirectory
        original_save_dir = pipeline.sample_save_dir
        val_save_dir = os.path.join(original_save_dir, "validation")
        os.makedirs(val_save_dir, exist_ok=True)
        pipeline.sample_save_dir = val_save_dir

        all_scores = []
        for i, data in enumerate(self.val_dataloader, start=1):
            print(f"🔍 Validation batch: {i} / {len(self.val_dataloader)}")

            data.update({"epoch": 0, "step": 0, "total_step": total_steps})

            data = pipeline.prepare_data(data)
            data = pipeline.sample_responses(data)

            score_list = algo.reward_model.evaluate(
                data=data["reward_inputs"],
                return_tensor=False,
            )
            scores = torch.tensor(
                [s if s is not None else 0.0 for s in score_list],
                device=algo.device,
                dtype=torch.float32,
            )
            all_scores.append(scores)

            torch.cuda.empty_cache()

        pipeline.sample_save_dir = original_save_dir

        all_scores = torch.cat(all_scores)
        gathered = [torch.empty_like(all_scores) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, all_scores)
        gathered = torch.cat(gathered)

        mean = gathered.mean().item()
        std = gathered.std().item()

        dec = 4 if mean > 1 else 6
        print(
            f"🔍 Validation: {len(gathered)} scores, "
            f"mean ± std: {mean:.{dec}f} ± {std:.{dec}f}"
        )

        self.logger.to_logger("val_reward_mean", mean, total_steps)
        self.logger.to_logger("val_reward_std", std, total_steps)

        dist.barrier()
        torch.cuda.empty_cache()

    def train(self) -> None:
        """
        Run the training loop.
        data -> (rollout) -> (reward) -> (advantage) -> train_one_step -> checkpoint -> log.
        """
        algo = self.algorithm

        # Load checkpoint (updates model, optimizer, lr_scheduler, train_state)
        train_state = {
            "epoch": algo.meters.epoch,
            "step": algo.meters.step,
            "total_steps": algo.meters.total_steps,
        }
        model_state_dict = {
            "model": algo.model,
            "optimizer": algo.optimizer,
            "lr_scheduler": algo.lr_scheduler,
            "train_state": train_state,
        }
        self.checkpoint_manager.load_checkpoint(model_state_dict=model_state_dict)
        algo.meters.update_train_state(train_state)
        
        self.logger.open_log()

        # Baseline validation before training starts
        if self.val_enabled:
            self.validate()

        # Start training
        print(f"🚀 Training started!")
        for epoch in range(algo.num_epochs):
            if epoch < algo.meters.epoch:
                continue

            print(f"\n{'=' * 80}\n🚀 Epoch {epoch}/{algo.num_epochs}\n{'=' * 80}")
            self.train_dataloader.sampler.set_epoch(epoch)
            algo.meters.start("epoch")

            for step, data in enumerate(self.train_dataloader):
                if step < algo.meters.step:
                    continue
                if algo.meters.total_steps > algo.max_train_total_steps:
                    self.logger.close_log()
                    return

                algo.meters.start("step")
                train_state = {
                    "epoch": algo.meters.epoch,
                    "step": algo.meters.step,
                    "total_step": algo.meters.total_steps,
                }
                data.update(train_state)

                epoch_f = f"Epoch {algo.meters.epoch}/{algo.num_epochs}"
                step_f = f"Step {algo.meters.step}/{len(self.train_dataloader)}"
                total_steps_f = f"Total step {algo.meters.total_steps}/{algo.max_train_total_steps}"
                print(f"\n{'=' * 80}\n🚀  {epoch_f}  {step_f}  {total_steps_f}\n{'=' * 80}")

                print(f"➡️  Preparing data")
                data = algo.prepare_data(data)

                print(f"➡️  Sampling responses (rollout)")
                data = algo.sample_responses(data)
                
                print(f"➡️  Computing rewards")
                data = algo.compute_rewards(data)
                
                print(f"➡️  Computing advantages")
                data = algo.compute_advantages(data)

                print(f"➡️  Updating model")
                algo.train_one_step(data)

                algo.meters.end("step")

                self.checkpoint_manager.save_checkpoint(
                    model_state_dict={
                        "model": algo.model,
                        "optimizer": algo.optimizer,
                        "lr_scheduler": algo.lr_scheduler,
                    },
                    train_state={
                        "epoch": algo.meters.epoch,
                        "step": algo.meters.step,
                        "total_steps": algo.meters.total_steps,
                    },
                )

                self.logger.log_all_meters(algo.meters)

                # Periodic validation
                if self.val_enabled and algo.meters.total_steps % self.val_every_n_steps == 0:
                    self.validate()

            algo.meters.end("epoch")
            self.logger.log_all_meters(algo.meters)

        self.logger.close_log()
