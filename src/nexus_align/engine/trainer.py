"""Generic training loop."""

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
            "total_step": algo.meters.total_step,
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
                if algo.meters.total_step > algo.max_train_total_steps:
                    self.logger.close_log()
                    return

                algo.meters.start("step")
                train_state = {
                    "epoch": algo.meters.epoch,
                    "step": algo.meters.step,
                    "total_step": algo.meters.total_step,
                }
                data.update(train_state)

                epoch_f = f"Epoch {algo.meters.epoch}/{algo.num_epochs}"
                step_f = f"Step {algo.meters.step}/{len(self.train_dataloader)}"
                total_step_f = f"Total step {algo.meters.total_step}/{algo.max_train_total_steps}"
                print(f"\n{'=' * 80}\n🚀  {epoch_f}  {step_f}  {total_step_f}\n{'=' * 80}")

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
                        "total_step": algo.meters.total_step,
                    },
                )

                self.logger.log_all_meters(algo.meters)

            algo.meters.end("epoch")
            self.logger.log_all_meters(algo.meters)

        self.logger.close_log()
