"""Checkpoint save and load (model, optimizer, lr_scheduler, train_state)."""

import os
import time
import shutil
import json
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


def get_fsdp_context(model: torch.nn.Module, fsdp: bool):
    """Get FSDP context manager for state_dict save/load."""
    if fsdp:
        return FSDP.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            ),
        )
    return nullcontext()


class CheckpointManager:
    """Saves and loads training checkpoints under a configurable schedule."""

    def __init__(self, kwargs: dict, fsdp: bool = True) -> None:
        self.save_ckpt_every_n_steps = kwargs["save_ckpt_every_n_steps"]
        self.save_ckpt_keep_last_n_steps = kwargs["save_ckpt_keep_last_n_steps"]
        self.save_ckpt_every_n_epochs = kwargs["save_ckpt_every_n_epochs"]
        self.save_ckpt_keep_last_n_epochs = kwargs["save_ckpt_keep_last_n_epochs"]
        self.save_ckpt_only_model = kwargs["save_ckpt_only_model"]
        self.save_final_checkpoint = kwargs.get("save_final_checkpoint", True)

        self.save_ckpt_root = kwargs["save_ckpt_root"]
        os.makedirs(self.save_ckpt_root, exist_ok=True)

        self.saved_step_paths = []
        self.saved_epoch_paths = []

        self.load_ckpt_path = kwargs["load_ckpt_path"]
        self.load_ckpt_only_model = kwargs["load_ckpt_only_model"]

        self.fsdp = fsdp

    def save_checkpoint(
        self, model_state_dict: dict, train_state: dict, force: bool = False
    ) -> None:
        """
        Save checkpoint to `self.save_ckpt_root`.

        Args:
            model_state_dict (dict):
                - "model" (`torch.nn.Module`)
                - "optimizer" (`torch.optim.Optimizer` | None)
                - "lr_scheduler" (`torch.optim.lr_scheduler.LRScheduler` | None)
            train_state (dict):
                - "epoch" (`int`)
                - "step" (`int`)
                - "total_step" (`int`)
            force (bool): If True, save regardless of schedule (e.g. end of training).
        """
        step = train_state["step"]
        epoch = train_state["epoch"]
        total_step = train_state["total_step"]

        to_save_step = to_save_epoch = False
        if isinstance(self.save_ckpt_every_n_steps, int):
            if (step + 1) % self.save_ckpt_every_n_steps == 0:
                to_save_step = True
        if force:
            to_save_step = True

        to_remove_step = len(self.saved_step_paths) >= self.save_ckpt_keep_last_n_steps
        to_remove_epoch = len(self.saved_epoch_paths) >= self.save_ckpt_keep_last_n_epochs
        if to_remove_step or to_remove_epoch:
            start_time = time.time()
            while len(self.saved_step_paths) > self.save_ckpt_keep_last_n_steps:
                shutil.rmtree(self.saved_step_paths.pop(0), ignore_errors=True)
            while len(self.saved_epoch_paths) > self.save_ckpt_keep_last_n_epochs:
                shutil.rmtree(self.saved_epoch_paths.pop(0), ignore_errors=True)
            end_time = time.time()
            print(f"✅ Removed old checkpoints in {end_time - start_time:.1f}s")

        if to_save_step or to_save_epoch:
            save_dir = f"ckpt-epoch_{epoch}-step_{step}-total_step_{total_step}"
            save_path = os.path.join(self.save_ckpt_root, save_dir)
            os.makedirs(save_path, exist_ok=True)
            success_save_all = True

            model = model_state_dict["model"]
            model_path = os.path.join(save_path, "model.pt")
            try:
                start_time = time.time()
                with get_fsdp_context(model, self.fsdp):
                    state_dict = model.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state_dict, model_path)
                cost = time.time() - start_time
                print(f"💾 Saved model to <{model_path}> in {cost:.1f}s")
            except Exception as e:
                print(f"❌ Failed to save model to <{model_path}>: {e}")
                success_save_all = False

            train_state_path = os.path.join(save_path, "train_state.json")
            try:
                start_time = time.time()
                if dist.get_rank() == 0:
                    with open(train_state_path, "w") as f:
                        json.dump(train_state, f, indent=4, sort_keys=True)
                cost = time.time() - start_time
                print(f"💾 Saved train_state to <{train_state_path}> in {cost:.1f}s")
            except Exception as e:
                print(f"❌ Failed to save train_state to <{train_state_path}>: {e}")
                success_save_all = False

            if not self.save_ckpt_only_model:
                optimizer = model_state_dict.get("optimizer", None)
                optimizer_path = os.path.join(save_path, "optimizer.pt")
                if optimizer is not None:
                    try:
                        start_time = time.time()
                        optim_state_dict = optimizer.state_dict()
                        if self.fsdp:
                            optim_state = FSDP.optim_state_dict(
                                model, optimizer, optim_state_dict=optim_state_dict
                            )
                        else:
                            optim_state = optim_state_dict
                        if dist.get_rank() == 0:
                            torch.save(optim_state, optimizer_path)
                        cost = time.time() - start_time
                        print(f"💾 Saved optimizer to <{optimizer_path}> in {cost:.1f}s")
                    except Exception as e:
                        print(f"❌ Failed to save optimizer to <{optimizer_path}>: {e}")
                        success_save_all = False

                lr_scheduler = model_state_dict.get("lr_scheduler", None)
                lr_scheduler_path = os.path.join(save_path, "lr_scheduler.pt")
                if lr_scheduler is not None:
                    try:
                        start_time = time.time()
                        if dist.get_rank() == 0:
                            torch.save(lr_scheduler.state_dict(), lr_scheduler_path)
                        cost = time.time() - start_time
                        print(f"💾 Saved lr scheduler to <{lr_scheduler_path}> in {cost:.1f}s")
                    except Exception as e:
                        print(f"❌ Failed to save lr scheduler to <{lr_scheduler_path}>: {e}")
                        success_save_all = False

            if success_save_all:
                if to_save_step:
                    self.saved_step_paths.append(save_path)
                if to_save_epoch:
                    self.saved_epoch_paths.append(save_path)
                print(f"✅ Saved checkpoint to <{save_path}>")
            else:
                print(f"❌ Failed to save all checkpoint components to <{save_path}>")

    def save_epoch_checkpoint(self, model_state_dict: dict, train_state: dict) -> None:
        """Save checkpoint at the end of an epoch if the epoch-based schedule matches.

        Note: this is called AFTER meters.end("epoch"), so train_state["epoch"] is
        already incremented (e.g. epoch=1 means epoch 0 just finished).
        """
        epoch = train_state["epoch"]
        if not isinstance(self.save_ckpt_every_n_epochs, int):
            return
        if epoch % self.save_ckpt_every_n_epochs != 0:
            return
        self.save_checkpoint(model_state_dict, train_state, force=True)

    def load_checkpoint(self, model_state_dict: dict) -> None:
        """
        Load checkpoint from `self.load_ckpt_path`.

        Args:
            model_state_dict (dict):
                - "model" (`torch.nn.Module`)
                - "train_state" (`dict`)
                - "optimizer" (`torch.optim.Optimizer` | None)
                - "lr_scheduler" (`torch.optim.lr_scheduler.LRScheduler` | None)
        """
        if not isinstance(self.load_ckpt_path, str):
            return
        if not os.path.exists(self.load_ckpt_path):
            return

        model = model_state_dict.get("model", None)
        model_path = os.path.join(self.load_ckpt_path, "model.pt")
        if isinstance(model, torch.nn.Module) and os.path.exists(model_path):
            try:
                start_time = time.time()
                state_dict = torch.load(
                    model_path, map_location="cpu", weights_only=True
                )
                with get_fsdp_context(model, self.fsdp):
                    model.load_state_dict(state_dict)
                cost = time.time() - start_time
                print(f"✅ Loaded model from <{model_path}> in {cost:.1f}s")
            except Exception as e:
                print(f"❌ Failed to load model from <{model_path}>: {e}")

        train_state = model_state_dict.get("train_state", None)
        train_state_path = os.path.join(self.load_ckpt_path, "train_state.json")
        if train_state is not None and os.path.exists(train_state_path):
            try:
                start_time = time.time()
                with open(train_state_path, "r") as f:
                    loaded_train_state = json.load(f)
                if loaded_train_state is not None:
                    train_state.update(
                        {k: v for k, v in loaded_train_state.items() if k in train_state}
                    )
                cost = time.time() - start_time
                print(f"✅ Loaded train state from <{train_state_path}> in {cost:.1f}s")
            except Exception as e:
                print(f"❌ Failed to load train state from <{train_state_path}>: {e}")

        if not self.load_ckpt_only_model:
            optimizer = model_state_dict.get("optimizer", None)
            optimizer_path = os.path.join(self.load_ckpt_path, "optimizer.pt")
            if optimizer is not None and os.path.exists(optimizer_path):
                try:
                    start_time = time.time()
                    optim_state_dict = torch.load(
                        optimizer_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    if self.fsdp:
                        optim_state_dict = FSDP.optim_state_dict_to_load(
                            model=model,
                            optim=optimizer,
                            optim_state_dict=optim_state_dict,
                        )
                    optimizer.load_state_dict(optim_state_dict)
                    cost = time.time() - start_time
                    print(f"✅ Loaded optimizer from <{optimizer_path}> in {cost:.1f}s")
                except Exception as e:
                    print(f"❌ Failed to load optimizer from <{optimizer_path}>: {e}")

            lr_scheduler = model_state_dict.get("lr_scheduler", None)
            lr_scheduler_path = os.path.join(self.load_ckpt_path, "lr_scheduler.pt")
            if lr_scheduler is not None and os.path.exists(lr_scheduler_path):
                try:
                    start_time = time.time()
                    lr_scheduler.load_state_dict(
                        torch.load(
                            lr_scheduler_path,
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
                    cost = time.time() - start_time
                    print(f"✅ Loaded lr scheduler from <{lr_scheduler_path}> in {cost:.1f}s")
                except Exception as e:
                    print(f"❌ Failed to load lr scheduler from <{lr_scheduler_path}>: {e}")
