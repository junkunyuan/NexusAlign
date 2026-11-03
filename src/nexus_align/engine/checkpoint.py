"""Checkpoint: helpers to save and load the training state."""

import os
import json
import shutil
from functools import partial

import torch
import torch.distributed as dist

from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
    get_model_state_dict,
    set_optimizer_state_dict,
    get_optimizer_state_dict
)

from nexus_align.engine.distributed import broadcast_from_rank0


# Only rank 0 reads and writes checkpoint files. On multi-node jobs with local disks, it
# is the only rank that can see them. set_model_state_dict and set_optimizer_state_dict
# broadcast the model, ema, and optimizer states. broadcast_from_rank0 broadcasts all
# other states.
BROADCAST_KEYS = ("model", "ema", "optimizer")


class CheckpointManager:
    """Manage directory-based training checkpoints."""

    def __init__(
        self,
        save_ckpt_dir: str,
        load_ckpt_dir: str | None = None,
        save_ckpt_every_n_epochs: int | None = None,
        save_ckpt_keep_last_n_epochs: int = 10,
        save_ckpt_every_n_steps: int | None = None,
        save_ckpt_keep_last_n_steps: int = 10
    ) -> None:
        self.save_ckpt_dir = save_ckpt_dir
        self.load_ckpt_dir = load_ckpt_dir
        self.save_ckpt_every_n_epochs = save_ckpt_every_n_epochs
        self.save_ckpt_keep_last_n_epochs = save_ckpt_keep_last_n_epochs
        self.save_ckpt_every_n_steps = save_ckpt_every_n_steps
        self.save_ckpt_keep_last_n_steps = save_ckpt_keep_last_n_steps
        self.saved_epoch_paths = []
        self.saved_step_paths = []

        if dist.get_rank() == 0:
            os.makedirs(self.save_ckpt_dir, exist_ok=True)
        dist.barrier()

    def load_ckpt(self, trainer) -> None:
        """Load the checkpoint at load_ckpt_dir into the trainer."""
        if self.load_ckpt_dir is None:
            return None

        print(f"⏳ Loading checkpoints from <{self.load_ckpt_dir}>")
        is_rank0 = dist.get_rank() == 0
        if not broadcast_from_rank0(is_rank0 and os.path.exists(self.load_ckpt_dir)):
            raise FileNotFoundError(f"❌ Checkpoint dir not found: <{self.load_ckpt_dir}>")
        for key in trainer.load_ckpt_keys:
            # Load the runtime state.
            if key == "train_state":
                path = os.path.join(self.load_ckpt_dir, "train_state.json")
                train_state = None
                if is_rank0 and os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        train_state = json.load(f)
                train_state = broadcast_from_rank0(train_state)
                if train_state is None:
                    raise FileNotFoundError(f"❌ Missing train_state in checkpoint: <{path}>")
                trainer.load_state_dict(train_state)
                print(f"✅ Loaded train_state from <{path}>")
                continue

            # Skip the components that the trainer does not build, for example a
            # disabled EMA or a missing lr_scheduler.
            owner = trainer.model if key in ["model", "ema"] else trainer
            component = getattr(owner, key, None)
            if component is None:
                continue
            if key == "algorithm" and not hasattr(component, "load_state_dict"):
                continue
            path = os.path.join(self.load_ckpt_dir, key, "state.pt")
            exists = broadcast_from_rank0(is_rank0 and os.path.exists(path))
            if key == "algorithm" and not exists:
                continue
            if not exists:
                raise FileNotFoundError(
                    f"❌ Missing {key} in checkpoint: <{path}>; "
                    f"set trainer.load_ckpt_only_model=true to load only model and ema"
                )

            # Load the model, ema, optimizer, lr_scheduler, scaler, or algorithm.
            loaded_state_dict = {} if not is_rank0 else torch.load(
                path, map_location="cpu", weights_only=False
            )
            if key not in BROADCAST_KEYS:
                loaded_state_dict = broadcast_from_rank0(loaded_state_dict)
            if key in ["model", "ema"]:
                # A forward pass under shard_grad_op leaves the parameters gathered as
                # plain tensors. Reshard them, so that the load copies the state dict
                # into DTensors.
                for module in component.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                trainable_only = getattr(trainer.model, "save_trainable_only", False)
                options = StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=not trainable_only
                )
                result = set_model_state_dict(
                    model=component,
                    model_state_dict=loaded_state_dict,
                    options=options
                )
                if result.missing_keys and not trainable_only:
                    print(f"⚠️ WARNING: model missing keys: {result.missing_keys}")
                if result.unexpected_keys:
                    print(
                        f"⚠️ WARNING: model unexpected keys: {result.unexpected_keys}"
                    )
                print(f"✅ Loaded {key} from <{path}>")
            elif key == "optimizer":
                # The load restores the saved param_groups (lr, betas, ...). Keep the
                # current ones, so that config overrides such as trainer.learning_rate
                # apply on resume.
                hparams = [
                    {k: v for k, v in group.items() if k != "params"}
                    for group in component.param_groups
                ]
                options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
                set_optimizer_state_dict(
                    model=trainer.model.model,
                    optimizers=component,
                    optim_state_dict=loaded_state_dict,
                    options=options
                )
                for group, group_hparams in zip(component.param_groups, hparams):
                    group.update(group_hparams)
                print(f"✅ Loaded optimizer from <{path}>")
            elif key in ["lr_scheduler", "scaler", "algorithm"]:
                component.load_state_dict(loaded_state_dict)
                print(f"✅ Loaded {key} from <{path}>")

    def save_ckpt(self, trainer, epoch_finished: bool = False) -> None:
        """Save a checkpoint of the trainer if one is due."""
        train_state = trainer.state_dict()
        
        # Decide whether to save. Save only if an interval is set.
        # NOTE: Also save the last step, so that the run always keeps the final weights.
        epoch = train_state["epoch"]
        step = train_state["step"]
        total_step = train_state["total_step"]
        every_n_epochs = self.save_ckpt_every_n_epochs
        every_n_steps = self.save_ckpt_every_n_steps
        is_last = total_step >= trainer.max_total_steps
        to_save_epoch = (
            every_n_epochs is not None
            and epoch_finished
            and (epoch % every_n_epochs == 0 or is_last)
        )
        to_save_step = every_n_steps is not None and (
            total_step % every_n_steps == 0 or is_last
        )
        if not (to_save_epoch or to_save_step):
            return
        
        # Make the new checkpoint directory. Write into a temporary directory and
        # rename it when it is complete, so that an interrupted save never leaves a
        # partial checkpoint. Only rank 0 writes.
        ckpt_dir_name = f"epoch_{epoch}-step_{step}-total_step_{total_step}"
        ckpt_dir = os.path.join(self.save_ckpt_dir, ckpt_dir_name)
        if ckpt_dir in self.saved_epoch_paths + self.saved_step_paths:
            return
        tmp_dir = ckpt_dir + ".tmp"
        if dist.get_rank() == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.makedirs(tmp_dir)

        def skip_failed_ckpt(error: str) -> None:
            if dist.get_rank() == 0:
                print(f"⚠️ Skipping checkpoint <{ckpt_dir}> because save failed: {error}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
            dist.barrier()

        def write_train_state(path) -> None:
            with open(path, "w") as f:
                json.dump(train_state, f, indent=2, sort_keys=True)

        def save_on_rank0(write, path) -> bool:
            error = None
            if dist.get_rank() == 0:
                try:
                    write(path)
                    print(f"💾 Saved {key} to <{path}>")
                except Exception as e:
                    error = f"❌ Failed to save {key} to <{path}>: {e}"
                    print(error)
            error = broadcast_from_rank0(error)
            if error is not None:
                skip_failed_ckpt(error)
                return False
            return True

        # Start to save.
        print(f"⏳ Saving checkpoints to <{ckpt_dir}>")
        for key in trainer.save_ckpt_keys:
            # Save the runtime state.
            if key == "train_state":
                path = os.path.join(tmp_dir, "train_state.json")
                if not save_on_rank0(write_train_state, path):
                    return
                continue

            owner = trainer.model if key in ["model", "ema"] else trainer
            component = getattr(owner, key, None)
            if component is None:
                continue
            if key == "algorithm" and not hasattr(component, "state_dict"):
                continue
            component_dir = os.path.join(tmp_dir, key)
            if dist.get_rank() == 0:
                os.makedirs(component_dir, exist_ok=True)
            path = os.path.join(component_dir, "state.pt")

            # Save the model, ema, optimizer, lr_scheduler, scaler, or algorithm.
            if key in ["model", "ema"]:
                if getattr(trainer.model, "save_trainable_only", False):
                    # Select by the trainable names of the model, because the EMA
                    # parameters are frozen.
                    names = {
                        name for name, p in trainer.model.model.named_parameters()
                        if p.requires_grad
                    }
                    state_dict = {
                        name: p.full_tensor().cpu()
                        for name, p in component.named_parameters() if name in names
                    }
                else:
                    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                    state_dict = get_model_state_dict(component, options=options)
            elif key == "optimizer":
                options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                state_dict = get_optimizer_state_dict(
                    trainer.model.model,
                    component,
                    options=options
                )
            else:
                state_dict = component.state_dict()
            if not save_on_rank0(partial(torch.save, state_dict), path):
                return

        # Publish the complete checkpoint, then remove the oldest checkpoints.
        if dist.get_rank() == 0:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            os.rename(tmp_dir, ckpt_dir)
        dist.barrier()
        if to_save_epoch:
            self.saved_epoch_paths.append(ckpt_dir)
            saved_paths, keep_last_n = self.saved_epoch_paths, self.save_ckpt_keep_last_n_epochs
        else:
            self.saved_step_paths.append(ckpt_dir)
            saved_paths, keep_last_n = self.saved_step_paths, self.save_ckpt_keep_last_n_steps
        while len(saved_paths) > keep_last_n:
            if dist.get_rank() == 0:
                shutil.rmtree(saved_paths[0], ignore_errors=True)
            saved_paths.pop(0)

        print(f"✅ Saved checkpoint to <{ckpt_dir}>")
