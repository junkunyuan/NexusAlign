"""Base model: generic model construction, FSDP2 setup, and EMA maintenance."""

from copy import deepcopy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.utils._pytree import tree_flatten, tree_map_only

from nexus_align.engine.fsdp import fsdp_wrap
from nexus_align.utils.dtype import DTYPE_MAP


class BaseModel(ABC):
    """
    Base model: builds a model and an optional EMA copy, and shards them with FSDP2.

    - Subclasses implement build_model() and wrap_modules().
    - __init__ builds the module, broadcasts the rank-0 weights, and creates the EMA.
    - build_keys selects the modules to build. The other modules stay None.
    - self.model is the sharded trainable module.
    - self.ema is the sharded frozen EMA module.
    """

    def __init__(self, cfg_model, build_keys: tuple[str, ...] = ("model", "ema")) -> None:
        self.cfg_model = cfg_model
        self.device = torch.device("cuda", torch.cuda.current_device())

        # ----------------------------------------
        # Build the model and the EMA model.
        # ----------------------------------------
        # Build the model.
        model = self.build_model()
        model_dtype = DTYPE_MAP[cfg_model.get("dtype", "fp32")]
        model = model.to(dtype=model_dtype, device=self.device)
        # Autocast casts Linear and Conv weights to its dtype at every call. Frozen
        # weights that already have this dtype give the same results without the casts.
        frozen_dtype = cfg_model.get("frozen_dtype")
        if frozen_dtype is not None:
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    for p in module.parameters(recurse=False):
                        if not p.requires_grad:
                            p.data = p.data.to(DTYPE_MAP[frozen_dtype])

        # Broadcast the rank-0 weights, so that all ranks start with the same weights.
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor.data, src=0)

        # Build the EMA model. If build_keys has no "model", reuse the built module.
        ema = None
        if cfg_model.get("ema_enabled", False) and "ema" in build_keys:
            ema = deepcopy(model) if "model" in build_keys else model
            ema.requires_grad_(False)

        # ----------------------------------------
        # Wrap the modules with FSDP.
        # ----------------------------------------
        wrap_modules = self.wrap_modules()
        fsdp_configs = dict(self.cfg_model.fsdp)
        replicate = fsdp_configs.pop("replicate", 1)
        shard = fsdp_configs.pop("shard", -1)
        self.fsdp_param_dtype = DTYPE_MAP[fsdp_configs["param_dtype"]]
        # FSDP casts the frozen weights back to param_dtype. To keep their dtype, give
        # FSDP no param_dtype, and cast the inputs of each FSDP unit to param_dtype.
        input_dtype = None
        if frozen_dtype is not None:
            input_dtype, fsdp_configs["param_dtype"] = self.fsdp_param_dtype, None
        self.model = None
        if "model" in build_keys:
            self.model = fsdp_wrap(
                module=model,
                replicate=replicate,
                wrap_modules=wrap_modules,
                shard=shard,
                model_name="model",
                **fsdp_configs,
            )
            self.model.train()
        if ema is not None:
            ema = fsdp_wrap(
                module=ema,
                wrap_modules=wrap_modules,
                replicate=replicate,
                shard=shard,
                model_name="ema",
                **{**fsdp_configs, "activation_ckpt": False},
            )
            ema.eval()
        self.ema = ema

        # ----------------------------------------
        # Cast the inputs with forward pre-hooks, and compile the modules.
        # ----------------------------------------
        def cast_inputs(module, args, kwargs):
            cast = lambda x: x.to(input_dtype) if x.is_floating_point() else x
            return tree_map_only(torch.Tensor, cast, (args, kwargs))

        def mark_batch_dynamic(module, args, kwargs):
            # Filtered batches, for example in GRPO, have different sizes. A dynamic
            # batch dimension prevents recompiles.
            for x in tree_flatten((args, kwargs))[0]:
                if isinstance(x, torch.Tensor) and x.dim() > 0:
                    torch._dynamo.maybe_mark_dynamic(x, 0)

        # Compile each wrapped module. Its FSDP hooks stay outside the compiled graph.
        compile_enabled = cfg_model.get("compile", False)
        if compile_enabled:
            # Each block variant, grad mode, and LoRA state (on or off) compiles its own
            # graph.
            torch._dynamo.config.recompile_limit = 64
        for module in (self.model, self.ema):
            for submodule in module.modules() if module is not None else []:
                if input_dtype is not None and isinstance(submodule, FSDPModule):
                    submodule.register_forward_pre_hook(cast_inputs, with_kwargs=True)
                if compile_enabled and isinstance(submodule, wrap_modules):
                    submodule.register_forward_pre_hook(
                        mark_batch_dynamic, with_kwargs=True
                    )
                    submodule.forward = torch.compile(submodule.forward)

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model module, before FSDP shards it."""
        ...

    @abstractmethod
    def wrap_modules(self) -> tuple:
        """Return the module classes that FSDP shards as separate units."""
        ...

    @torch.no_grad()
    def ema_step(self, decay: float) -> None:
        """Move the EMA parameters toward the model. Both have the same sharding."""
        for ema_p, p in zip(self.ema.parameters(), self.model.parameters()):
            ema_p.mul_(decay).add_(p.data, alpha=1 - decay)
