"""SFT trainer: generic FSDP training with algorithm-owned forward logic."""

from contextlib import contextmanager

import torch
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from nexus_align.models.base_model import BaseModel
from nexus_align.trainers.base_trainer import BaseTrainer
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


class SFTTrainer(BaseTrainer):
    """Generic trainer for algorithm-defined supervised fine-tuning steps."""

    def __init__(
        self,
        cfg_trainer,
        train_dataset: Dataset,
        model: BaseModel,
        algorithm: BaseAlgorithm,
        log_dir: str = "logs/temp"
    ) -> None:
        super().__init__(
            cfg_trainer=cfg_trainer,
            train_dataset=train_dataset,
            model=model,
            algorithm=algorithm,
            log_dir=log_dir
        )

        self.algorithm.prepare(model.model, model.device)

        self.optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )
        warmup_steps = getattr(self, "warmup_epochs", 0) * self.steps_per_epoch
        if warmup_steps > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: min(step / warmup_steps, 1.0)
            )
        self.scaler = ShardedGradScaler(enabled=model.fsdp_param_dtype==torch.float16)
        
    def train_mode(self):
        self.model.model.train()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def forward(self, data):
        results = self.algorithm.forward(data)

        loss = results["loss"]
        
        metrics = dict(results.get("metrics", {}))
        metrics.setdefault("loss", loss.detach())
        self._metrics = metrics
        
        return {"loss": loss}

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    @contextmanager
    def no_sync(self):
        self.model.model.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            self.model.model.set_requires_gradient_sync(True)

    def clip_grad(self):
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm_(self.model.model.parameters(), self.max_grad_norm)
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        self._grad_norm = grad_norm.detach()

    def optimizer_step(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def ema_step(self):
        if self.model.ema is not None:
            self.model.ema_step(decay=self.ema_decay)
