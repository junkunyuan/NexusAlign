"""Training entry point."""

import os
import hydra

from nexus_align.engine.trainer import Trainer
from nexus_align.engine import with_env_setup, dist_safe_exit
from nexus_align.engine.optimizer import get_optimizer, get_lr_scheduler

from nexus_align.core.registry import registry
from nexus_align.core import validate_train_config


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="train",
    version_base="1.3",
)
@with_env_setup(validator=validate_train_config)
def main(cfg, env):
    # --------------------------------------------------------------------------------
    # 1. Prepare environment (with env from by with_env_setup decorator)
    # --------------------------------------------------------------------------------
    world_size, rank, device = env.world_size, env.rank, env.device
    cfg_dict = env.cfg_dict

    # --------------------------------------------------------------------------------
    # 2. Prepare dataset
    # --------------------------------------------------------------------------------
    train_dataset = registry.get("dataset", cfg.data.name)(cfg_dict)
    
    coef = world_size * cfg.algorithm.train.grad_accu_step
    total_train_batch_size = cfg.algorithm.train.train_batch_size * coef
    info = ["\n📚 Training data:"]
    info += [f"    dataset: {cfg.data.name}"]
    info += [f"    sample count: {len(train_dataset)}"]
    info += [f"    batchsize: {cfg.algorithm.train.train_batch_size}"]
    info += [f"    total batchsize: {total_train_batch_size}"]
    if cfg.data.load.sample_ratio != 1.:
        info += [f"    sample_ratio: {cfg.data.load.sample_ratio}"]
    if cfg.data.load.drop_last:
        info += ["    drop_last: true"]
    if cfg.data.load.cache_dir:
        info += [f"    cache_dir: {cfg.data.load.cache_dir}"]
    print("\n".join(info))
    print("✅ Prepared training dataset")

    # --------------------------------------------------------------------------------
    # 3. Prepare model, reward model, optimizer, lr_scheduler
    # --------------------------------------------------------------------------------
    model = registry.get("model", cfg.model.name)(
        device=device,
        model_dtype=cfg.model.model_dtype,
        kwargs=cfg_dict,
        env=env
    )
    reward_model = registry.get("reward_model", cfg.reward_model.name)(
        device=device,
        kwargs=cfg_dict,
    )
    optimizer = get_optimizer[cfg.algorithm.optimizer.optimizer](
        params_train=model.params_train,
        kwargs=cfg_dict,
    )
    lr_scheduler = get_lr_scheduler[cfg.algorithm.optimizer.lr_schedule](
        optimizer=optimizer,
        kwargs=cfg_dict,
    )

    # --------------------------------------------------------------------------------
    # 4. Prepare algorithm
    # --------------------------------------------------------------------------------
    algorithm = registry.get("algorithm", cfg.algorithm.name)(
        model=model,
        reward_model=reward_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        amp_dtype=cfg.model.amp_dtype,
        kwargs=cfg_dict,
    )

    # --------------------------------------------------------------------------------
    # 5. Build Trainer and run training
    # --------------------------------------------------------------------------------
    trainer = Trainer(
        train_dataset=train_dataset,
        algorithm=algorithm,
        cfg=cfg_dict,
    )
    trainer.train()
    
    if env.profiler is not None:
        print("⏱️ Time cost analysis:")
        env.profiler.stop()
        env.profiler.print()


if __name__ == "__main__":
    main()
    dist_safe_exit(exit_code=0)
