"""CLI entry: Hydra-configured entry point with environment setup."""

import os

import hydra
from omegaconf import open_dict

from nexus_align.registry import registry
from nexus_align.engine.setup import prepare_env


def prepare_reward(cfg) -> dict:
    """Build the configured reward as a keyword argument; empty when there is none."""
    if cfg.get("reward") is None:
        return {}
    reward = registry.get("reward", cfg.reward.name)(cfg_reward=cfg.reward)
    print("✅ Prepared reward")
    return {"reward": reward}


def prepare_stage_datasets(cfg) -> dict:
    """Build the datasets of training stages as a keyword argument; empty if none."""
    if cfg.trainer.get("stages") is None:
        return {}
    stage_datasets = {
        name: registry.get("dataset", cfg_data.name)(cfg_data=cfg_data)
        for name, cfg_data in cfg.stage_data.items()
    }
    print(f"✅ Prepared stage datasets: {', '.join(stage_datasets)}")
    return {"stage_datasets": stage_datasets}


def prepare_rewards(cfg) -> dict:
    """Build the rewards to evaluate, keyed by name; empty when there are none."""
    if cfg.get("rewards") is None:
        return {}
    rewards = {}
    for name, cfg_reward in cfg.rewards.items():
        with open_dict(cfg_reward):
            cfg_reward.normalize = False  # report the raw scores, as papers do
        rewards[name] = registry.get("reward", cfg_reward.name)(cfg_reward=cfg_reward)
    print(f"✅ Prepared rewards: {', '.join(rewards)}")
    return {"rewards": rewards}


def train(cfg) -> None:
    """Run training."""
    # 1. Prepare dataset.
    train_dataset = registry.get("dataset", cfg.data.name)(cfg_data=cfg.data)
    print("✅ Prepared training dataset")

    # 2. Prepare model.
    model = registry.get("model", cfg.model.name)(cfg_model=cfg.model)
    print("✅ Prepared model")

    # 3. Prepare algorithm.
    algorithm = registry.get("algorithm", cfg.algorithm.name)(cfg.algorithm)
    print("✅ Prepared algorithm")

    # 4. Prepare trainer, with the reward of online RL and the datasets of training
    # stages if they are configured.
    trainer = registry.get("trainer", cfg.trainer.name)(
        cfg_trainer=cfg.trainer,
        train_dataset=train_dataset,
        model=model,
        algorithm=algorithm,
        log_dir=cfg.log_dir,
        **prepare_reward(cfg),
        **prepare_stage_datasets(cfg),
    )
    print("✅ Prepared trainer")

    # 5. Run training.
    trainer.run()
    print("✅ Training completed")


def evaluate(cfg) -> None:
    """Run evaluation."""
    # 1. Prepare model; only the weights the evaluator samples with are built.
    model = registry.get("model", cfg.model.name)(
        cfg_model=cfg.model,
        build_keys=(cfg.evaluator.model_key,),
    )
    print("✅ Prepared model")

    # 2. Prepare algorithm.
    algorithm = registry.get("algorithm", cfg.algorithm.name)(cfg.algorithm)
    print("✅ Prepared algorithm")

    # 3. Prepare evaluator, with the rewards to score if any are configured.
    evaluator = registry.get("evaluator", cfg.evaluator.name)(
        cfg_evaluator=cfg.evaluator,
        model=model,
        algorithm=algorithm,
        log_dir=cfg.log_dir,
        **prepare_rewards(cfg),
    )
    print("✅ Prepared evaluator")

    # 4. Run evaluation.
    evaluator.run()
    print("✅ Evaluation completed")


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
def main(cfg):
    # 0. Prepare environment.
    prepare_env(cfg)

    # 1. Run the job selected by job_type.
    if cfg.job_type == "train":
        train(cfg)
    elif cfg.job_type == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"❌ Unknown job type: {cfg.job_type}")


if __name__ == "__main__":
    main()
