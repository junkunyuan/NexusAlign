"""CLI entry: Hydra-configured entry point with environment setup."""

import os

import hydra
from omegaconf import open_dict

from nexus_align.registry import registry
from nexus_align.engine.setup import prepare_env

# --------------------------------------------------------------------------------
# Reward models.
# Training takes one `reward` with its configured normalization.
# Evaluation takes a `rewards` dict keyed by name and reports the raw scores.
# If we merge them, the code needs more branches.
# --------------------------------------------------------------------------------
def prepare_reward(cfg) -> dict:
    """Return the configured reward as a keyword argument, or {} if there is none."""
    if cfg.get("reward") is None:
        return {}
    reward = registry.get("reward", cfg.reward.name)(cfg_reward=cfg.reward)
    print("✅ Prepared reward")
    return {"reward": reward}


def prepare_rewards(cfg) -> dict:
    """Return the rewards to evaluate, keyed by name, or {} if there are none."""
    if cfg.get("rewards") is None:
        return {}
    rewards = {}
    for name, cfg_reward in cfg.rewards.items():
        with open_dict(cfg_reward):
            cfg_reward.normalize = False  # report the raw scores, as papers do
        rewards[name] = registry.get("reward", cfg_reward.name)(cfg_reward=cfg_reward)
    print(f"✅ Prepared rewards: {', '.join(rewards)}")
    return {"rewards": rewards}


# --------------------------------------------------------------------------------
# Entry point for training and evaluation.
# --------------------------------------------------------------------------------
def train(cfg) -> None:
    """Run the training."""
    # 1. Prepare the dataset.
    train_dataset = registry.get("dataset", cfg.data.name)(cfg_data=cfg.data)
    print("✅ Prepared training dataset")

    # 2. Prepare the model.
    model = registry.get("model", cfg.model.name)(cfg_model=cfg.model)
    print("✅ Prepared model")

    # 3. Prepare the algorithm.
    algorithm = registry.get("algorithm", cfg.algorithm.name)(cfg.algorithm)
    print("✅ Prepared algorithm")

    # 4. Prepare the trainer.
    trainer = registry.get("trainer", cfg.trainer.name)(
        cfg_trainer=cfg.trainer,
        train_dataset=train_dataset,
        model=model,
        algorithm=algorithm,
        log_dir=cfg.log_dir,
        **prepare_reward(cfg),
    )
    print("✅ Prepared trainer")

    # 5. Run the training.
    trainer.run()
    print("✅ Training completed")


def evaluate(cfg) -> None:
    """Run the evaluation."""
    # 1. Prepare the model. Build only the weights that the evaluator samples with.
    model = registry.get("model", cfg.model.name)(
        cfg_model=cfg.model,
        build_keys=(cfg.evaluator.model_key,),
    )
    print("✅ Prepared model")

    # 2. Prepare the algorithm.
    algorithm = registry.get("algorithm", cfg.algorithm.name)(cfg.algorithm)
    print("✅ Prepared algorithm")

    # 3. Prepare the evaluator.
    evaluator = registry.get("evaluator", cfg.evaluator.name)(
        cfg_evaluator=cfg.evaluator,
        model=model,
        algorithm=algorithm,
        log_dir=cfg.log_dir,
        **prepare_rewards(cfg),
    )
    print("✅ Prepared evaluator")

    # 4. Run the evaluation.
    evaluator.run()
    print("✅ Evaluation completed")


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
def main(cfg):
    prepare_env(cfg)

    if cfg.job_type == "train":
        train(cfg)
    elif cfg.job_type == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"❌ Unknown job type: {cfg.job_type}")


if __name__ == "__main__":
    main()
