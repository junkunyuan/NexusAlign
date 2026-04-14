"""Evaluation entry point.

Supports multiple reward-model evaluators in a single run: inference
runs once, then each evaluator scores the generated images sequentially.

Config accepts either ``reward_model`` (single, legacy) or
``reward_models`` (list).
"""

import copy
import inspect
import os

import hydra

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nexus_align.core.registry import registry
from nexus_align.core import validate_eval_config
from nexus_align.core.config import DTYPE_MAP
from nexus_align.core.base_dataset import BaseTextImageDataset
from nexus_align.engine import with_env_setup, dist_safe_exit
from nexus_align.engine.meter import WindowMeter
from nexus_align.engine.evaluator import (
    cache_check,
    save_meta_data,
    synchronize_and_save_results,
    get_statistics,
)


def _resolve_reward_model_configs(cfg, cfg_dict: dict) -> list[dict]:
    """Return a list of plain-dict reward-model configs.

    Supports both ``cfg.reward_models`` (list, preferred) and the legacy
    ``cfg.reward_model`` (single).
    """
    from omegaconf import OmegaConf

    if hasattr(cfg, "reward_models") and cfg.reward_models is not None:
        return [OmegaConf.to_container(rm, resolve=True) for rm in cfg.reward_models]
    if hasattr(cfg, "reward_model") and cfg.reward_model is not None:
        return [cfg_dict["reward_model"]]
    raise ValueError("❌ Config must contain either 'reward_model' or 'reward_models'")


def run_inference(
    pipeline,
    bench_dataloader,
    result_dir: str,
    infer_file: str,
    log_dir: str,
    meta_data: dict,
    world_size: int,
) -> None:
    """Run inference with the given pipeline and save results.

    Args:
        pipeline: The inference pipeline to call on each batch.
        bench_dataloader: DataLoader yielding benchmark batches.
        result_dir: Directory to save generated images.
        infer_file: Path to the output JSONL file for inference results.
        log_dir: Directory to save meta data.
        meta_data: Dict of meta information to persist alongside results.
        world_size: Number of distributed processes.
    """
    infer_batch_count = len(bench_dataloader)
    infer_meters = WindowMeter()
    infer_meters.add_epoch_step(epoch_window=5, step_window=100)

    os.makedirs(result_dir, exist_ok=True)
    infer_results = []
    for i, data in enumerate(bench_dataloader, start=1):
        print(f"🚀 Inference on batch: {i} / {infer_batch_count}")
        infer_meters.start("step")

        with torch.inference_mode():
            result = pipeline(data)

        for idx, md5 in enumerate(data["md5"]):
            item = {}
            for k, v in data.items():
                if isinstance(v, list):
                    item[k] = v[idx]
                elif isinstance(v, torch.Tensor) and v.ndim == 1:
                    item[k] = v[idx].item()
            if "image" in result.keys():
                save_path = os.path.join(result_dir, f"{md5}.png")
                result["image"][idx].save(save_path)
                item["image"] = save_path
            infer_results.append(item)

        infer_meters.end("step")
        print(infer_meters.info(exp_info=False))

    dist.barrier()

    synchronize_and_save_results(infer_results, infer_file, world_size)
    save_meta_data(meta_data, log_dir)


def run_reward_eval(
    evaluator,
    evaluator_name: str,
    infer_file: str,
    eval_file: str,
    task: str,
    model_name: str,
    data_name: str,
    world_size: int,
    rank: int,
    pin_memory: bool,
    num_workers: int,
    eval_batch_size: int,
    result_key: str = "result",
) -> None:
    """Run a single reward-model evaluation and save results + statistics.

    Args:
        evaluator: An instantiated reward model evaluator.
        evaluator_name: Name of the reward model.
        infer_file: Path to the inference results JSONL.
        eval_file: Path to write evaluation results JSONL.
        task: Task type (e.g. ``"image_gen"``, ``"text_rendering"``).
        model_name: Name of the generative model being evaluated.
        data_name: Name of the benchmark dataset.
        world_size: Number of distributed processes.
        rank: Rank of the current process.
        pin_memory: Whether to pin memory in DataLoader.
        num_workers: Number of DataLoader workers.
        eval_batch_size: Batch size for evaluation.
        result_key: Key under which to store per-sample results.
    """
    print("\n" + "=" * 80)
    print(f"🔍 Evaluator: {evaluator_name}")
    print("=" * 80)

    if task in ("image_gen", "text_rendering"):
        eval_dataset = BaseTextImageDataset(
            data_file_path=infer_file,
            dedup=True,
            kwargs=evaluator.dataset_kwargs,
        )
    else:
        raise ValueError(f"❌ Unsupported task for evaluation: {task}")

    eval_sampler = DistributedSampler(
        eval_dataset, rank=rank, num_replicas=world_size, shuffle=False
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        collate_fn=eval_dataset.collate_fn,
        pin_memory=pin_memory,
        batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    eval_batch_count = len(eval_dataloader)

    total_eval_batch_size = eval_batch_size * world_size
    info = [f"\n📚 Evaluation dataset ({evaluator_name}):"]
    info += [f"    sample count: {len(eval_dataset)}"]
    info += [f"    batchsize: {eval_batch_size}"]
    info += [f"    total batchsize: {total_eval_batch_size}"]
    print("\n".join(info))

    has_detail = "return_detail" in inspect.signature(evaluator.evaluate).parameters

    eval_meters = WindowMeter()
    eval_meters.add_epoch_step(epoch_window=5, step_window=100)

    eval_results = []
    for i, data in enumerate(eval_dataloader, start=1):
        batch_info = f"batch: {i} / {eval_batch_count}"
        print(f"🚀 Evaluating {model_name} on {data_name} by {evaluator_name}: {batch_info}")
        eval_meters.start("step")

        if has_detail:
            results = evaluator.evaluate(data, return_detail=True)
        else:
            results = evaluator.evaluate(data)

        for idx in range(len(data["md5"])):
            item = {}
            for k, v in data.items():
                if isinstance(v, list) and isinstance(v[idx], str):
                    item[k] = v[idx]
                elif k == "md5":
                    item[k] = v[idx]
            item[result_key] = results[idx]
            eval_results.append(item)

        eval_meters.end("step")
        print(eval_meters.info(exp_info=False))

    dist.barrier()

    synchronize_and_save_results(eval_results, eval_file, world_size)
    get_statistics(eval_file=eval_file, result_key=result_key)
    print(f"✅ Completed evaluation with {evaluator_name}")


def _pipeline_accepts_model(model_name: str) -> bool:
    """Check whether the infer pipeline class accepts a ``model`` parameter.

    FLUX-style pipelines receive an external model object; QwenImage / ZImage /
    SD3 pipelines build their own model internally and load ``ckpt_path`` from
    kwargs instead.
    """
    pipeline_cls = registry.get("pipeline", f"{model_name}_infer")
    sig = inspect.signature(pipeline_cls.__init__)
    return "model" in sig.parameters


def _build_pipeline(model_name, dtype, device, cfg_dict, model=None):
    """Build an inference pipeline, adapting to its constructor signature.

    When *model* is provided and the pipeline accepts it (FLUX-style), it is
    passed through.  Otherwise the pipeline is constructed without it
    (QwenImage / ZImage / SD3 style -- these load weights via ``ckpt_path``
    in *cfg_dict* internally).
    """
    pipeline_cls = registry.get("pipeline", f"{model_name}_infer")
    sig = inspect.signature(pipeline_cls.__init__)
    if model is not None and "model" in sig.parameters:
        return pipeline_cls(model=model, dtype=dtype, device=device, kwargs=cfg_dict)
    return pipeline_cls(dtype=dtype, device=device, kwargs=cfg_dict)


def _print_dataset_info(data_name: str, dataset_len: int, batch_size: int, world_size: int) -> None:
    """Print a summary of the inference dataset configuration."""
    total = batch_size * world_size
    info = ["\n📚 Inference dataset:"]
    info += [f"    dataset: {data_name}"]
    info += [f"    sample count: {dataset_len}"]
    info += [f"    batchsize: {batch_size}"]
    info += [f"    total batchsize: {total}"]
    print("\n".join(info))


def _find_checkpoint_dirs(root_dir: str) -> list[str]:
    """Recursively find all directories containing ``model.pt``, sorted by path."""
    ckpt_dirs: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if "model.pt" in filenames:
            ckpt_dirs.append(dirpath)
    ckpt_dirs.sort()
    return ckpt_dirs


def _is_eval_complete(
    eval_dir: str,
    evaluator_names: list[str],
    result_base_name: str,
) -> bool:
    """Return True if all expected eval result files already exist."""
    for name in evaluator_names:
        eval_file = os.path.join(
            eval_dir, f"eval_results--{result_base_name}--{name}.jsonl"
        )
        if not os.path.exists(eval_file):
            return False
    return True


def _run_evaluators(
    cfg, cfg_dict, eval_rm_configs, infer_file, eval_dir,
    result_base_name, device, world_size, rank, model_name, data_name,
):
    """Run all reward-model evaluators on a single inference result."""
    for rm_cfg in eval_rm_configs:
        evaluator_name = rm_cfg["name"]
        eval_file = os.path.join(
            eval_dir, f"eval_results--{result_base_name}--{evaluator_name}.jsonl"
        )

        if os.path.exists(eval_file):
            print(f"⏭️ Eval results already exist for {evaluator_name}: {eval_file}")
            continue

        eval_cfg_dict = copy.copy(cfg_dict)
        eval_cfg_dict["reward_model"] = rm_cfg

        evaluator = registry.get("reward_model", evaluator_name)(
            device=device, kwargs=eval_cfg_dict
        )

        eval_batch_size = rm_cfg.get("eval", {}).get("eval_batch_size", 4)

        run_reward_eval(
            evaluator=evaluator,
            evaluator_name=evaluator_name,
            infer_file=infer_file,
            eval_file=eval_file,
            task=cfg.common.task,
            model_name=model_name,
            data_name=data_name,
            world_size=world_size,
            rank=rank,
            pin_memory=cfg.data.load.pin_memory,
            num_workers=cfg.data.load.num_workers,
            eval_batch_size=eval_batch_size,
        )

        del evaluator
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation (original behaviour)
# ---------------------------------------------------------------------------

def _run_single_eval(cfg, env):
    world_size, rank, device = env.world_size, env.rank, env.device
    cfg_dict = env.cfg_dict

    cfg.log.result_dir = os.path.join(cfg.log.log_dir, cfg.log.result_dir)
    seed = env.seed
    data_name = cfg.data.name
    model_name = cfg.model.name
    result_base_name = f"{model_name}--{data_name}--{seed}"
    meta_data = {
        "data_name": data_name,
        "model_name": model_name,
        "seed": seed,
        "model_ckpt_path": cfg.model.eval.ckpt_path,
    }

    eval_rm_configs = _resolve_reward_model_configs(cfg, cfg_dict)
    evaluator_names = [rm["name"] for rm in eval_rm_configs]
    print(f"📋 Evaluators to run: {evaluator_names}")

    infer_file = f"infer_results--{result_base_name}.jsonl"
    infer_file = os.path.join(cfg.common.cache_log_dir, infer_file)

    # -- Prepare Dataset ----------------------------------------------------------
    if_infer_file_exists = cache_check(
        cache_log_dir=cfg.common.cache_log_dir, 
        meta_data=meta_data,
        infer_file=infer_file,
        eval_file=None,
    )

    if if_infer_file_exists:
        bench_dataset = None
        print(f"✅ Start evaluating by using the cached inference file <{infer_file}>")
    else:
        bench_dataset = registry.get("dataset", data_name)(cfg_dict)
        bench_sampler = DistributedSampler(
            bench_dataset, 
            rank=rank, 
            num_replicas=world_size, 
            shuffle=False
        )
        bench_dataloader = DataLoader(
            bench_dataset,
            sampler=bench_sampler,
            collate_fn=getattr(bench_dataset, "collate_fn", None),
            pin_memory=cfg.data.load.pin_memory,
            batch_size=cfg.model.eval.eval_batch_size,
            num_workers=cfg.data.load.num_workers,
        )
        
        _print_dataset_info(data_name, len(bench_dataset), cfg.model.eval.eval_batch_size, world_size)
        print("✅ Prepared inference dataset")

    # -- Build Inference Pipeline & Run Inference ---------------------------------
    if bench_dataset is not None:
        model = None
        if _pipeline_accepts_model(model_name):
            model = registry.get("model", model_name)(
                device=device,
                model_dtype=cfg.model.model_dtype,
                kwargs=cfg_dict,
                env=env,
            )

        pipeline = _build_pipeline(
            model_name, DTYPE_MAP[cfg.model.amp_dtype], device, cfg_dict, model=model,
        )
        print(f"✅ Built inference pipeline: {model_name}")

        run_inference(
            pipeline=pipeline,
            bench_dataloader=bench_dataloader,
            result_dir=cfg.log.result_dir,
            infer_file=infer_file,
            log_dir=cfg.log.log_dir,
            meta_data=meta_data,
            world_size=world_size,
        )

    # -- Evaluate with each reward model ------------------------------------------
    _run_evaluators(
        cfg, cfg_dict, eval_rm_configs, infer_file, cfg.log.log_dir,
        result_base_name, device, world_size, rank, model_name, data_name,
    )


# ---------------------------------------------------------------------------
# Batch evaluation (--root_dir mode)
# ---------------------------------------------------------------------------

def _run_batch_eval(cfg, env):
    world_size, rank, device = env.world_size, env.rank, env.device
    cfg_dict = env.cfg_dict

    seed = env.seed
    data_name = cfg.data.name
    model_name = cfg.model.name
    result_base_name = f"{model_name}--{data_name}--{seed}"

    batch_eval_root = cfg.common.batch_eval_root
    if not os.path.isdir(batch_eval_root):
        raise ValueError(
            f"❌ common.batch_eval_root must be a valid directory, got: {batch_eval_root}"
        )

    ckpt_dirs = _find_checkpoint_dirs(batch_eval_root)
    if not ckpt_dirs:
        print(f"⚠️ No checkpoint directories (containing model.pt) found under {batch_eval_root}")
        return

    print(f"📋 Found {len(ckpt_dirs)} checkpoint(s) under {batch_eval_root}:")
    for d in ckpt_dirs:
        print(f"    {d}")

    eval_rm_configs = _resolve_reward_model_configs(cfg, cfg_dict)
    evaluator_names = [rm["name"] for rm in eval_rm_configs]
    print(f"📋 Evaluators to run: {evaluator_names}")

    # -- Build model & pipeline ONCE ----------------------------------------------
    model = None
    if _pipeline_accepts_model(model_name):
        model = registry.get("model", model_name)(
            device=device,
            model_dtype=cfg.model.model_dtype,
            kwargs=cfg_dict,
            env=env,
        )

    pipeline = _build_pipeline(
        model_name, DTYPE_MAP[cfg.model.amp_dtype], device, cfg_dict, model=model,
    )
    print(f"✅ Built inference pipeline: {model_name} (VAE / text encoder / scheduler loaded once)")

    # -- Prepare benchmark dataset ONCE -------------------------------------------
    bench_dataset = registry.get("dataset", data_name)(cfg_dict)
    bench_sampler = DistributedSampler(
        bench_dataset, rank=rank, num_replicas=world_size, shuffle=False
    )
    bench_dataloader = DataLoader(
        bench_dataset,
        sampler=bench_sampler,
        collate_fn=getattr(bench_dataset, "collate_fn", None),
        pin_memory=cfg.data.load.pin_memory,
        batch_size=cfg.model.eval.eval_batch_size,
        num_workers=cfg.data.load.num_workers,
    )

    _print_dataset_info(data_name, len(bench_dataset), cfg.model.eval.eval_batch_size, world_size)

    # -- Create eval root directory (sibling of batch_eval_root) ------------------
    eval_root = os.path.join(os.path.dirname(batch_eval_root), "eval")
    os.makedirs(eval_root, exist_ok=True)
    print(f"📁 Eval results will be saved to: {eval_root}")

    # -- Loop over checkpoints ----------------------------------------------------
    for ckpt_idx, ckpt_dir in enumerate(ckpt_dirs, start=1):
        ckpt_path = os.path.join(ckpt_dir, "model.pt")
        ckpt_name = os.path.basename(ckpt_dir)
        eval_dir = os.path.join(eval_root, ckpt_name)

        print("\n" + "#" * 80)
        print(f"📦 Checkpoint [{ckpt_idx}/{len(ckpt_dirs)}]: {ckpt_dir}")
        print("#" * 80)

        if os.path.isdir(eval_dir) and _is_eval_complete(eval_dir, evaluator_names, result_base_name):
            print(f"⏭️ Skipping (eval/ already complete)")
            continue

        os.makedirs(eval_dir, exist_ok=True)
        result_dir = os.path.join(eval_dir, "inference_results")
        infer_file = os.path.join(eval_dir, f"infer_results--{result_base_name}.jsonl")

        meta_data = {
            "data_name": data_name,
            "model_name": model_name,
            "seed": seed,
            "model_ckpt_path": ckpt_path,
        }

        # -- Swap transformer weights only (VAE / text encoder stay unchanged) ----
        print(f"⏳ Loading transformer weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        pipeline.pipe.transformer.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache()
        print(f"✅ Loaded transformer weights")

        # -- Run inference (skip if already exists) -------------------------------
        if os.path.exists(infer_file):
            print(f"⏭️ Inference results already exist: {infer_file}")
        else:
            run_inference(
                pipeline=pipeline,
                bench_dataloader=bench_dataloader,
                result_dir=result_dir,
                infer_file=infer_file,
                log_dir=eval_dir,
                meta_data=meta_data,
                world_size=world_size,
            )

        # -- Run reward model evaluations -----------------------------------------
        _run_evaluators(
            cfg, cfg_dict, eval_rm_configs, infer_file, eval_dir,
            result_base_name, device, world_size, rank, model_name, data_name,
        )

        print(f"✅ Finished checkpoint [{ckpt_idx}/{len(ckpt_dirs)}]: {ckpt_dir}")

    del pipeline
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"🎉 Batch evaluation complete. Evaluated {len(ckpt_dirs)} checkpoint(s).")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="evaluation",
    version_base="1.3",
)
@with_env_setup(validator=validate_eval_config)
def main(cfg, env):
    batch_eval_root = getattr(cfg.common, "batch_eval_root", None) or ""
    if batch_eval_root:
        _run_batch_eval(cfg, env)
    else:
        _run_single_eval(cfg, env)

    if env.profiler is not None:
        env.profiler.stop()
        env.profiler.print()


if __name__ == "__main__":
    main()
    dist_safe_exit(exit_code=0)
