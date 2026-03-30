"""Evaluation entry point."""

import os
import sys
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


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="evaluation",
    version_base="1.3",
)
@with_env_setup(validator=validate_eval_config)
def main(cfg, env):
    # --------------------------------------------------------------------------------
    # 1. Prepare environment (done by decorator)
    # --------------------------------------------------------------------------------
    world_size, rank, device = env.world_size, env.rank, env.device
    cfg_dict = env.cfg_dict

    cfg.log.result_dir = os.path.join(cfg.log.log_dir, cfg.log.result_dir)
    seed = env.seed
    data_name = cfg.data.name
    model_name = cfg.model.name
    evaluator_name = cfg.reward_model.name
    result_base_name = f"{model_name}--{data_name}--{seed}"
    meta_data = {
        "data_name": data_name,
        "model_name": model_name,
        "seed": seed,
        "model_ckpt_path": cfg.model.eval.ckpt_path,
    }
    
    infer_file = f"infer_results--{result_base_name}.jsonl"
    infer_file = os.path.join(cfg.common.cache_log_dir, infer_file)
    eval_file = f"eval_results--{result_base_name}--{evaluator_name}.jsonl"
    eval_file = os.path.join(cfg.log.log_dir, eval_file)

    # --------------------------------------------------------------------------------
    # 2. Prepare Dataset
    # --------------------------------------------------------------------------------
    if_infer_file_exists = cache_check(
        cache_log_dir=cfg.common.cache_log_dir, 
        meta_data=meta_data,
        infer_file=infer_file,
        eval_file=eval_file
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
        infer_batch_count = len(bench_dataloader)
        
        total_infer_batch_size = cfg.model.eval.eval_batch_size * world_size
        info = ["\n📚 Inference dataset:"]
        info += [f"    dataset: {cfg.data.name}"]
        info += [f"    sample count: {len(bench_dataset)}"]
        info += [f"    batchsize: {cfg.model.eval.eval_batch_size}"]
        info += [f"    total batchsize: {total_infer_batch_size}"]
        if cfg.data.load.sample_ratio != 1.:
            info += [f"    sample_ratio: {cfg.data.load.sample_ratio}"]
        if cfg.data.load.drop_last:
            info += ["    drop_last: true"]
        if cfg.data.load.cache_dir:
            info += [f"    cache_dir: {cfg.data.load.cache_dir}"]
        print("\n".join(info))
        print("✅ Prepared inference dataset")

    # --------------------------------------------------------------------------------
    # 3. Build Inference Pipeline
    # --------------------------------------------------------------------------------
    if bench_dataset is not None:
        model = registry.get("model", model_name)(
            device=device,
            model_dtype=cfg.model.model_dtype,
            kwargs=cfg_dict,
            env=env,
        )

        pipeline = registry.get("pipeline", f"{model_name}_infer")(
            model=model,
            dtype=DTYPE_MAP[cfg.model.amp_dtype],
            device=device,
            kwargs=cfg_dict,
        )
        print(f"✅ Built inference pipeline: {model_name}")
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # 4. Inference
    # --------------------------------------------------------------------------------
    if bench_dataset is not None:
        infer_meters = WindowMeter()
        infer_meters.add_epoch_step(epoch_window=5, step_window=100)

        os.makedirs(cfg.log.result_dir, exist_ok=True)
        infer_results = []
        for i, data in enumerate(bench_dataloader, start=1):
            print(f"🚀 Inference on batch: {i} / {infer_batch_count}")
            infer_meters.start("step")

            with torch.inference_mode():
                result = pipeline(data)  # inference

            # Save results
            for idx, md5 in enumerate(data["md5"]):
                item = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        item[k] = v[idx]
                    elif isinstance(v, torch.Tensor) and v.ndim == 1:
                        item[k] = v[idx].item()
                if "image" in result.keys():
                    save_path = os.path.join(cfg.log.result_dir, f"{md5}.png")
                    result["image"][idx].save(save_path)  # save result
                    item["image"] = save_path
                infer_results.append(item)

            infer_meters.end("step")
            print(infer_meters.info(exp_info=False))

        dist.barrier()

        synchronize_and_save_results(infer_results, infer_file, world_size)
        save_meta_data(meta_data, cfg.log.log_dir)

    # --------------------------------------------------------------------------------
    # 5. Prepare Evaluators
    # --------------------------------------------------------------------------------
    evaluator = registry.get("reward_model", cfg.reward_model.name)(device=device, kwargs=cfg_dict)

    # --------------------------------------------------------------------------------
    # 6. Prepare Evaluation Dataset
    # --------------------------------------------------------------------------------
    if cfg.common.task == "image_gen":
        eval_dataset = BaseTextImageDataset(
            data_file_path=infer_file, 
            dedup=True,
            kwargs=evaluator.dataset_kwargs
        )
    eval_sampler = DistributedSampler(
        eval_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=False
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        collate_fn=eval_dataset.collate_fn,
        pin_memory=cfg.data.load.pin_memory,
        batch_size=cfg.reward_model.eval.eval_batch_size,
        num_workers=cfg.data.load.num_workers,
    )
    eval_batch_count = len(eval_dataloader)
    
    total_eval_batch_size = cfg.reward_model.eval.eval_batch_size * world_size
    info = ["\n📚 Evaluation dataset:"]
    info += [f"    dataset: {cfg.data.name}"]
    info += [f"    sample count: {len(eval_dataset)}"]
    info += [f"    batchsize: {cfg.reward_model.eval.eval_batch_size}"]
    info += [f"    total batchsize: {total_eval_batch_size}"]
    print("\n".join(info))
    print("✅ Prepared evaluation dataset")

    # --------------------------------------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------------------------------------
    eval_meters = WindowMeter()
    eval_meters.add_epoch_step(epoch_window=5, step_window=100)

    eval_results = []
    result_key = "result"
    for i, data in enumerate(eval_dataloader, start=1):
        batch_info = f"batch: {i} / {eval_batch_count}"
        print(f"🚀 Evaluating {model_name} on {data_name} by {evaluator_name}: {batch_info}")
        eval_meters.start("step")

        results = evaluator.evaluate(data)  # evaluation

        # Save results
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

    # --------------------------------------------------------------------------------
    # 8. Get Statistics
    # --------------------------------------------------------------------------------
    get_statistics(eval_file=eval_file, result_key=result_key)
    # --------------------------------------------------------------------------------

    if env.profiler is not None:
        env.profiler.stop()
        env.profiler.print()


if __name__ == "__main__":
    main()
    dist_safe_exit(exit_code=0)
