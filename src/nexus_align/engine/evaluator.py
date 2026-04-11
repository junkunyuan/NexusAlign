"""
Evaluation utilities: cache check, result sync/save, and statistics.
Used by the evaluation CLI.
"""

import os
import json

import torch.distributed as dist


def cache_check(
    cache_log_dir: str, 
    meta_data: dict,
    infer_file: str | None = None,
    eval_file: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Check that the cache directory exists and meta matches; return (eval_results_file, infer_results_file).
    One of the returned paths may be None.
    """
    def meta_check(cache_log_dir: str, meta_data: dict) -> None:
        meta_file = os.path.join(cache_log_dir, "meta.json")
        if not os.path.exists(meta_file):
            raise ValueError(f"❌ Meta file does not exist in {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as f:
            loaded_meta_data = json.load(f)
        if loaded_meta_data != meta_data:
            raise ValueError(f"❌ Meta data does not match the one in {meta_file}")
    
    is_infer_file_exists = False
    if isinstance(cache_log_dir, str) and len(cache_log_dir) > 0:
        if not os.path.exists(cache_log_dir):
            raise ValueError(f"❌ Cache log dir does not exist in {cache_log_dir}")
        
        meta_check(cache_log_dir, meta_data)
        
        if not os.path.exists(infer_file):
            raise ValueError(f"❌ No appropriate inference files found in {cache_log_dir}")
        
        eval_file_name = eval_file.rsplit("/", 1)[-1]
        if os.path.exists(os.path.join(cache_log_dir, eval_file_name)):
           raise ValueError(f"❌ Evaluation file already exists in {eval_file}")

        is_infer_file_exists = True
    
    return is_infer_file_exists

def save_meta_data(meta_data: dict, log_dir: str) -> None:
    """Save meta data to meta.json in log_dir. Only rank 0 writes."""
    if dist.get_rank() == 0:
        meta_file = os.path.join(log_dir, "meta.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved meta data to <{meta_file}>")
    dist.barrier()


def synchronize_and_save_results(
    results: list, 
    file_path: str, 
    world_size: int
) -> None:
    """Synchronize and save results to a file."""
    try:
        all_results_list = [None] * world_size
        dist.all_gather_object(all_results_list, results)
        results = [item for res in all_results_list for item in res]
        results.sort(key=lambda x: x["md5"])
        if dist.get_rank() == 0:
            with open(file_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        dist.barrier()
        print(f"💾 Synchronized and saved results to <{file_path}>")
    except (RuntimeError, OSError) as e:
        print(f"❌ Failed to synchronize and save results: {e}")
        return


def get_statistics(eval_file: str, result_key: str = "result") -> None:
    """Compute and print statistics for the evaluation result file; write a summary to a .stat.txt file in the same directory."""
    print("📊 Statistics analysis:")

    results = []
    invalid = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if result_key in data:
                    result = data[result_key]
                    if result is None:
                        invalid.append(result)
                    else:
                        results.append(float(result))
            except json.JSONDecodeError:
                continue
    stat_mean = sum(results) / len(results) if results else None
    stat_max = max(results) if results else None
    stat_min = min(results) if results else None

    lines = [
        "\n" + "-" * 80 + f"\nStatistics: {eval_file}\n" + "-" * 80,
        f"Valid Sample Count: {len(results):,}",
        f"Invalid Sample Count: {len(invalid):,}",
    ]
    if len(results) > 0:
        lines.extend(
            [
                "Statistics:",
                f"   ├─ Mean:  {stat_mean:.6f}",
                f"   ├─ Max:   {stat_max:.6f}",
                f"   └─ Min:   {stat_min:.6f}",
            ]
        )
    else:
        lines.append("❌ Warning: No valid result data found in file")
    stats_str = "\n".join(lines)

    eval_dir = os.path.dirname(eval_file)
    stat_basename = os.path.splitext(os.path.basename(eval_file))[0] + ".stat.txt"
    output_path = os.path.join(eval_dir, stat_basename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats_str)
    print(stats_str)
    print(f"💾 Statistics saved to: <{output_path}>")
