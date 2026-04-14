"""Main CLI entry point."""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from omegaconf import OmegaConf

from nexus_align.launcher import launch
from nexus_align.cli.draw import run_draw
from nexus_align.utils.download import run_download


def _append_datetime_to_exp_info(script_args: list[str]) -> None:
    """If log.exp_info exists in overrides, append date and time to it."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    last_idx = -1
    for i, override in enumerate(script_args):
        if "log.exp_info=" in override:
            last_idx = i
    if last_idx >= 0:
        prefix, val = script_args[last_idx].split("log.exp_info=", 1)
        script_args[last_idx] = f"{prefix}log.exp_info={val}--{timestamp}"


_HYDRA_CONFIG_GROUPS = {"data", "model", "algorithm", "reward_model"}


def _config_to_hydra_overrides(config_path: str) -> list[str]:
    """Load YAML config file and convert to Hydra override strings.

    Config group sections (data, model, reward_model, algorithm) with a ``name``
    field produce a Hydra *config-group override* (e.g. ``data=text_rendering``)
    so the correct sub-config is loaded.  All other keys use the ``++`` prefix
    to safely create-or-override.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Config file not found: <{config_path}>")

    cfg = OmegaConf.load(path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    group_overrides: list[str] = []
    overrides: list[str] = []

    for group in _HYDRA_CONFIG_GROUPS:
        if group in cfg_dict and isinstance(cfg_dict[group], dict):
            name = cfg_dict[group].pop("name", None)
            if name is not None:
                group_overrides.append(f"{group}={name}")

    if "reward_models" in cfg_dict and "reward_model" not in cfg_dict:
        group_overrides.append("~reward_model")

    def _format_value(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        if isinstance(v, dict):
            pairs = ", ".join(f"{k}: {_format_value(val)}" for k, val in v.items())
            return "{" + pairs + "}"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_format_value(item) for item in v) + "]"
        return str(v)

    def _flatten(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                overrides.append(f"++{key}={_format_value(v)}")

    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            _flatten(v, k)
        else:
            overrides.append(f"++{k}={_format_value(v)}")

    return group_overrides + overrides


def _add_distributed_args(parser: argparse.ArgumentParser) -> None:
    """Add common distributed launch args to a subparser."""
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        dest="node_rank",
        help="Rank of this node (default: 0)",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of GPUs per node (default: 1)",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="Master node IP or hostname (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        dest="master_port",
        help="Master port for rendezvous (default: 29500)",
    )
    parser.add_argument(
        "--rdzv_id",
        type=str,
        default="nexus_align",
        help="Rendezvous job ID for isolating concurrent runs (default: nexus_align)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        dest="config",
        help="Path to YAML config file. Values override defaults; CLI overrides take precedence.",
    )


def _parse_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher-specific args, return (parsed, remaining for hydra)."""
    description = "NexusAlign: a unified and extensible framework for aligning foundation models."
    parser = argparse.ArgumentParser(description=description)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train: nexus-align train [options] [hydra overrides]
    train_parser = subparsers.add_parser("train", help="Launch training")
    _add_distributed_args(train_parser)

    # Evaluation: nexus-align eval [options] [hydra overrides]
    eval_parser = subparsers.add_parser("eval", help="Launch evaluation")
    _add_distributed_args(eval_parser)
    eval_parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory to recursively scan for model.pt checkpoints (batch eval mode)",
    )

    # Download: nexus-align download <repo_id> [options]
    download_parser = subparsers.add_parser("download", help="Launch downloading")
    download_parser.add_argument(
        "--repo_id",
        type=str,
        help="Repo ID to download, e.g. 'black-forest-labs/FLUX.1-dev'.",
    )
    download_parser.add_argument(
        "--cache_dir",
        "-c",
        type=str,
        default=None,
        help="Optional cache directory for the downloaded repo.",
    )
    download_parser.add_argument(
        "--token",
        "-t",
        type=str,
        default=None,
        help="Optional HuggingFace access token for downloading gated repo.",
    )

    # Draw: nexus-align draw -r <results.jsonl> [options]
    draw_parser = subparsers.add_parser("draw", help="Plot results metrics")
    draw_parser.add_argument(
        "--result",
        "-r",
        type=str,
        required=True,
        help="Path to results.jsonl file",
    )
    draw_parser.add_argument(
        "--x_data",
        "-x",
        type=str,
        default="total_step",
        help="Key for x-axis (default: total_step)",
    )
    draw_parser.add_argument(
        "--y_data",
        "-y",
        type=str,
        default="all",
        help="Comma-separated metric keys for y-axis, or 'all' to draw all metrics (default: all), e.g., loss,lr",
    )
    draw_parser.add_argument(
        "--color",
        "-c",
        type=str,
        default=None,
        help="Line color as RGB 0-255, comma-separated (default: blue), e.g., 59,130,246",
    )
    draw_parser.add_argument(
        "--ema",
        "-e",
        type=float,
        default=0.8,
        metavar="ALPHA",
        help="EMA smoothing factor (0 < alpha < 1). Raw data becomes semi-transparent (default: 0.8).",
    )
    draw_parser.add_argument(
        "--dpi",
        "-d",
        type=int,
        default=150,
        help="Output image DPI (default: 150).",
    )
    draw_parser.add_argument(
        "--fontsize",
        "-f",
        type=int,
        default=15,
        help="Base font size (default: 15).",
    )

    args, remaining = parser.parse_known_args(argv)

    return args, remaining


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point.
    Enter this by `nexus-align <command>`, as set by pyproject.toml's entry_points.
    """
    argv = argv or sys.argv[1:]
    if not argv:
        argv = ["--help"]

    args, remaining = _parse_launcher_args(argv)

    if args.command == "train":
        script_args = []
        if getattr(args, "config", None):
            script_args.extend(_config_to_hydra_overrides(args.config))
        script_args.extend(remaining)
        _append_datetime_to_exp_info(script_args)

        return launch(
            script="nexus_align.cli.train",
            script_args=script_args,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            nproc_per_node=args.nproc_per_node,
            master_addr=args.master_addr,
            master_port=args.master_port,
            rdzv_id=args.rdzv_id,
        )

    if args.command == "eval":
        script_args = []
        if getattr(args, "root_dir", None):
            script_args.append(f"++common.batch_eval_root={args.root_dir}")
        if getattr(args, "config", None):
            script_args.extend(_config_to_hydra_overrides(args.config))
        script_args.extend(remaining)
        _append_datetime_to_exp_info(script_args)

        return launch(
            script="nexus_align.cli.evaluation",
            script_args=script_args,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            nproc_per_node=args.nproc_per_node,
            master_addr=args.master_addr,
            master_port=args.master_port,
            rdzv_id=args.rdzv_id,
        )

    if args.command == "draw":
        return run_draw(args)

    if args.command == "download":
        return run_download(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
