"""Main CLI entry point."""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from omegaconf import OmegaConf

from nexus_align.launcher import launch


def _append_datetime_to_exp_info(script_args: list[str]) -> None:
    """If log.exp_info exists in overrides, append date and time to it."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    last_idx = -1
    for i, override in enumerate(script_args):
        if override.startswith("log.exp_info="):
            last_idx = i
    if last_idx >= 0:
        val = script_args[last_idx].split("=", 1)[1]
        script_args[last_idx] = f"log.exp_info={val}--{timestamp}"


def _config_to_hydra_overrides(config_path: str) -> list[str]:
    """Load YAML config file and convert to Hydra override strings."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(path)
    overrides: list[str] = []

    def _format_value(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (list, tuple)):
            return ",".join(str(x) for x in v)
        return str(v)

    def _flatten(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            v_plain = OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
            if isinstance(v_plain, dict):
                _flatten(v_plain, key)
            else:
                overrides.append(f"{key}={_format_value(v_plain)}")

    for k, v in OmegaConf.to_container(cfg, resolve=True).items():
        if isinstance(v, dict):
            _flatten(v, k)
        else:
            overrides.append(f"{k}={_format_value(v)}")

    return overrides


def _add_distributed_args(parser: argparse.ArgumentParser) -> None:
    """Add common distributed launch args to a subparser."""
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of GPUs per node (default: 1)",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        dest="node_rank",
        help="Rank of this node (default: 0)",
    )
    parser.add_argument(
        "--master-ip",
        type=str,
        default="127.0.0.1",
        dest="master_ip",
        help="Master node IP or hostname (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--master-port",
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
    description = "NexusAlign: a unified framework for aligning foundation models."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "preset",
        nargs="?",
        default=None,
        help="Optional config preset.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train: nexus-align train [preset] [options] [hydra overrides]
    train_parser = subparsers.add_parser("train", help="Launch training")
    _add_distributed_args(train_parser)

    # Evaluation: nexus-align eval [options] [hydra overrides]
    eval_parser = subparsers.add_parser("eval", help="Launch distributed evaluation")
    _add_distributed_args(eval_parser)

    args, remaining = parser.parse_known_args(argv)

    return args, remaining


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    argv = argv or sys.argv[1:]
    if not argv:
        argv = ["--help"]

    args, remaining = _parse_launcher_args(argv)

    if args.command == "train":
        script_args = []
        if getattr(args, "preset", None):
            script_args.append(args.preset)
        if getattr(args, "config", None):
            script_args.extend(_config_to_hydra_overrides(args.config))
        script_args.extend(remaining)
        _append_datetime_to_exp_info(script_args)

        return launch(
            script="nexus_align.cli.train",
            script_args=script_args,
            nnodes=args.nnodes,
            nproc_per_node=args.nproc_per_node,
            node_rank=args.node_rank,
            master_addr=args.master_ip,
            master_port=args.master_port,
            rdzv_id=args.rdzv_id,
        )

    if args.command == "eval":
        script_args = []
        if getattr(args, "config", None):
            script_args.extend(_config_to_hydra_overrides(args.config))
        script_args.extend(remaining)
        _append_datetime_to_exp_info(script_args)

        return launch(
            script="nexus_align.cli.evaluation",
            script_args=script_args,
            nnodes=args.nnodes,
            nproc_per_node=args.nproc_per_node,
            node_rank=args.node_rank,
            master_addr=args.master_ip,
            master_port=args.master_port,
            rdzv_id=args.rdzv_id,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
