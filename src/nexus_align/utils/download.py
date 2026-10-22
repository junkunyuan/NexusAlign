"""
Download models, datasets, and evaluation assets into the data_and_model dir.

Every asset is stored at <data_and_model_dir>/<name>, so the names below match the
paths used by the configs (e.g. data.vae, evaluator.inception_weights).

Usage (from the project root):
    python src/nexus_align/utils/download.py --list
    python src/nexus_align/utils/download.py stabilityai/sd-vae-ft-mse
    python src/nexus_align/utils/download.py meanflow  # a preset: all assets of one experiment
    python src/nexus_align/utils/download.py paddleocr  # a dir: all assets under it
    python src/nexus_align/utils/download.py all
    python src/nexus_align/utils/download.py ILSVRC/imagenet-1k --include "data/train-*"

A name not in ASSETS is downloaded as a custom asset when its source is given:
    python src/nexus_align/utils/download.py Qwen/Qwen3-8B --source hf
    python src/nexus_align/utils/download.py ylecun/mnist --source hf --repo-type dataset
    python src/nexus_align/utils/download.py others/file.pth --source url --url <url>

Gated Hugging Face repos (e.g. ImageNet) need a token: pass --token or set HF_TOKEN.
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

# --------------------------------------------------------------------------------
# Preset assests to download here.
# --------------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DIR = PROJECT_DIR / "data_and_model"

# name -> {"kind": "datasets" | "models" | "others", "source": "hf" | "url", ...};
# the name is also the local path.
ASSETS = {
    # Datasets.
    "ILSVRC/imagenet-1k": {"kind": "datasets", "source": "hf", "repo_type": "dataset"},
    # Models.
    "stabilityai/sd-vae-ft-mse": {"kind": "models", "source": "hf", "ignore": ["*.bin"]},
    "stabilityai/stable-diffusion-3.5-medium": {
        "kind": "models",
        "source": "hf",
        "ignore": ["sd3.5_medium.safetensors", "text_encoders/*", "*.fp16*"],
    },
    "yuvalkirstain/PickScore_v1": {"kind": "models", "source": "hf", "ignore": ["*.bin"]},
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
        "kind": "models",
        "source": "hf",
        "include": ["*.json", "*.txt"],
    },
    "timm/vit_large_patch14_clip_224.openai": {
        "kind": "models",
        "source": "hf",
        "include": ["open_clip_config.json", "open_clip_model.safetensors"],
    },
    "torch_fidelity/weights-inception-2015-12-05-6726825d.pth": {
        "kind": "others",
        "source": "url",
        "url": "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth",
    },
    "fid_stats/adm_in256_stats.npz": {  # ImageNet 256x256 reference statistics of MeanFlow
        "kind": "others",
        "source": "url",
        "url": "https://raw.githubusercontent.com/zhuyu-cs/MeanFlow/b006921898d82ea0dd75cfae5d242863fcfba56e/fid_stats/adm_in256_stats.npz",
    },
    "paddleocr/whl/det/en/en_PP-OCRv3_det_infer": {
        "kind": "others",
        "source": "url",
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "archive": "tar",
    },
    "paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer": {
        "kind": "others",
        "source": "url",
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        "archive": "tar",
    },
    "paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer": {
        "kind": "others",
        "source": "url",
        "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
        "archive": "tar",
    },
    "mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth": {
        "kind": "others",
        "source": "url",
        "url": "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth",
    },
    "yifan123/reward-server": {
        "kind": "others",
        "source": "url",
        "url": "https://github.com/yifan123/reward-server/archive/refs/heads/main.zip",
        "archive": "zip",
    },
    "open-mmlab/mmdetection": {
        "kind": "others",
        "source": "url",
        "url": "https://github.com/open-mmlab/mmdetection/archive/refs/heads/2.x.zip",
        "archive": "zip",
    },
    "yifan123/flow_grpo": {
        "kind": "datasets",
        "source": "url",
        "url": "https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da",
        "files": [
            "dataset/geneval/train_metadata.jsonl",
            "dataset/geneval/test_metadata.jsonl",
            "dataset/ocr/train.txt",
            "dataset/ocr/test.txt",
            "dataset/pickscore/train.txt",
            "dataset/pickscore/test.txt",
        ],
    },
    "NVlabs/DiffusionNFT": {
        "kind": "datasets",
        "source": "url",
        "url": "https://raw.githubusercontent.com/NVlabs/DiffusionNFT/cbb14f84b8312b620390dfcbe2fab69c1383b104",
        "files": ["dataset/drawbench/test.txt"],
    },
    # DrawBench scorers of Flow-GRPO (paper Table 2) and DiffusionNFT (evaluation.py).
    "openai/clip-vit-large-patch14": {
        "kind": "models",
        "source": "hf",
        "include": ["*.json", "*.txt", "model.safetensors"],
    },
    "aesthetic/sac+logos+ava1-l14-linearMSE.pth": {
        "kind": "models",
        "source": "url",
        "url": "https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/assets/sac+logos+ava1-l14-linearMSE.pth",
    },
    "THUDM/ImageReward": {
        "kind": "models",
        "source": "hf",
        "include": ["ImageReward.pt", "med_config.json"],
    },
    "google-bert/bert-base-uncased": {  # tokenizer of ImageReward's BLIP
        "kind": "models",
        "source": "hf",
        "include": ["config.json", "tokenizer*.json", "vocab.txt"],
    },
    "zhiyuanyou/DeQA-Score-Mix3": {
        "kind": "models",
        "source": "hf",
        "ignore": ["*.bin"],
    },
    "CodeGoat24/UnifiedReward-7b-v1.5": {"kind": "models", "source": "hf"},
    # Vision tower that sglang builds UnifiedReward with, before loading its own weights.
    "google/siglip-so400m-patch14-384": {
        "kind": "models",
        "source": "hf",
        "include": ["config.json", "preprocessor_config.json", "model.safetensors"],
    },
    "xswu/HPSv2": {
        "kind": "models",
        "source": "hf",
        "include": ["HPS_v2.1_compressed.pt"],
    },
    # HPSv3 (https://github.com/MizzenAI/HPSv3), built on Qwen2-VL-7B-Instruct.
    "MizzenAI/HPSv3": {
        "kind": "models",
        "source": "hf",
        "include": ["HPSv3.safetensors"],
    },
    "Qwen/Qwen2-VL-7B-Instruct": {"kind": "models", "source": "hf"},
}
KINDS = {"datasets": "📚", "models": "🤖", "others": "🧩"}

# preset -> asset names; "all" is added automatically.
PRESETS = {
    "meanflow": [
        "stabilityai/sd-vae-ft-mse",
        "ILSVRC/imagenet-1k",
        "torch_fidelity/weights-inception-2015-12-05-6726825d.pth",
        "fid_stats/adm_in256_stats.npz",
    ],
    "imeanflow": [
        "stabilityai/sd-vae-ft-mse",
        "ILSVRC/imagenet-1k",
        "torch_fidelity/weights-inception-2015-12-05-6726825d.pth",
        "fid_stats/adm_in256_stats.npz",
    ],
    "flow_grpo": [
        "stabilityai/stable-diffusion-3.5-medium",
        "yuvalkirstain/PickScore_v1",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "timm/vit_large_patch14_clip_224.openai",
        "paddleocr/whl/det/en/en_PP-OCRv3_det_infer",
        "paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer",
        "paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer",
        "mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
        "yifan123/reward-server",
        "open-mmlab/mmdetection",
        "yifan123/flow_grpo",
    ],
    "diffusion_nft": [
        "stabilityai/stable-diffusion-3.5-medium",
        "yuvalkirstain/PickScore_v1",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "timm/vit_large_patch14_clip_224.openai",
        "paddleocr/whl/det/en/en_PP-OCRv3_det_infer",
        "paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer",
        "paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer",
        "mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
        "yifan123/reward-server",
        "open-mmlab/mmdetection",
        "yifan123/flow_grpo",
        "NVlabs/DiffusionNFT",
    ],
    "flow_grpo_drawbench": [
        "NVlabs/DiffusionNFT",  # DrawBench prompts, identical to Flow-GRPO's
        "yuvalkirstain/PickScore_v1",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "openai/clip-vit-large-patch14",  # Aesthetic and ClipScore
        "aesthetic/sac+logos+ava1-l14-linearMSE.pth",
        "THUDM/ImageReward",
        "google-bert/bert-base-uncased",
        "zhiyuanyou/DeQA-Score-Mix3",
        "yifan123/reward-server",  # app_deqa.py
        "CodeGoat24/UnifiedReward-7b-v1.5",
        "google/siglip-so400m-patch14-384",
        "xswu/HPSv2",
        "MizzenAI/HPSv3",  # not used by Flow-GRPO or DiffusionNFT
        "Qwen/Qwen2-VL-7B-Instruct",
    ],
}
PRESETS["all"] = list(ASSETS)


def resolve_names(names: list[str]) -> list[str]:
    """Expand presets and dirs, validate asset names, keep the order, and drop duplicates."""
    resolved = []
    for name in names:
        expanded = PRESETS.get(name) or [n for n in ASSETS if n.startswith(f"{name}/")] or [name]
        unknown = [n for n in expanded if n not in ASSETS]
        if unknown:
            print(f"❌ Unknown asset: {unknown}; use --list to see the choices or pass --source", file=sys.stderr)
            sys.exit(1)
        resolved += [n for n in expanded if n not in resolved]
    return resolved


def download_hf(name: str, info: dict, dest: Path, args) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=name,
        repo_type=info.get("repo_type", "model"),
        local_dir=dest,
        allow_patterns=args.include or info.get("include"),
        ignore_patterns=info.get("ignore"),
        token=args.token,
        max_workers=args.workers,
        force_download=args.force,
    )


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(path, "wb") as f:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm.wrapattr(f, "write", total=total, unit="B", unit_scale=True, desc=path.name) as out:
            shutil.copyfileobj(response, out)


def download_url(name: str, info: dict, dest: Path, args) -> None:
    if dest.exists() and not args.force:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    if "files" in info:  # only the listed files under url, kept at their relative paths
        shutil.rmtree(tmp, ignore_errors=True)
        for file in info["files"]:
            fetch(f"{info['url']}/{file}", tmp / file)
        shutil.rmtree(dest, ignore_errors=True)
        tmp.replace(dest)
        return
    fetch(info["url"], tmp)
    if "archive" in info:
        unpack(tmp, dest, info["archive"])
        tmp.unlink()
    else:
        tmp.replace(dest)


def unpack(archive: Path, dest: Path, fmt: str) -> None:
    """Unpack an archive into dest, dropping its top-level dir if it is the only entry."""
    tmp_dir = dest.with_name(dest.name + ".unpack")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.unpack_archive(archive, tmp_dir, format=fmt)
    entries = list(tmp_dir.iterdir())
    src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp_dir
    shutil.rmtree(dest, ignore_errors=True)
    src.replace(dest)
    shutil.rmtree(tmp_dir, ignore_errors=True)


DOWNLOADERS = {"hf": download_hf, "url": download_url}


def download(names: list[str], args) -> None:
    """Download every asset, keep going on failures, and report them at the end."""
    root = Path(args.dir).expanduser().resolve()
    failed = []
    for i, name in enumerate(names, 1):
        info = ASSETS[name]
        dest = root / name
        print(f"🚀 [{i}/{len(names)}] Downloading {name} from {info['source']} to <{dest}>")
        try:
            DOWNLOADERS[info["source"]](name, info, dest, args)
            print(f"✅ Ready: {name}")
        except Exception as e:
            failed.append(name)
            print(f"❌ Failed to download {name}: {e}", file=sys.stderr)
    if failed:
        print(f"⚠️ Failed: {failed}", file=sys.stderr)
        sys.exit(1)
    print(f"💾 All {len(names)} assets are ready in <{root}>")


def print_by_kind(names: list[str], root: Path, indent: int) -> None:
    """Print assets grouped by kind, marking the ones already present under root."""
    pad = " " * indent
    for kind in KINDS:
        members = [n for n in names if ASSETS[n]["kind"] == kind]
        if not members:
            continue
        print(f"{pad}{KINDS[kind]} {kind}:")
        for name in members:
            status = "✅" if (root / name).exists() else "❌"
            print(f"{pad}    {status} {name}  ({ASSETS[name]['source']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="asset names and/or presets")
    parser.add_argument("--list", action="store_true", help="list assets and presets, then exit")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="root dir to store assets")
    parser.add_argument("--token", default=None, help="Hugging Face token; defaults to HF_TOKEN")
    parser.add_argument("--include", nargs="+", default=None, help="glob patterns to download (HF only)")
    parser.add_argument("--workers", type=int, default=8, help="parallel file downloads (HF only)")
    parser.add_argument("--force", action="store_true", help="re-download even if the asset exists")
    parser.add_argument("--source", choices=list(DOWNLOADERS), default=None, help="source of a custom asset")
    parser.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model", help="HF repo type of a custom asset")
    parser.add_argument("--url", default=None, help="URL of a custom asset with --source url")
    args = parser.parse_args()
    if args.source == "url" and not args.url:
        parser.error("--source url requires --url")

    if args.list or not args.names:
        root = Path(args.dir).expanduser().resolve()
        print("📦 Assets:")
        print_by_kind(list(ASSETS), root, indent=4)
        print("📦 Presets:")
        for preset, members in PRESETS.items():
            print(f"    📦 {preset}:")
            print_by_kind(members, root, indent=8)
        return

    if args.source:  # register names not in ASSETS or PRESETS as custom assets
        info = {"source": args.source, "repo_type": args.repo_type, "url": args.url}
        ASSETS.update({n: info for n in args.names if n not in ASSETS and n not in PRESETS})
    download(resolve_names(args.names), args)


if __name__ == "__main__":
    main()
