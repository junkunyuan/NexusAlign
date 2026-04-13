"""Model download utilities."""

import os
import sys
from typing import Dict, TypedDict

from huggingface_hub import snapshot_download as hf_download


class RepoInfo(TypedDict, total=False):
    platform: str
    repo_id: str
    url: str


PREDEFINED_REPOS: Dict[str, RepoInfo] = {
    # Foundation Models
    "black-forest-labs/FLUX.1-dev": {
        "platform": "huggingface",
        "repo_id": "black-forest-labs/FLUX.1-dev",
    },
    "Qwen/Qwen-Image": {
        "platform": "huggingface",
        "repo_id": "Qwen/Qwen-Image",
    },
    "stabilityai/stable-diffusion-3-medium": {
        "platform": "huggingface",
        "repo_id": "stabilityai/stable-diffusion-3-medium",
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "platform": "huggingface",
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
    },

    # Data
    "ymhao/HPDv2": {
        "platform": "url",
        "url": "https://huggingface.co/datasets/ymhao/HPDv2/resolve/main/train.json",
    },
    
    # Reward Models & Processors
    "MizzenAI/HPSv3": {
        "platform": "huggingface",
        "repo_id": "MizzenAI/HPSv3",
    },
    "xswu/HPSv2": {
        "platform": "huggingface",
        "repo_id": "xswu/HPSv2",
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "platform": "huggingface",
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
    },
    "Qwen/Qwen3-VL-32B-Instruct": {
        "platform": "huggingface",
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct",
    },
    "Qwen/Qwen3.5-9B": {
        "platform": "huggingface",
        "repo_id": "Qwen/Qwen3.5-9B",
    },
    "Qwen/Qwen3.5-27B": {
        "platform": "huggingface",
        "repo_id": "Qwen/Qwen3.5-27B",
    },
    "zai-org/ImageReward": {
        "platform": "huggingface",
        "repo_id": "zai-org/ImageReward",
    },
    "yuvalkirstain/PickScore_v1": {
        "platform": "huggingface",
        "repo_id": "yuvalkirstain/PickScore_v1",
    },
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
        "platform": "huggingface",
        "repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    },
    "openai/clip-vit-large-patch14": {
        "platform": "huggingface",
        "repo_id": "openai/clip-vit-large-patch14",
    },
    "shunk031/aesthetics-predictor-v1-vit-large-patch14": {
        "platform": "huggingface",
        "repo_id": "shunk031/aesthetics-predictor-v1-vit-large-patch14",
    },
    "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE": {
        "platform": "huggingface",
        "repo_id": "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
    },
    "zai-org/GLM-4.6V-Flash": {
        "platform": "huggingface",
        "repo_id": "zai-org/GLM-4.6V-Flash",
    },
    "OpenGVLab/InternVL3_5-8B-HF": {
        "platform": "huggingface",
        "repo_id": "OpenGVLab/InternVL3_5-8B-HF",
    },
    "PaddlePaddle/PP-OCRv5_server_rec": {
        "platform": "huggingface",
        "repo_id": "PaddlePaddle/PP-OCRv5_server_rec",
    },
    "PaddlePaddle/PP-OCRv5_server_det": {
        "platform": "huggingface",
        "repo_id": "PaddlePaddle/PP-OCRv5_server_det",
    },
}


def download_repo(repo_id: str, cache_dir: str | None = None, token: str | None = None) -> None:
    """Download a specific repo (model/dataset) from its designated platform."""
    if repo_id not in PREDEFINED_REPOS:
        print(f"❌ Error: Repo '{repo_id}' is not predefined in the project.", file=sys.stderr)
        print("Available repos:", ", ".join(PREDEFINED_REPOS.keys()), file=sys.stderr)
        sys.exit(1)

    info = PREDEFINED_REPOS[repo_id]
    platform = info["platform"]

    print(f"🚀 Starting download for '{repo_id}' from {platform}...")

    if platform == "huggingface":
        hf_repo_id = info["repo_id"]
        # Determine the target directory for HF
        base_dir = cache_dir or os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub")
        # Create directory structure matching Hugging Face repo ID (e.g., org_name/repo_name)
        target_dir = os.path.join(base_dir, hf_repo_id)
        
        print(f"🔍 Checking and downloading '{repo_id}' to <{target_dir}>...")
        try:
            # hf_download automatically handles integrity checks (ETag validation).
            # If files are complete, it skips downloading. If missing/corrupted, it resumes/overwrites.
            hf_download(repo_id=hf_repo_id, local_dir=target_dir, token=token)
            print(f"✅ Repo '{repo_id}' is ready at <{target_dir}>")
        except Exception as e:
            print(f"❌ Error downloading from huggingface: {e}", file=sys.stderr)
            sys.exit(1)
    elif platform == "url":
        url = info["url"]
        base_dir = cache_dir or os.path.expanduser("~/.cache/nexus_align/downloads")
        target_dir = os.path.join(base_dir, repo_id)
        os.makedirs(target_dir, exist_ok=True)
        filename = url.split("/")[-1]
        target_path = os.path.join(target_dir, filename)
        
        print(f"⬇️ Downloading '{repo_id}' to <{target_path}>...")
        try:
            if not os.path.exists(target_path):
                import urllib.request
                import shutil
                req = urllib.request.Request(url)
                if token:
                    req.add_header("Authorization", f"Bearer {token}")
                with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            print(f"✅ Repo '{repo_id}' is ready at <{target_path}>")
        except Exception as e:
            print(f"❌ Error downloading from url: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"❌ Error: Unsupported platform '{platform}' for repo '{repo_id}'.", file=sys.stderr)
        sys.exit(1)


def run_download(args) -> int:
    """CLI entry point for downloading repos."""
    repo_id = args.repo_id
    cache_dir = args.cache_dir
    token = getattr(args, "token", None)

    download_repo(repo_id, cache_dir, token)
    
    return 0
