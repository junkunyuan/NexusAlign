"""
HPSv3 reward server: human preference score with Qwen2-VL-7B, under gunicorn.

It runs in .venv-hpsv3, because HPSv3 pins transformers 4.45.2. Launch it with
scripts/tools/reward_server.sh. Without flash-attn, HPSv3 uses SDPA.

References:
    - HPSv3:
        - Paper: HPSv3: Towards Wide-Spectrum Human Preference Score (ICCV 2025)
        - Official code: https://github.com/MizzenAI/HPSv3
"""

import os
import tempfile

import torch
from PIL import Image
from flask import Flask
from omegaconf import OmegaConf
from hpsv3 import HPSv3RewardInferencer
from hpsv3.inference import _MODEL_CONFIG_PATH

from common import pickle_route


def create_app(checkpoint_path: str, base_model_path: str) -> Flask:
    """Build the app. A POST of pickled {"images", "prompts"} returns {"scores"}."""
    # The packaged config names the base model by its hub id. Use the local copy.
    config = OmegaConf.load(os.path.join(_MODEL_CONFIG_PATH, "HPSv3_7B.yaml"))
    config.model_name_or_path = base_model_path
    config_path = os.path.join(tempfile.mkdtemp(), "HPSv3_7B.yaml")
    OmegaConf.save(config, config_path)
    inferencer = HPSv3RewardInferencer(
        config_path=config_path, checkpoint_path=checkpoint_path, device="cuda"
    )
    app = Flask(__name__)

    def inference(data):
        images = [Image.fromarray(image) for image in data["images"]]
        # Use keywords. The PyPI release and the repo order the arguments differently.
        with torch.inference_mode():
            rewards = inferencer.reward(image_paths=images, prompts=data["prompts"])
        scores = rewards[:, 0].float()  # the mean mu of the predicted (mu, sigma)
        return {"scores": scores.tolist()}

    pickle_route(app, "/", inference)
    return app
