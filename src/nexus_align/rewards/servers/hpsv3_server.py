"""HPSv3 reward server: Qwen2-VL-7B human preference score, under gunicorn.

Runs in .venv-hpsv3, as HPSv3 pins transformers 4.45.2; launch with
scripts/tools/reward_server.sh. Without flash-attn installed, HPSv3 falls back to SDPA.

References:
    - HPSv3:
        - Paper: HPSv3: Towards Wide-Spectrum Human Preference Score
        - Official code: https://github.com/MizzenAI/HPSv3
"""

import os
import pickle
import tempfile
import traceback

import torch
from PIL import Image
from flask import Flask, request
from omegaconf import OmegaConf
from hpsv3 import HPSv3RewardInferencer
from hpsv3.inference import _MODEL_CONFIG_PATH


def create_app(checkpoint_path: str, base_model_path: str) -> Flask:
    """Build the app; a POST of pickled {"images", "prompts"} returns {"scores"}."""
    # The packaged config names the base model by its hub id; use the local copy.
    config = OmegaConf.load(os.path.join(_MODEL_CONFIG_PATH, "HPSv3_7B.yaml"))
    config.model_name_or_path = base_model_path
    config_path = os.path.join(tempfile.mkdtemp(), "HPSv3_7B.yaml")
    OmegaConf.save(config, config_path)
    inferencer = HPSv3RewardInferencer(
        config_path=config_path, checkpoint_path=checkpoint_path, device="cuda"
    )
    app = Flask(__name__)

    @app.route("/", methods=["POST"])
    def inference():
        try:
            data = pickle.loads(request.get_data())
            images = [Image.fromarray(image) for image in data["images"]]
            # Keywords, as the PyPI release and the repo order the arguments differently.
            with torch.inference_mode():
                rewards = inferencer.reward(image_paths=images, prompts=data["prompts"])
            scores = rewards[:, 0].float()  # the mean mu of the predicted (mu, sigma)
            return pickle.dumps({"scores": scores.tolist()}), 200
        except Exception:
            response = traceback.format_exc()
            print(response)
            return response.encode("utf-8"), 500

    return app
