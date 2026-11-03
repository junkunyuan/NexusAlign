"""
DeQA reward server: DeQA-Score image quality from 1 to 5, under gunicorn.

It runs in .venv-deqa, because the mPLUG-Owl2 code of DeQA-Score needs
transformers 4.36. Launch it with scripts/tools/reward_server.sh.

References:
    - DeQA-Score:
        - Official code: https://github.com/zhiyuanyou/DeQA-Score
    - Flow-GRPO:
        - Reward server code: https://github.com/yifan123/reward-server
          (reward_server/deqa.py)
"""

from io import BytesIO

import torch
from PIL import Image
from flask import Flask
from transformers import AutoModelForCausalLM

from common import pickle_route


def create_app(model_path: str) -> Flask:
    """Build the app. A POST of pickled {"images"} as JPEG bytes returns {"scores"}."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map=None,
    ).cuda()
    model.requires_grad_(False)
    app = Flask(__name__)

    def inference(data):
        images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
        with torch.no_grad():
            scores = model.score(images)
        return {"scores": scores.tolist()}

    pickle_route(app, "/", inference)
    return app
