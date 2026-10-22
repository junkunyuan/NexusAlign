"""DeQA reward server: DeQA-Score image quality from 1 to 5, under gunicorn.

Runs in .venv-deqa, as the mPLUG-Owl2 code of DeQA-Score needs transformers 4.36;
launch with scripts/tools/reward_server.sh.

References:
    - DeQA-Score:
        - Official code: https://github.com/zhiyuanyou/DeQA-Score
    - Flow-GRPO:
        - Reward server code: https://github.com/yifan123/reward-server
          (reward_server/deqa.py)
"""

import pickle
import traceback
from io import BytesIO

import torch
from PIL import Image
from flask import Flask, request
from transformers import AutoModelForCausalLM


def create_app(model_path: str) -> Flask:
    """Build the app; a POST of pickled {"images"} as JPEG bytes returns {"scores"}."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map=None,
    ).cuda()
    model.requires_grad_(False)
    app = Flask(__name__)

    @app.route("/", methods=["POST"])
    def inference():
        try:
            data = pickle.loads(request.get_data())
            images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
            with torch.no_grad():
                scores = model.score(images)
            return pickle.dumps({"scores": scores.tolist()}), 200
        except Exception:
            response = traceback.format_exc()
            print(response)
            return response.encode("utf-8"), 500

    return app
