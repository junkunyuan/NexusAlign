"""ImageReward reward server: BLIP-based human preference score, under gunicorn.

Runs in .venv-imagereward, as ImageReward needs timm 0.6 and transformers 4; launch with
scripts/tools/reward_server.sh.

References:
    - ImageReward:
        - Official code: https://github.com/zai-org/ImageReward
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
          (flow_grpo/imagereward_scorer.py)
"""

import os
import pickle
import traceback

import torch
from PIL import Image
from flask import Flask, request
from transformers import BertTokenizer

import ImageReward as RM
from ImageReward.models.BLIP import blip_pretrain


def create_app(model_dir: str, bert_path: str) -> Flask:
    """Build the app; a POST of pickled {"images", "prompts"} returns {"scores"}."""

    def init_tokenizer():
        # ImageReward's init_tokenizer, reading bert-base-uncased from a local dir.
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer

    blip_pretrain.init_tokenizer = init_tokenizer
    model = RM.load(
        os.path.join(model_dir, "ImageReward.pt"),
        device="cuda",
        med_config=os.path.join(model_dir, "med_config.json"),
    )
    model.eval().to(dtype=torch.float32).requires_grad_(False)
    app = Flask(__name__)

    @app.route("/", methods=["POST"])
    def inference():
        try:
            data = pickle.loads(request.get_data())
            scores = []
            with torch.no_grad():  # one image per call, as Flow-GRPO
                for prompt, image in zip(data["prompts"], data["images"]):
                    _, reward = model.inference_rank(prompt, [Image.fromarray(image)])
                    scores.append(reward)
            return pickle.dumps({"scores": scores}), 200
        except Exception:
            response = traceback.format_exc()
            print(response)
            return response.encode("utf-8"), 500

    return app
