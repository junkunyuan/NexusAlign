"""
ImageReward reward server: human preference score with BLIP, under gunicorn.

It runs in .venv-imagereward, because ImageReward needs timm 0.6 and transformers 4.
Launch it with scripts/tools/reward_server.sh.

References:
    - ImageReward:
        - Official code: https://github.com/zai-org/ImageReward
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
          (flow_grpo/imagereward_scorer.py)
"""

import os

import torch
from PIL import Image
from flask import Flask
from transformers import BertTokenizer

import ImageReward as RM
from ImageReward.models.BLIP import blip_pretrain

from common import pickle_route


def create_app(model_dir: str, bert_path: str) -> Flask:
    """Build the app. A POST of pickled {"images", "prompts"} returns {"scores"}."""

    def init_tokenizer():
        # ImageReward's init_tokenizer, with bert-base-uncased from a local directory.
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

    def inference(data):
        scores = []
        with torch.no_grad():  # one image per call, as Flow-GRPO does
            for prompt, image in zip(data["prompts"], data["images"]):
                _, reward = model.inference_rank(prompt, [Image.fromarray(image)])
                scores.append(reward)
        return {"scores": scores}

    pickle_route(app, "/", inference)
    return app
