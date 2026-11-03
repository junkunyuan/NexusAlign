"""
OCR reward server: PaddleOCR scored against the quoted prompt text, under gunicorn.

It runs in .venv-ocr, because PaddleOCR needs numpy<2. Launch it with
scripts/tools/reward_server.sh.

References:
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo (flow_grpo/ocr.py)
"""

import os

import numpy as np
from flask import Flask
from Levenshtein import distance
from paddleocr import PaddleOCR

from common import pickle_route


class OcrScorer:
    """OCR reward of Flow-GRPO: 1 - edit distance / text length, on CPU."""

    def __init__(self, model_dir: str) -> None:
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,
            show_log=False,
            det_model_dir=os.path.join(model_dir, "det/en/en_PP-OCRv3_det_infer"),
            rec_model_dir=os.path.join(model_dir, "rec/en/en_PP-OCRv4_rec_infer"),
            cls_model_dir=os.path.join(model_dir, "cls/ch_ppocr_mobile_v2.0_cls_infer"),
        )

    def __call__(self, images: np.ndarray, prompts: list[str]) -> list[float]:
        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            try:
                result = self.ocr.ocr(img, cls=False)
                # Join the recognized lines, but skip the lines with zero confidence.
                recognized_text = ''.join(
                    [res[1][0] if res[1][1] > 0 else '' for res in result[0]]
                ) if result[0] else ''
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Many unrelated characters cost at most the full length.
                if dist > len(prompt):
                    dist = len(prompt)
            except Exception as e:
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)
            rewards.append(1 - dist / len(prompt))
        return rewards


def create_app(model_dir: str) -> Flask:
    """Build the app. A POST of pickled {"images", "prompts"} returns {"scores"}."""
    scorer = OcrScorer(model_dir)
    app = Flask(__name__)

    def inference(data):
        return {"scores": scorer(data["images"], data["prompts"])}

    pickle_route(app, "/", inference)
    return app
