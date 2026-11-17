"""
Qwen judge reward server: the expected holistic score of a vision-language judge.

The judge of the TwinTilt code has 27.8 B parameters, which is 56 GB in bf16, so this
server holds one copy per node across its GPUs and serves every training process of
that node. It runs in .venv-qwen_judge. Launch it with scripts/tools/reward_server.sh.

References:
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private)
            - src/nexus_align/rewards/qwen_judge.py: QwenJudge.score and next_digit
"""

import torch
from PIL import Image
from flask import Flask
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import pickle_route

JUDGE_PROMPT = """\
You are an expert evaluator of AI-generated images. You will be given a text
prompt and an image generated from it. Evaluate the image objectively and
assign a single holistic score between 0 and 1.

Consider the following aspects together:

1. Instruction following: Does the image accurately reflect all elements of
   the prompt (objects, attributes, counts, actions, spatial relations, text,
   style, and any knowledge or reasoning the prompt implies)?
2. Plausibility: Is the image structurally and physically coherent within its
   style (anatomy, proportions, perspective, lighting, shadows, and object
   interactions)?
3. Aesthetics: Is the image visually well-crafted (composition, color harmony,
   lighting, and clarity)?

Assign a score between 0 and 1 that reflects the overall quality of the image,
where 0 indicates the lowest possible quality and 1 indicates the highest.

Respond only with valid JSON in the following format:
{"score": <a number between 0 and 1, with two decimals>}"""
ANSWER_PREFIX = '{"score": 0.'  # the forced start of the answer


def expected_digits(logits: torch.Tensor, digit_ids: torch.Tensor) -> tuple:
    """Return the expected digit and the most likely digit of each row of logits."""
    probs = logits[:, digit_ids].float().softmax(dim=-1)
    digits = torch.arange(10, device=probs.device, dtype=probs.dtype)
    return probs @ digits, probs.argmax(dim=-1)


class Judge:
    """The judge model of one node, sharded over its visible GPUs."""

    def __init__(self, model_path: str, dtype: str, batch_size: int) -> None:
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"  # the last token is the answer
        # device_map spreads the layers over the GPUs of this node, so the judge and
        # the policy of every rank fit together.
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype={"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype],
            device_map="auto",
        )
        self.model.eval().requires_grad_(False)
        self.device = next(self.model.parameters()).device

        tokenizer = self.processor.tokenizer
        ids = [tokenizer.encode(str(d), add_special_tokens=False) for d in range(10)]
        if any(len(i) != 1 for i in ids):
            raise ValueError("❌ The judge tokenizer splits a digit into tokens")
        self.digit_ids = torch.tensor([i[0] for i in ids], device=self.device)
        self.prefix_ids = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
        print(f"✅ Prepared the Qwen judge from <{model_path}>")

    @torch.no_grad()
    def score_all(self, images: list, prompts: list[str]) -> list[float]:
        scores = []
        for i in range(0, len(images), self.batch_size):
            batch = slice(i, i + self.batch_size)
            scores += self.score(images[batch], prompts[batch]).tolist()
        return scores

    def score(self, images: list, prompts: list[str]) -> torch.Tensor:
        """Score a batch of images with the forced answer prefix."""
        texts = [
            self.processor.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": JUDGE_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"Text prompt: {prompt}"},
                        ],
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        inputs = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(self.device)
        prefix = torch.tensor(self.prefix_ids, device=self.device)
        tenths, tenth_digits = self.next_digit(inputs, prefix.expand(len(texts), -1))
        tenth_ids = self.digit_ids[tenth_digits][:, None]
        answer = torch.cat([prefix.expand(len(texts), -1), tenth_ids], dim=1)
        hundredths, _ = self.next_digit(inputs, answer)
        return 0.1 * tenths + 0.01 * hundredths

    def next_digit(self, inputs, answer: torch.Tensor) -> tuple:
        """Get the expected and the most likely digit after the prompt and answer."""
        answer_inputs = {
            "input_ids": answer,
            "attention_mask": torch.ones_like(answer),
            "mm_token_type_ids": torch.zeros_like(answer),
        }
        # Only Qwen3-VL processors return mm_token_type_ids, so extend the per-token
        # inputs that this processor made and leave the others alone.
        forced = {
            key: torch.cat([inputs[key], value.to(inputs[key].dtype)], dim=1)
            for key, value in answer_inputs.items() if key in inputs
        }
        logits = self.model(
            **{**inputs, **forced}, use_cache=False, logits_to_keep=1
        ).logits[:, -1]
        return expected_digits(logits, self.digit_ids)


def create_app(model_path: str, dtype: str = "bf16", batch_size: int = 8) -> Flask:
    """Build the app. A POST of pickled {"images", "prompts"} returns {"scores"}."""
    judge = Judge(model_path, dtype, int(batch_size))
    app = Flask(__name__)

    def inference(data):
        images = [Image.fromarray(image) for image in data["images"]]
        return {"scores": judge.score_all(images, data["prompts"])}

    pickle_route(app, "/", inference)
    return app
