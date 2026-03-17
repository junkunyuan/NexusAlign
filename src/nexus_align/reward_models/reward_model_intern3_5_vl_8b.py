"""Intern3.5-VL reward model for evaluating image generation."""

import json
import re
import torch

from nexus_align.core.base_reward_model import BaseRewardModel

EVAL_MAX_NEW_TOKENS = 32
EVAL_KEYS = ["aesthetic_quality_score", "semantic_alignment_score", "overall_score"]
EVAL_INSTRUCTION = """
You are a professional image generation evaluation expert. Evaluate the image using the following two criteria.

- Aesthetic Quality (0-100 points):
    - Visual Appeal: Evaluate color harmony, lighting, composition, and style consistency.
    - Realism/Detail: Assess the fidelity of details and textures.
- Semantic Alignment (0-100 points):
    - Prompt Fidelity: How accurately does the image reflect all key subjects, attributes, and actions specified in the text prompt?
    - Contextual Consistency: Are all elements within the image logically and spatially consistent with each other?
- Overall Evaluation (0-100 points):
    - Overall: Evaluate the image based on the two criteria above. The overall score should reflect a balanced assessment, typically around 50 points for average quality images.

IMPORTANT SCORING GUIDELINES:
- Be strict and balanced in your scoring. The average score across all evaluations should be around 50 points.
- Reserve scores above 80 for truly outstanding images, and scores below 20 for significantly flawed images.

Return the evaluation result in JSON format:
{
  "aesthetic_quality_score": integer,
  "semantic_alignment_score": integer,
  "overall_score": integer
}
"""


class Intern3_5_VL_8B(BaseRewardModel):
    """
    Intern3.5-VL reward model for evaluating image generation.

    References:
        - Intern3.5-VL paper: https://arxiv.org/pdf/2508.18265.
        - Official repo: https://github.com/OpenGVLab/InternVL.
        - Checkpoint: https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("Intern3.5-VL-8B", device, kwargs)
        self.model, self.processor = self.load_model()
        self.dataset_kwargs = {"image_open": True}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load model.

        Follow the repo: https://github.com/OpenGVLab/InternVL
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText

        print(f"⏳ Loading {self.model_name} processor from <{self.model_path}>")
        processor = AutoProcessor.from_pretrained(self.model_path)
        print(f"⏳ Loading {self.model_name} model from <{self.model_path}>")
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.model_dtype,
            device_map=f"cuda:{self.device.index}",
        )
        model.generation_config.eos_token_id = 151645
        model.generation_config.pad_token_id = 151645

        if self.mode == "train":
            model.train()
        elif self.mode == "eval":
            model.eval()
            model.requires_grad_(False)
        else:
            raise ValueError(f"❌ Invalid mode: {self.mode}")

        return model, processor

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """
        Evaluate a batch of (image, text) pairs.

        Follow the repo: https://github.com/OpenGVLab/InternVL.
        
        Args:
            data (`dict`):
                image (`list`): List of paths to images.
                text (`list`): List of text strings.
            return_tensor (`bool`): Whether to return a tensor.

        Returns:
            `list` | `torch.Tensor`: Scores between each pair of images and texts.
        """
        images = data["image"]
        prompts = data["text"]

        overall_scores = []
        for image, prompt in zip(images, prompts):
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": EVAL_INSTRUCTION.strip()},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Text prompt: {prompt}"},
                        {"type": "image", "image": f"{image}"},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            res = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            scores = {}
            try:
                res = re.sub(r"<\|.*?\|>", "", res).strip()
                # Strip markdown code fences (```json ... ``` or ``` ... ```)
                res = re.sub(r"```(?:json)?\s*", "", res).strip()
                res = res.strip("`").strip()
                eval_result = json.loads(res)

                for key in EVAL_KEYS:
                    scores[key] = int(eval_result[key])

                overall_scores.append(scores["overall_score"])
            except Exception as e:
                print(
                    f"⚠️ Warning: JSON parsing error, the format of the model's output is incorrect:\n {e}"
                )
                print(f"Model's output: {res}")
                overall_scores.append(None)

        if return_tensor:
            return torch.tensor(overall_scores, device=self.device).contiguous()  
        else:
            return overall_scores
