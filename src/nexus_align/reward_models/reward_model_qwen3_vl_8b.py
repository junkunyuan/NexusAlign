"""Qwen3-VL reward model for evaluating image generation."""

import json
import re
import torch

from nexus_align.core import BaseRewardModel

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


class Qwen3_VL_8B(BaseRewardModel):
    """
    Qwen3-VL reward model for evaluating image generation.
    
    References:
        - Qwen3-VL paper: https://arxiv.org/pdf/2511.21631.
        - Official repo: https://github.com/QwenLM/Qwen3-VL.
        - Checkpoint: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("Qwen3-VL-8B", device, kwargs)

        self.model, self.processor = self.load_model()

        self.dataset_kwargs = {}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load model.

        Follow the repo: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct.
        """
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        # Load components
        print(f"⏳ Loading {self.model_name} processor from <{self.model_path}>")
        processor = AutoProcessor.from_pretrained(self.model_path)
        print(f"⏳ Loading {self.model_name} model from <{self.model_path}>")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            dtype=self.model_dtype, 
            device_map=f"cuda:{self.device.index}"
        )

        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        # Set model mode
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

        Follow the repo: https://github.com/QwenLM/Qwen3-VL.
        
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

        # Qwen3-VL needs to disable deterministic
        deter_status = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)

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
                # Fallback: try regex extraction, otherwise mark as None for later imputation
                match = re.search(r'"overall_score"\s*:\s*(\d+)', res)
                if match:
                    overall_scores.append(float(match.group(1)))
                else:
                    overall_scores.append(None)

        torch.use_deterministic_algorithms(deter_status)

        # Impute missing scores with the batch average (or 50.0 if all failed)
        valid_scores = [s for s in overall_scores if s is not None]
        batch_avg = sum(valid_scores) / len(valid_scores) if valid_scores else 50.0
        overall_scores = [s if s is not None else batch_avg for s in overall_scores]

        if return_tensor:
            return torch.tensor(overall_scores, device=self.device).contiguous()  
        else:
            return overall_scores
