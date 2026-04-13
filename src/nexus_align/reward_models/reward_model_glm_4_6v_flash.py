"""GLM-4.6V-Flash reward model for evaluating image generation."""

import re
import json
import torch

from nexus_align.core import BaseRewardModel
from nexus_align.reward_models.text_rendering_prompts import (
    TEXT_RENDERING_PROMPTS,
    TEXT_RENDERING_MAX_NEW_TOKENS,
    DIMENSIONS,
    PERFECT_SCORE,
)

EVAL_MAX_NEW_TOKENS = 128
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

class GLM_4_6V_Flash(BaseRewardModel):
    """
    GLM-4.6V-Flash reward model for evaluating image generation.

    References:
        - GLM-4 paper: https://arxiv.org/pdf/2406.12793.
        - Official repo: https://github.com/THUDM/GLM-4.
        - Checkpoint: https://huggingface.co/zai-org/GLM-4.6V-Flash.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("GLM-4.6V-Flash", device, kwargs)

        self.model, self.processor = self.load_model()

        self.dataset_kwargs = {}

        prompts_config = TEXT_RENDERING_PROMPTS.get(self.model_name, {})
        self._key_to_dim = {}
        for dim_name, dim_cfg in prompts_config.items():
            for key in dim_cfg["keys"]:
                self._key_to_dim[key] = dim_name

        if self.eval_keys is not None:
            all_valid_keys = set(self._key_to_dim.keys())
            invalid = set(self.eval_keys) - all_valid_keys
            if invalid:
                raise ValueError(f"❌ Invalid eval_keys: {invalid}. Valid: {sorted(all_valid_keys)}")
            self.active_keys = tuple(self.eval_keys)
            self.active_dimensions = tuple(dict.fromkeys(self._key_to_dim[k] for k in self.active_keys))
        else:
            self.active_keys = None
            if self.eval_dimensions is not None:
                invalid = set(self.eval_dimensions) - set(DIMENSIONS)
                if invalid:
                    raise ValueError(f"❌ Invalid eval_dimensions: {invalid}. Valid: {DIMENSIONS}")
                self.active_dimensions = tuple(self.eval_dimensions)
            else:
                self.active_dimensions = DIMENSIONS

        keys_info = list(self.active_keys) if self.active_keys else "all"
        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode, task: {self.task}, dims: {self.active_dimensions}, keys: {keys_info})")

    def load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load model.

        Follow the repo: https://huggingface.co/zai-org/GLM-4.6V-Flash.
        """
        from transformers import AutoProcessor, Glm4vForConditionalGeneration

        print(f"⏳ Loading {self.model_name} processor from <{self.model_path}>")
        processor = AutoProcessor.from_pretrained(self.model_path)
        print(f"⏳ Loading {self.model_name} model from <{self.model_path}>")

        self.cpu_offload = self.fsdp_kwargs.get("cpu_offload", False)
        if self.cpu_offload:
            model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_path,
                torch_dtype=self.model_dtype,
            )
        else:
            model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_path,
                torch_dtype=self.model_dtype,
                device_map=f"cuda:{self.device.index}",
            )

        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        if self.mode == "train":
            model.train()
        elif self.mode == "eval":
            model.eval()
            model.requires_grad_(False)
        else:
            raise ValueError(f"❌ Invalid mode: {self.mode}")

        return model, processor

    def _call_model_aesthetic(self, image, system_prompt: str, user_text: str, max_new_tokens: int) -> str:
        """Aesthetic evaluation: system message (instruction) + user message (prompt + image)."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": f"{image}"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _call_model_text_rendering(self, image, prompt_text: str, max_new_tokens: int) -> str:
        """Text rendering evaluation: single user message (image + prompt), thinking enabled."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{image}"},
                    {"type": "text", "text": prompt_text},
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
        inputs.pop("token_type_ids", None)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @staticmethod
    def _clean_response(res: str) -> str:
        """Strip thinking tags, special tokens, and markdown fences for aesthetic JSON."""
        res = re.sub(r"<think>.*?(</think>|$)", "", res, flags=re.DOTALL)
        res = re.sub(r"<\|.*?\|>", "", res).strip()
        res = re.sub(r"```(?:json)?\s*", "", res).strip()
        res = res.strip("`").strip()
        match = re.search(r"\{[^{}]*\}", res)
        if match:
            return match.group(0)
        return res

    @staticmethod
    def _parse_text_rendering_response(raw: str, keys: list[str]) -> dict | None:
        """Parse VLM response for text rendering, searching the full text (incl. thinking).

        Strategy (mirrors the original evaluation script):
          1. Find JSON objects in the full raw response (including <think> content)
          2. Fall back to key:value regex patterns
          3. Default missing keys to 1 (assume problem present)
        """
        result = {k: None for k in keys}

        for m in re.finditer(r"\{[^{}]*\}", raw):
            try:
                parsed = json.loads(m.group(0))
                for k in keys:
                    if k in parsed and result[k] is None:
                        val = int(parsed[k])
                        if val in (0, 1):
                            result[k] = val
                if all(v is not None for v in result.values()):
                    return result
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        for k in keys:
            if result[k] is not None:
                continue
            m = re.search(rf'["\']?{k}["\']?\s*[:=]\s*(\d)', raw)
            if m:
                val = int(m.group(1))
                if val in (0, 1):
                    result[k] = val

        if any(v is not None for v in result.values()):
            for k in keys:
                if result[k] is None:
                    result[k] = 1
            return result

        return None

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False, return_detail: bool = False) -> list | torch.Tensor:
        if self.cpu_offload:
            self.model.to(self.device)

        if self.task == "aesthetic":
            result = self._evaluate_aesthetic(data, return_tensor)
        elif self.task == "text_rendering":
            result = self._evaluate_text_rendering(data, return_tensor, return_detail)
        else:
            raise ValueError(f"❌ Unknown task: {self.task}")

        if self.cpu_offload:
            self.model.to("cpu")
            torch.cuda.empty_cache()

        return result

    def _evaluate_aesthetic(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """Original aesthetic + semantic alignment evaluation."""
        images = data["image"]
        prompts = data["text"]

        deter_status = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)

        overall_scores = []
        for image, prompt in zip(images, prompts):
            res = self._call_model_aesthetic(
                image,
                system_prompt=EVAL_INSTRUCTION.strip(),
                user_text=f"Text prompt: {prompt}",
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
            )

            scores = {}
            try:
                res = self._clean_response(res)
                eval_result = json.loads(res)

                for key in EVAL_KEYS:
                    scores[key] = int(eval_result[key])

                overall_scores.append(scores["overall_score"])
            except Exception as e:
                print(
                    f"⚠️ Warning: JSON parsing error, the format of the model's output is incorrect:\n {e}"
                )
                print(f"Model's output: {res}")
                overall_scores.append(0.0)

        torch.use_deterministic_algorithms(deter_status)

        if return_tensor:
            return torch.tensor(overall_scores, device=self.device).contiguous()
        else:
            return overall_scores

    def _evaluate_text_rendering(
        self, data: dict, return_tensor: bool = False, return_detail: bool = False,
    ) -> list | torch.Tensor:
        """Three-stage text rendering quality evaluation (no retry, discard on parse failure).

        When *return_detail* is True (evaluation mode), returns a
        ``list[dict | None]`` where each dict contains the aggregate
        ``"score"`` plus per-question 0/100 entries.
        """
        prompts_config = TEXT_RENDERING_PROMPTS[self.model_name]
        images = data["image"]
        ground_truths = data["text"]

        deter_status = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)

        overall_scores = []
        detailed_results: list[dict | None] = []
        total_calls = 0
        discard_count = 0
        num_evaluated_dims = 0

        for image, gt in zip(images, ground_truths):
            per_question: dict[str, int] = {}
            total_problems = 0
            valid_indicators = 0
            image_discarded = False

            for dim in self.active_dimensions:
                dim_cfg = prompts_config[dim]
                prompt_text = dim_cfg["prompt"].replace("{ground_truth}", gt)
                total_calls += 1

                raw = self._call_model_text_rendering(
                    image,
                    prompt_text=prompt_text,
                    max_new_tokens=TEXT_RENDERING_MAX_NEW_TOKENS,
                )
                result = self._parse_text_rendering_response(raw, dim_cfg["keys"])

                if result is None:
                    discard_count += 1
                    image_discarded = True
                    print(
                        f"⚠️ Discard [{dim}]: parse failed, raw output: {raw[:200]}"
                    )
                    continue

                num_evaluated_dims += 1
                scoring_keys = [k for k in dim_cfg["keys"] if k in self.active_keys] if self.active_keys else dim_cfg["keys"]
                for key in scoring_keys:
                    val = result.get(key, 0)
                    problem = int(val) if val in (0, 1) else 1
                    per_question[key] = 0 if problem else 100
                    total_problems += problem
                    valid_indicators += 1

            if image_discarded or valid_indicators == 0:
                overall_scores.append(float("nan"))
                detailed_results.append(None)
            else:
                score = (valid_indicators - total_problems) / valid_indicators * PERFECT_SCORE
                overall_scores.append(score)
                detailed_results.append({"score": score, **per_question})

        if discard_count > 0:
            print(f"⚠️ Parse discard summary: {discard_count}/{total_calls}")

        torch.use_deterministic_algorithms(deter_status)

        if return_detail:
            return detailed_results
        if return_tensor:
            return torch.tensor(overall_scores, dtype=torch.float32, device=self.device).contiguous()
        return overall_scores
