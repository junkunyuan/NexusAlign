"""Normalized Edit Distance reward model for evaluating image generation.

Uses PaddleOCR-VL-1.5 (via transformers) to extract text from generated
images and computes the normalized Levenshtein edit distance against target
text extracted from the prompt.  Returns ``1 - NED`` as the reward so that
higher means better.

Target text is extracted from quoted substrings in the prompt (e.g.
``'hello'`` or ``"world"``).  If no quotes are found the full prompt is used.
"""

import re

import torch

from nexus_align.core.base_reward_model import BaseRewardModel


class NormalizedEditDistance(BaseRewardModel):
    """
    Normalized Edit Distance reward model using PaddleOCR-VL-1.5.

    Loads the VLM via ``transformers`` (AutoModelForImageTextToText) and
    uses the ``"OCR:"`` prompt to extract text from each generated image.
    The recognized text is compared against the target via normalized
    Levenshtein edit distance.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("NormalizedEditDistance", device, kwargs)

        self.model, self.processor = self.load_model()

        self.dataset_kwargs = {"image_open": True}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self):
        """Load PaddleOCR-VL-1.5 via transformers."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"⏳ Loading PaddleOCR-VL-1.5 from <{self.model_path}>")

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, torch_dtype=self.model_dtype,
        ).to(self.device).eval()

        processor = AutoProcessor.from_pretrained(self.model_path)

        return model, processor

    @staticmethod
    def _extract_target_text(prompt: str) -> str:
        """Return quoted substrings joined by space, or the full prompt."""
        matches = re.findall(
            r'"([^"]+)"|'
            r"'([^']+)'|"
            r'\u201c([^\u201d]+)\u201d',
            prompt,
        )
        if matches:
            return " ".join(m for group in matches for m in group if m)
        return prompt

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Standard DP Levenshtein distance."""
        if len(s1) < len(s2):
            return NormalizedEditDistance._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev = list(range(len(s2) + 1))
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(
                    curr[j] + 1,
                    prev[j + 1] + 1,
                    prev[j] + (0 if c1 == c2 else 1),
                ))
            prev = curr
        return prev[-1]

    def _ocr_image(self, image) -> str:
        """Extract all text from a PIL image via PaddleOCR-VL-1.5.

        Sends the image with an ``"OCR:"`` prompt to the VLM and returns
        the decoded text output as a single string.
        """
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "OCR:"},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=512)
        text = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:-1],
            skip_special_tokens=True,
        )
        return text.strip()

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """
        Evaluate text rendering quality for a batch of (image, text) pairs.

        Args:
            data (`dict`):
                image_pil (`list`): List of PIL.Image objects.
                text (`list`): List of text prompts (target text is auto-extracted
                    from quoted substrings).
            return_tensor (`bool`): Whether to return a tensor.

        Returns:
            `list` | `torch.Tensor`: Reward scores in [0, 1], higher is better.
        """
        images = data["image_pil"]
        prompts = data["text"]

        scores: list[float | None] = []
        failed_indices: list[int] = []

        for i, (img, prompt) in enumerate(zip(images, prompts)):
            try:
                target = self._extract_target_text(prompt).lower()
                ocr_text = self._ocr_image(img).lower()

                if len(target) == 0 and len(ocr_text) == 0:
                    ned = 0.0
                elif len(target) == 0 or len(ocr_text) == 0:
                    ned = 1.0
                else:
                    dist = self._levenshtein_distance(ocr_text, target)
                    ned = dist / max(len(ocr_text), len(target))

                scores.append(1.0 - ned)
            except Exception as e:
                print(f"⚠️ OCR failed for image {i}: {e}")
                scores.append(None)
                failed_indices.append(i)

        if failed_indices:
            successful_scores = [s for s in scores if s is not None]
            if not successful_scores:
                raise RuntimeError(
                    "All images in batch failed OCR, cannot compute fallback average."
                )
            avg_score = sum(successful_scores) / len(successful_scores)
            if avg_score == 0.0:
                raise RuntimeError(
                    "Average score of successful images is 0, cannot use as fallback."
                )
            for idx in failed_indices:
                print(f"⚠️ Replacing failed image {idx} score with batch average: {avg_score:.4f}")
                scores[idx] = avg_score

        if return_tensor:
            return torch.tensor(scores, device=self.device).contiguous()
        else:
            return scores
