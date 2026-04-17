"""OCR-based text rendering metrics reward model.

Uses PaddleOCR-VL-1.5 (via transformers) to extract text from generated
images and computes word-level precision, recall, F1, and accuracy metrics
inspired by TextDiffuser (https://arxiv.org/abs/2305.10855).

Target text is extracted from quoted substrings in the prompt (e.g.
``'hello'`` or ``"world"``).  If no quotes are found the full prompt is used.
"""

import re

import torch

from nexus_align.core.base_reward_model import BaseRewardModel


class OCRMetrics(BaseRewardModel):
    """Word-level OCR metrics reward model using PaddleOCR-VL-1.5.

    Loads the VLM via ``transformers`` (AutoModelForImageTextToText) and
    uses the ``"OCR:"`` prompt to extract text from each generated image.

    Computes word-level precision, recall, F1, and accuracy by comparing
    OCR-detected words from the generated image against ground-truth words
    extracted from the prompt.

    The matching algorithm follows TextDiffuser's ``get_p_r_acc``: iterate
    over predicted words, remove each matched word from both pred and gt
    copies (prevents double-counting), then compute set-based metrics.

    Accuracy is defined as: sorted predicted words joined == sorted GT words
    joined (i.e. same multiset of words, regardless of order).
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("OCRMetrics", device, kwargs)

        self.model, self.processor = self.load_model()

        valid_keys = ("word_precision", "word_recall", "word_f1", "word_acc")
        if self.reward_key is None:
            self.reward_key = "word_f1"
        if self.reward_key not in valid_keys:
            raise ValueError(
                f"❌ Invalid reward_key: '{self.reward_key}'. Valid: {valid_keys}"
            )

        self.dataset_kwargs = {"image_open": True}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode, reward_key: {self.reward_key})")

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

    def _ocr_image(self, image) -> list[str]:
        """Extract all detected text lines from a PIL image via PaddleOCR-VL-1.5.

        Sends the image with an ``"OCR:"`` prompt to the VLM and splits the
        decoded output by newlines.  Returns a list of recognized text strings
        so callers can apply their own tokenization.
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
        ).strip()

        return text.split("\n") if text else []

    @staticmethod
    def _compute_word_metrics(
        pred_words: list[str], gt_words: list[str]
    ) -> tuple[float, float, float, float]:
        """Compute word-level precision, recall, F1, and accuracy.

        Follows TextDiffuser's ``get_p_r_acc`` matching logic:
        iterate over predicted words; if a word exists in the GT copy,
        remove it from both copies (prevents double-counting duplicates).

        Accuracy: 1 if sorted(pred) joined == sorted(gt) joined, else 0.
        This checks that pred and gt contain exactly the same multiset of
        words (order-independent exact match).

        Args:
            pred_words: Tokenized OCR output words (lowercased).
            gt_words:   Tokenized GT words (lowercased).

        Returns:
            (precision, recall, f1, accuracy) all in [0, 1].
        """
        pred = [w.strip() for w in pred_words]
        gt = [w.strip() for w in gt_words]

        pred_copy = pred.copy()
        gt_copy = gt.copy()

        for p in pred:
            if p in gt_copy:
                pred_copy.remove(p)
                gt_copy.remove(p)

        matched = len(pred) - len(pred_copy)
        precision = matched / (len(pred) + 1e-8)
        recall = matched / (len(gt) + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = 1.0 if "".join(sorted(pred)) == "".join(sorted(gt)) else 0.0

        return precision, recall, f1, acc

    _METRIC_KEYS = ("word_precision", "word_recall", "word_f1", "word_acc")

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """Evaluate word-level text rendering quality for a batch.

        Args:
            data (`dict`):
                image_pil (`list`): List of PIL.Image objects.
                text (`list`): List of text prompts (target text is
                    auto-extracted from quoted substrings).
            return_tensor (`bool`): Whether to return a tensor of word_f1
                scores (for compatibility). The full dict is always computed.

        Returns:
            `list[dict]`: One dict per sample with keys:
                - ``word_precision``: fraction of predicted words in GT
                - ``word_recall``:    fraction of GT words in prediction
                - ``word_f1``:        harmonic mean of precision and recall
                - ``word_acc``:       1 if multisets are identical, else 0
        """
        images = data["image_pil"]
        prompts = data["text"]

        results: list[dict | None] = []
        failed_indices: list[int] = []

        for i, (img, prompt) in enumerate(zip(images, prompts)):
            try:
                target = self._extract_target_text(prompt).lower()
                gt_words = target.split()

                ocr_lines = self._ocr_image(img)
                ocr_text = " ".join(ocr_lines).lower()
                pred_words = ocr_text.split()

                word_p, word_r, word_f1, word_acc = self._compute_word_metrics(
                    pred_words, gt_words
                )

                results.append({
                    "word_precision": word_p,
                    "word_recall": word_r,
                    "word_f1": word_f1,
                    "word_acc": word_acc,
                })
            except Exception as e:
                print(f"⚠️ OCR failed for image {i}: {e}")
                results.append(None)
                failed_indices.append(i)

        if failed_indices:
            successful_results = [r for r in results if r is not None]
            if not successful_results:
                raise RuntimeError(
                    "All images in batch failed OCR, cannot compute fallback average."
                )
            avg_result = {
                key: sum(r[key] for r in successful_results) / len(successful_results)
                for key in self._METRIC_KEYS
            }
            if avg_result[self.reward_key] == 0.0:
                raise RuntimeError(
                    f"Average {self.reward_key} of successful images is 0, "
                    "cannot use as fallback."
                )
            for idx in failed_indices:
                print(
                    f"⚠️ Replacing failed image {idx} metrics with batch average: "
                    f"{', '.join(f'{k}={v:.4f}' for k, v in avg_result.items())}"
                )
                results[idx] = avg_result.copy()

        if return_tensor:
            return torch.tensor(
                [r[self.reward_key] for r in results],
                device=self.device,
            ).contiguous()

        return results
