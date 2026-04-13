"""OCR-based text rendering metrics reward model.

Uses PP-OCRv5 to extract text from generated images and computes
word-level precision, recall, F1, and accuracy metrics inspired by
TextDiffuser (https://arxiv.org/abs/2305.10855).

Target text is extracted from quoted substrings in the prompt (e.g.
``'hello'`` or ``"world"``). If no quotes are found the full prompt is used.

Expected model directory layout (pointed to by ``model_path``)::
    PPOCRv5/
    ├── PP-OCRv5_server_det/   (inference.json + inference.pdiparams)
    └── PP-OCRv5_server_rec/   (inference.json + inference.pdiparams)
"""

import os
import re
import tempfile

import torch

from nexus_align.core.base_reward_model import BaseRewardModel


class OCRMetrics(BaseRewardModel):
    """Word-level OCR metrics reward model for evaluating text rendering quality.

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
        self.model_path_2 = os.path.join(
            kwargs["common"]["data_and_model_dir"],
            kwargs["reward_model"]["path2"]
        )

        self.ocr_engine = self.load_model()

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
        """Load PP-OCRv5 pipeline from local model directory."""
        from paddleocr import PaddleOCR

        print(f"⏳ Loading PP-OCRv5_server_rec from <{self.model_path}>")
        print(f"⏳ Loading PP-OCRv5_server_det from <{self.model_path_2}>")

        device_str = f"gpu:{self.device.index}"

        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_detection_model_dir=self.model_path_2,
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_recognition_model_dir=self.model_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device_str,
        )
        return ocr

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
    def _extract_rec_texts(res) -> list[str]:
        """Extract ``rec_texts`` from a PaddleOCR 3.x ``OCRResult``."""
        try:
            return list(res["rec_texts"])
        except (KeyError, TypeError):
            pass

        texts = getattr(res, "rec_texts", None)
        if texts is not None:
            return list(texts)

        return []

    def _ocr_image(self, image) -> list[str]:
        """Extract all detected text lines from a PIL image via PP-OCRv5.

        Returns a list of recognized text strings, one per detected text
        region (``rec_texts``). The raw list is returned rather than a joined
        string so callers can apply their own tokenization.
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tmp:
                image.save(tmp, format="PNG")
                tmp_path = tmp.name

            results = self.ocr_engine.predict(input=tmp_path)

            all_texts: list[str] = []
            for res in results:
                rec_texts = self._extract_rec_texts(res)
                if rec_texts:
                    all_texts.extend(rec_texts)
            return all_texts
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

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

        results = []
        for img, prompt in zip(images, prompts):
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

        if return_tensor:
            return torch.tensor(
                [r[self.reward_key] for r in results],
                device=self.device,
            ).contiguous()

        return results
