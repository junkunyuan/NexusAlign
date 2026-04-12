"""Normalized Edit Distance reward model for evaluating image generation."""

import os
import re
import tempfile

import torch

from nexus_align.core.base_reward_model import BaseRewardModel


class NormalizedEditDistance(BaseRewardModel):
    """
    Normalized Edit Distance reward model for evaluating image generation.

    Uses PP-OCRv5 to extract text from generated images and
    computes the normalized Levenshtein edit distance against target text
    extracted from the prompt. Returns '1 - NED' as the reward so that higher
    means better.

    Target text is extracted from quoted substrings in the prompt (e.g.
    'hello' or "world"). If no quotes are found the full prompt is used.

    Expected model directory layout (pointed to by `model_path`)::
        PPOCRv5/
        ├── PP-OCRv5_server_det/   (inference.json + inference.pdiparams)
        └── PP-OCRv5_server_rec/   (inference.json + inference.pdiparams)
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("NormalizedEditDistance", device, kwargs)
        self.model_path_2 = os.path.join(
            kwargs["common"]["data_and_model_dir"], 
            kwargs["reward_model"]["path2"]
        )

        self.ocr_engine = self.load_model()

        self.dataset_kwargs = {"image_open": True}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

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
        """Extract all text from a PIL image via PP-OCRv5.

        PaddleOCR 3.x ``predict()`` expects a file path, so we write the
        PIL image to a temporary PNG and pass that path.
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
            return " ".join(all_texts).strip()
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _extract_rec_texts(res) -> list[str]:
        """Extract ``rec_texts`` from a PaddleOCR 3.x ``OCRResult``.

        ``OCRResult`` inherits from ``dict``, so ``rec_texts`` is a direct
        key.  The ``{'res': {...}}`` wrapper seen in printed output is only
        produced by ``_to_str()`` for display purposes.
        """
        try:
            return list(res["rec_texts"])
        except (KeyError, TypeError):
            pass

        texts = getattr(res, "rec_texts", None)
        if texts is not None:
            return list(texts)

        return []

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

        scores = []
        for img, prompt in zip(images, prompts):
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

        if return_tensor:
            return torch.tensor(scores, device=self.device).contiguous()
        else:
            return scores
