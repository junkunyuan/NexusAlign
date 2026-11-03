"""
GenEval reward: object, count, color, and position checks by a detector and CLIP.

The Mask2Former detector runs in the GenEval server. See
nexus_align/rewards/servers/geneval_server.py.

References:
    - GenEval:
        - Paper: GenEval: An Object-Focused Framework for Evaluating Text-to-Image
          Alignment (NeurIPS 2023)
        - Reward server code: https://github.com/yifan123/reward-server
"""

from collections import defaultdict

import numpy as np
import torch

from nexus_align.rewards.remote import RemoteReward, encode_image, to_uint8


class GenEval(RemoteReward):
    """
    Client of the GenEval server. It sends the images in batches, as JPEG or PNG bytes.

    - geneval: the partial score, which is the training reward.
    - accuracy / strict_accuracy: GenEval correctness. The server calculates accuracy
      only if only_strict is False, that is, at evaluation.
    - <tag>_accuracy / <tag>_strict_accuracy: correctness per GenEval task, and -10
      for the prompts of other tasks.
    """

    metrics = ("accuracy", "strict_accuracy", "geneval")  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        super().__init__(cfg_reward)
        self.batch_size = cfg_reward.batch_size
        self.jpeg = cfg_reward.jpeg

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = to_uint8(images)
        num_chunks = int(np.ceil(len(images) / self.batch_size))
        results = defaultdict(list)
        for image_batch, metadata_batch in zip(
            np.array_split(images, num_chunks), np.array_split(metadatas, num_chunks)
        ):
            path, image_format = ("", "JPEG") if self.jpeg else ("/png", "PNG")
            image_batch = [encode_image(image, image_format) for image in image_batch]
            response = self.post({
                "images": image_batch,
                "meta_datas": list(metadata_batch),
                "only_strict": only_strict,
            }, path)
            results["geneval"] += response["scores"]
            results["accuracy"] += response["rewards"]
            results["strict_accuracy"] += response["strict_rewards"]
            for key, value in response["group_rewards"].items():
                results[f"{key}_accuracy"] += value
            for key, value in response["group_strict_rewards"].items():
                results[f"{key}_strict_accuracy"] += value
        if only_strict:  # the server skips the non-strict check, so accuracy is all 0
            results = {
                k: v for k, v in results.items() if "accuracy" not in k or "strict" in k
            }
        return dict(results)
