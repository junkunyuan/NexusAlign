"""Remote rewards: clients of reward servers that run in their own environments."""

import pickle
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter, Retry


def to_uint8(images: torch.Tensor) -> np.ndarray:
    """Convert NCHW images in [0, 1] to uint8 NHWC arrays."""
    images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return images.transpose(0, 2, 3, 1)  # NCHW -> NHWC


def encode_image(image: np.ndarray, image_format: str) -> bytes:
    """Encode one uint8 HWC image as bytes in a format such as JPEG or PNG."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format=image_format)
    return buffer.getvalue()


class RemoteReward:
    """
    Client of a reward server on this node. It posts pickled requests over HTTP.

    - Reward models with dependencies that conflict with training run as servers.
      See nexus_align/rewards/servers.
    - The client retries server errors a limited number of times, so a dead server
      makes the job fail, not hang.
    - By default, the client sends the images as raw uint8 arrays with their prompts,
      and returns the scores as the first headline metric.
    """

    def __init__(self, cfg_reward) -> None:
        self.name = cfg_reward.name
        self.url = cfg_reward.url
        self.timeout = cfg_reward.get("timeout", 120)
        self.session = requests.Session()
        self.session.trust_env = False  # the server is local, so skip any HTTP proxy
        retries = Retry(
            total=cfg_reward.get("max_retries", 10),
            backoff_factor=1,
            status_forcelist=[500],
            allowed_methods=False,
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def post(self, data: dict, path: str = "") -> dict:
        """Send one request to the url plus path and return the unpickled response."""
        data = pickle.dumps(data)
        response = self.session.post(self.url + path, data=data, timeout=self.timeout)
        response.raise_for_status()
        return pickle.loads(response.content)

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        response = self.post({"images": to_uint8(images), "prompts": list(prompts)})
        return {self.metrics[0]: response["scores"]}
