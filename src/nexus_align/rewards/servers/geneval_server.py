"""
GenEval reward server: reward-server's GenEval app with local weights, under gunicorn.

It runs in .venv-geneval, because mmdet 2.x needs torch 2.1 and mmcv-full. Launch it
with scripts/tools/reward_server.sh. The scoring code of reward-server is unchanged.
This file only sets its weight paths and adds a /png route for lossless images.

References:
    - Reward server code: https://github.com/yifan123/reward-server
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT (flow_grpo/gen_eval.py)
"""

import sys
import os
from pathlib import Path
from functools import partial
from io import BytesIO

if __package__:
    from .common import pickle_route
else:
    from common import pickle_route


def create_app(
    reward_server_dir: str, mmdet_config: str, ckpt_dir: str, clip_path: str,
    detector_checkpoint: str | None = None, clip_weights_only: bool = True,
):
    """Build reward-server's app. Its working directory must be reward_server_dir."""
    sys.path.insert(0, reward_server_dir)

    # pretrained="openai" in reward-server resolves to this local file, not the hub.
    import open_clip
    open_clip.pretrained.get_pretrained_cfg("ViT-L-14", "openai")["file"] = clip_path

    from reward_server import gen_eval
    gen_eval.MY_CONFIG_PATH = mmdet_config
    gen_eval.MY_CKPT_PATH = ckpt_dir

    import app_geneval
    from PIL import Image
    # Override resource loading only while the unmodified scoring app initializes.
    original_detector = gen_eval.init_detector
    original_clip = open_clip.create_model_and_transforms
    if detector_checkpoint is not None:
        def init_local_detector(config, checkpoint, **kwargs):
            return original_detector(config, detector_checkpoint, **kwargs)

        gen_eval.init_detector = init_local_detector
    if not clip_weights_only:
        open_clip.create_model_and_transforms = partial(
            original_clip, weights_only=False
        )
    try:
        app = app_geneval.create_app()
    finally:
        gen_eval.init_detector = original_detector
        open_clip.create_model_and_transforms = original_clip

    def inference_png(data):
        """Score the exact pixels, as DiffusionNFT's in-process GenEval does."""
        images = [Image.open(BytesIO(d), formats=["png"]) for d in data["images"]]
        outputs = app_geneval.INFERENCE_FN(
            images, data["meta_datas"], data["only_strict"]
        )
        keys = ("scores", "rewards", "strict_rewards")
        keys += ("group_rewards", "group_strict_rewards")
        return dict(zip(keys, outputs))

    pickle_route(app, "/png", inference_png)
    return app


def create_local_app():
    """Use the parent project's existing resources without downloading weights."""
    root = Path(os.environ["TWINTILT_RESOURCE_ROOT"])
    reward_dir = Path(os.environ.get(
        "GENEVAL_REWARD_SERVER_DIR",
        root / "third_party/advantage_weighted_matching/reward-server",
    ))
    config = Path(os.environ.get(
        "GENEVAL_MMDET_CONFIG",
        root / "third_party/mmdetection/configs/mask2former/"
        "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
    ))
    checkpoint = Path(os.environ.get(
        "GENEVAL_DETECTOR_CHECKPOINT",
        root / "models/geneval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_"
        "20220504_001756-743b7d99.pth",
    ))
    clip = Path(os.environ.get(
        "GENEVAL_CLIP_PATH", root / "models/geneval/ViT-L-14-openai.pt"
    ))
    object_names = reward_dir / "reward_server/object_names.txt"
    resources = (config, checkpoint, clip, object_names)
    for path in resources:
        if not path.is_file():
            raise FileNotFoundError(f"Missing local GenEval resource: {path}")
    # The external scoring code resolves object_names.txt relative to its cwd.
    os.chdir(reward_dir)
    return create_app(
        str(reward_dir), str(config), str(checkpoint.parent), str(clip),
        detector_checkpoint=str(checkpoint), clip_weights_only=False,
    )
