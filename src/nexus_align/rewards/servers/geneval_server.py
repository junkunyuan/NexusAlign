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
from io import BytesIO

from common import pickle_route


def create_app(reward_server_dir: str, mmdet_config: str, ckpt_dir: str, clip_path: str):
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
    app = app_geneval.create_app()

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
