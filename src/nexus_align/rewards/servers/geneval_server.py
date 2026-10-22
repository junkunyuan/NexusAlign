"""GenEval reward server: reward-server's GenEval app with local weights, under gunicorn.

Runs in .venv-geneval (mmdet 2.x needs torch 2.1 and mmcv-full); launch with
scripts/tools/reward_server.sh. The scoring code is reward-server's own, unmodified;
only its weight paths are set here, and a /png route scores lossless images.

References:
    - Reward server code: https://github.com/yifan123/reward-server
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT (flow_grpo/gen_eval.py)
"""

import sys
import pickle
import traceback
from io import BytesIO


def create_app(reward_server_dir: str, mmdet_config: str, ckpt_dir: str, clip_path: str):
    """Build reward-server's app; run with reward_server_dir as the working dir."""
    sys.path.insert(0, reward_server_dir)

    # pretrained="openai" in reward-server resolves to this local file, not the hub.
    import open_clip
    open_clip.pretrained.get_pretrained_cfg("ViT-L-14", "openai")["file"] = clip_path

    from reward_server import gen_eval
    gen_eval.MY_CONFIG_PATH = mmdet_config
    gen_eval.MY_CKPT_PATH = ckpt_dir

    import app_geneval
    from PIL import Image
    from flask import request
    app = app_geneval.create_app()

    @app.route("/png", methods=["POST"])
    def inference_png():
        """Score the exact pixels, as DiffusionNFT's in-process GenEval does."""
        try:
            data = pickle.loads(request.get_data())
            images = [Image.open(BytesIO(d), formats=["png"]) for d in data["images"]]
            outputs = app_geneval.INFERENCE_FN(
                images, data["meta_datas"], data["only_strict"]
            )
            keys = ("scores", "rewards", "strict_rewards")
            keys += ("group_rewards", "group_strict_rewards")
            return pickle.dumps(dict(zip(keys, outputs))), 200
        except Exception:
            response = traceback.format_exc()
            print(response)
            return response.encode("utf-8"), 500

    return app
