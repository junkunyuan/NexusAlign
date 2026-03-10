from nexus_align.core.registry import registry
from nexus_align.reward_models.reward_model_hps_v2 import HPSv2
from nexus_align.reward_models.reward_model_hps_v3 import HPSv3
from nexus_align.reward_models.reward_model_clip_score import CLIPScore
from nexus_align.reward_models.reward_model_pick_score import PickScore
from nexus_align.reward_models.reward_model_qwen3_vl_8b import Qwen3_VL_8B
from nexus_align.reward_models.reward_model_image_reward import ImageReward
from nexus_align.reward_models.reward_model_intern3_5_vl_8b import Intern3_5_VL_8B
from nexus_align.reward_models.reward_model_aesthetic_predictor_v2 import AestheticPredictorV2
from nexus_align.reward_models.reward_model_aesthetic_predictor_v1 import AestheticPredictorV1

for _name, _cls in [
    ("hps_v2", HPSv2),
    ("hps_v3", HPSv3),
    ("clip_score", CLIPScore),
    ("pick_score", PickScore),
    ("qwen3_vl_8b", Qwen3_VL_8B),
    ("image_reward", ImageReward),
    ("intern3_5_vl_8b", Intern3_5_VL_8B),
    ("aesthetic_predictor_v2", AestheticPredictorV2),
    ("aesthetic_predictor_v1", AestheticPredictorV1),
]:
    registry.register("reward_model", _name, _cls)
