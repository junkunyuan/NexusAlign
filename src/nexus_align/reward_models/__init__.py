from nexus_align.core.registry import registry
from nexus_align.reward_models.reward_model_hps_v2 import HPSv2
from nexus_align.reward_models.reward_model_hps_v3 import HPSv3
from nexus_align.reward_models.reward_model_clip_score import CLIPScore
from nexus_align.reward_models.reward_model_pick_score import PickScore
from nexus_align.reward_models.reward_model_qwen3_5_27b import Qwen3_5_27B
from nexus_align.reward_models.reward_model_qwen3_5_9b import Qwen3_5_9B
from nexus_align.reward_models.reward_model_ocr_metrics import OCRMetrics
from nexus_align.reward_models.reward_model_qwen3_vl_8b import Qwen3_VL_8B
from nexus_align.reward_models.reward_model_qwen3_vl_32b import Qwen3_VL_32B
from nexus_align.reward_models.reward_model_image_reward import ImageReward
from nexus_align.reward_models.reward_model_glm_4_6v_flash import GLM_4_6V_Flash
from nexus_align.reward_models.reward_model_intern3_5_vl_8b import Intern3_5_VL_8B
from nexus_align.reward_models.reward_model_aesthetic_predictor_v2 import AestheticPredictorV2
from nexus_align.reward_models.reward_model_aesthetic_predictor_v1 import AestheticPredictorV1
from nexus_align.reward_models.reward_model_normalized_edit_distance import NormalizedEditDistance

for _name, _cls in [
    ("hps_v2", HPSv2),
    ("hps_v3", HPSv3),
    ("clip_score", CLIPScore),
    ("pick_score", PickScore),
    ("qwen3_5_27b", Qwen3_5_27B),
    ("qwen3_5_9b", Qwen3_5_9B),
    ("ocr_metrics", OCRMetrics),
    ("qwen3_vl_8b", Qwen3_VL_8B),
    ("qwen3_vl_32b", Qwen3_VL_32B),
    ("image_reward", ImageReward),
    ("glm_4_6v_flash", GLM_4_6V_Flash),
    ("intern3_5_vl_8b", Intern3_5_VL_8B),
    ("aesthetic_predictor_v2", AestheticPredictorV2),
    ("aesthetic_predictor_v1", AestheticPredictorV1),
    ("normalized_edit_distance", NormalizedEditDistance),
]:
    registry.register("reward_model", _name, _cls)
