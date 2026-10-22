"""FID/IS metrics: Inception features vs. cached reference statistics."""

import numpy as np
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_features_to_metric


def build_feature_extractor(device, inception_weights_path=None):
    """Build the Inception-v3 extractor feeding both the FID and the IS features.

    It consumes uint8 NCHW images in [0, 255].
    inception_weights_path loads local Inception weights; None downloads them.
    """
    return FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048", "logits_unbiased"],
        feature_extractor_weights_path=inception_weights_path,
    ).to(device).eval()


def features_to_metrics(feats_2048, feats_logits, fid_stats_file):
    """FID (vs. precomputed reference stats) and IS of already extracted features.

    Feeds the extractor outputs into torch_fidelity's stats-level helpers,
    sidestepping the installed calculate_metrics() which lacks a
    fid_statistics_file argument.
    """
    is_dict = isc_features_to_metric(feats_logits)

    # Reference .npz is float64; cast generated stats to match.
    stats_gen = fid_features_to_statistics(feats_2048)
    ref = np.load(fid_stats_file)
    stats_gen = {"mu": stats_gen["mu"].astype(np.float64),
                 "sigma": stats_gen["sigma"].astype(np.float64)}
    stats_ref = {"mu": ref["mu"].astype(np.float64),
                 "sigma": ref["sigma"].astype(np.float64)}
    fid_dict = fid_statistics_to_metric(stats_gen, stats_ref, verbose=False)

    return {**is_dict, **fid_dict}
