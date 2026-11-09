"""FID/IS metrics: compare Inception features with cached reference statistics."""

import numpy as np


def build_feature_extractor(device, inception_weights_path=None):
    """
    Build the Inception-v3 extractor that gives both the FID and the IS features.

    It takes uint8 NCHW images in [0, 255]. It loads the local Inception weights
    from inception_weights_path, or downloads them if the path is None.
    """
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    return FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048", "logits_unbiased"],
        feature_extractor_weights_path=inception_weights_path,
    ).to(device).eval()


def features_to_metrics(feats_2048, feats_logits, fid_stats_file):
    """
    Calculate the FID and IS of the extracted features.

    The FID compares the features with precomputed reference statistics. This
    function gives the extractor outputs to the statistics-level helpers of
    torch_fidelity. It does not use the installed calculate_metrics(), because
    that function has no fid_statistics_file argument.
    """
    from torch_fidelity.metric_fid import (
        fid_features_to_statistics, fid_statistics_to_metric,
    )
    from torch_fidelity.metric_isc import isc_features_to_metric

    is_dict = isc_features_to_metric(feats_logits)

    # The reference .npz is float64, so cast the generated statistics to match.
    stats_gen = fid_features_to_statistics(feats_2048)
    ref = np.load(fid_stats_file)
    stats_gen = {"mu": stats_gen["mu"].astype(np.float64),
                 "sigma": stats_gen["sigma"].astype(np.float64)}
    # JAX statistics name their keys ref_mu and ref_sigma.
    prefix = "ref_" if "ref_mu" in ref.files else ""
    stats_ref = {"mu": ref[f"{prefix}mu"].astype(np.float64),
                 "sigma": ref[f"{prefix}sigma"].astype(np.float64)}
    fid_dict = fid_statistics_to_metric(stats_gen, stats_ref, verbose=False)

    return {**is_dict, **fid_dict}
