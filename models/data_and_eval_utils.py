# data_and_eval_utils.py
"""Holds utility functions to load the preprocessed data and functions
to evaluate NN models specific to our work"""

# GPT Generated Code START ---
import h5py
import numpy as np

# ---------------------------------------------------------------------
# 1. Data-loading helper
# ---------------------------------------------------------------------
def load_preprocessed_snl(file_path):
    """
    Returns a dict of numpy arrays read from the HDF5 file at `file_path`.
    """
    with h5py.File(file_path, "r") as hf:
        return {
            "X_train":         hf["X_train"][:],
            "X_test":          hf["X_test"][:],
            "Ato4l":           hf["Ato4l_data"][:],
            "hToTauTau":       hf["hToTauTau_data"][:],
            "hChToTauNu":      hf["hChToTauNu_data"][:],
            "leptoquark":      hf["leptoquark_data"][:],
        }

# ---------------------------------------------------------------------
# 2. High-level utility
# ---------------------------------------------------------------------
def get_truth_and_scores(encoder, ad_metric, data, debug=True):
    """
    Parameters
    ----------
    encoder : keras.Model          # your trained encoder
    ad_metric : callable           # anomaly-detection metric
    data : dict                    # output of load_preprocessed_snl(...)
    debug : bool

    Returns
    -------
    truths  : list[np.ndarray]
    scores  : list[np.ndarray]
    bad_model : bool
    """
    # unpack once for readability
    X_test = data["X_test"]

    bg_score, bad_model = calc_anomaly_scores(X_test, encoder, ad_metric)

    truths, scores = [], []
    zeros = np.zeros(len(X_test))
    signal_keys = ["leptoquark", "Ato4l", "hChToTauNu", "hToTauTau"]

    if not bad_model:
        for key in signal_keys:
            sig = data[key]
            truths.append(np.concatenate((zeros, np.ones(len(sig)))))
            sig_score, _ = calc_anomaly_scores(sig, encoder, ad_metric, debug=debug)
            scores.append(np.concatenate((bg_score, sig_score)))

    return truths, scores, bad_model

# GPT Genereted Code END
