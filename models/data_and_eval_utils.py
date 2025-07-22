# data_and_eval_utils.py
"""Holds utility functions to load the preprocessed data and functions
to evaluate NN models specific to our work"""


import h5py
import numpy as np
import tensorflow.keras as keras

def load_preprocessed_snl(file_path=None):
    """
    Returns a dict of numpy arrays read from the HDF5 file at `file_path`.

    Parameters
    ----------
    file_path: str | If None, file_path defaults to "/global/cfs/cdirs/m2616/jananinf/projsIO/VAE_FS/preprocessed_SNL_data.h5"
    """
    # ---------------------------------------------------------------------
    # 1. Data-loading helper
    # ---------------------------------------------------------------------
    if file_path == None:
        file_path =  "/global/cfs/cdirs/m2616/jananinf/projsIO/VAE_FS/preprocessed_SNL_data.h5"
    with h5py.File(file_path, "r") as hf:
        X_train = hf['X_train'][:]                  # (3200000, 57)
        X_test  = hf['X_test'][:]                   # (800000,  57)
        Ato4l_data  = hf['Ato4l_data'][:]           # (55969,   57) Signal data? 
        hToTauTau_data  = hf['hToTauTau_data'][:]   # (691283,  57)
        hChToTauNu_data  = hf['hChToTauNu_data'][:] # (760272,  57)
        leptoquark_data = hf['leptoquark_data'][:]  # (340544,  57)
        print("Data loaded from preprocessed_SNL_data.h5")
        return {
            "X_train":      X_train,
            "X_test":       X_test,
            "Ato4l":        Ato4l_data,      
            "hToTauTau":    hToTauTau_data,  
            "hChToTauNu":   hChToTauNu_data, 
            "leptoquark":   leptoquark_data, 
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

def calc_anomaly_scores(data, encoder: keras.Model, AD_metric, debug = True):
    """
    Parameters:
    -----------
    debug: Optional bool to skip latent space vectors that produce infinities.
    Currently set to true as it seems only 2 specific cases are affected
    """
    dat_encoded = np.array(encoder.predict(data))[0] # This outputs shape (3, len(X_test), 3). Can't find satisfactory explanation for this behavior. (len(X_test), 3) makes sense. (3, len, 3) does not
    # Kenny only uses the first list so we'll follow that convention.
    # has shape (len(data), 3), where col 1 is z_mean, 2 is z_log_var and z. This is by design of encoder.
    scores = np.zeros(len(data))
    bad_model = False
    for i in range(len(scores)):
        z_mean, z_log_var = dat_encoded[i][0], dat_encoded[i][1]
        score = AD_metric(z_mean, z_log_var)
        if debug and (score == np.inf):
            print("Unstable model: inf encountered. Rejecting Model"
                  + f"z_mean: {z_mean}\n"
                  + f"z_log_var: {z_log_var}")
            
            bad_model = True
            break
        scores[i] = score

    return (scores, bad_model)

def AD_score_KL(z_mean, z_log_var):
    kl_loss = np.mean(-0.5 * (1 + z_log_var - (z_mean) ** 2 - np.exp(z_log_var)))
    # Comparing this to eq 2 in arXiv: 2108.03986 z_log_var = log(sigma**2)
    return kl_loss


def AD_score_CKL(z_mean, _): # z_log_var not used
    CKL = np.mean(z_mean**2)
    return CKL 

def AD_score_Rz(z_mean, z_log_var):
    return z_mean**2/np.exp(z_log_var)

