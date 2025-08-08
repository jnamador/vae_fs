# data_and_eval_utils.py
"""Holds utility functions to load the preprocessed data and functions
to evaluate NN models specific to our work"""


import h5py
import numpy as np
import tensorflow.keras as keras
from matplotlib import pyplot as plt
import os
import sklearn.metrics as sk
from sklearn.metrics import roc_curve
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # on NERSC filelocking is not allowed

SIG_KEYS = { # Only contains the keys to signal data. Background and training data is not included.
    'Ato4l':      {'human': "A to 4L",
                   'latex': "$A\\rightarrow 4\ell$"},
    'hToTauTau':  {'human': "h to Tau Tau",
                   'latex': "$h^0\\rightarrow\\tau\\tau$"},
    'hChToTauNu': {'human': "h to Tau Nu",
                   'latex': "$h^{\pm}\\rightarrow\\tau \\nu$"},
    'leptoquark': {'human': "Leptoquark",
                   'latex': "Leptoquark"},
    }

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
    truths  : dict{SIG_KEY: np.array}
    scores  : dict{SIG_KEY: np.array}
    bad_model : bool
    """
    # unpack once for readability
    X_test = data["X_test"]

    bg_score, bad_model = calc_anomaly_scores(X_test, encoder, ad_metric)

    truths, scores = [], []
    zeros = np.zeros(len(X_test))

    truths = {}
    scores = {}

    if not bad_model:
        for key in SIG_KEYS.keys():
            sig = data[key]
            truths[key] = np.concatenate((zeros, np.ones(len(sig))))
            sig_score, bad_model = calc_anomaly_scores(sig, encoder, ad_metric, debug=debug)
            if bad_model: break
            scores[key] = np.concatenate((bg_score, sig_score))

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
        if debug and (np.isinf(score) or np.isnan(score)):
            print("Unstable model: inf or nan encountered. Rejecting Model"
                  + f"z_mean: {z_mean}\n"
                  + f"z_log_var: {z_log_var}")
            
            bad_model = True
            break
        else:
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

def get_roc_performance(truth, score, target_fpr):
    """Calculates the AUC for the ROC as well as getting the True Positive Rate 
    at the Target False Positive Rate"""
    fpr, tpr, thresholds = roc_curve(truth, score)
    auc = sk.roc_auc_score(truth, score)
    idx = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target = tpr[idx]
    threshold_at_target = thresholds[idx]
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc' : auc,
        'tpr_at_target': tpr_at_target,
        'threshold_at_target': threshold_at_target,

    }

def eval_rocs(encoder, data, AD_metric, target_fpr = 1e-5):
    """Evaluates the ROC Curves for the encoder
    Returns: dict | None. dict is structured as {sig_name: {get_roc_perfomance output}}"""
    truths, scores, bad_model = get_truth_and_scores(encoder, AD_metric, data)
    roc_perfs = {}
    if bad_model:
        return None
    else:
        for k in SIG_KEYS.keys():
            roc_perfs[k] = get_roc_performance(truths[k], scores[k], target_fpr)
        return roc_perfs


def plot_rocs(roc_perfs, fig_title):
    fig, ax = plt.subplots()

    if roc_perfs is None:
        print("Unstable Model")
    else:
        for k in SIG_KEYS.keys():
            metrics = roc_perfs[k]
            label = SIG_KEYS[k]['human']
            ax.plot(metrics['fpr'], metrics['tpr'], label=label + f": {str(round(metrics['auc'], 3))}") # plot roc curve

        ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--")
        ax.vlines(10**-5, 0, 1, color="r" , linestyles="dashed")

        # Plot teaks
        ax.loglog()
        ax.legend()
        ax.grid()
        ax.set_xlabel("fpr")
        ax.set_ylabel("tpr")
        ax.set_title(fig_title) 
        plt.show()

        # Sanity check
        temp = [roc_perfs[k]['threshold_at_target'] for k in SIG_KEYS.keys()]
        test = [temp[0] == t for t in temp]
        if all(test):
            print("We good. All Tresholds match")
            print(f"Treshold at Target: {temp[0]}")
        else:
            print("We're cooked. No clue how this happened")

        for k in SIG_KEYS.keys():
            sig_name_hum = SIG_KEYS[k]['human']
            tpr_at_target = roc_perfs[k]['tpr_at_target']
            
            print(sig_name_hum + " TPR @ FPR 10e-5 (%): " + f"{tpr_at_target*100:.2f}\n")

        return fig

def calc_anomaly_dist(data, encoder: keras.Model, AD_metric):
    """
    Parameters:
    -----------
    data: dict | output from load_preproccessed_snl()
    AD_metric: func(z_mean, z_log_var) | Metric used for anomaly detection, inputs should be scalars
    """
    scores = []
    for _, dat in data.items():
        dat_encoded = np.array(encoder.predict(data))[0] # This outputs shape (3, len(X_test), 3). Can't find satisfactory explanation for this behavior. (len(X_test), 3) makes sense. (3, len, 3) does not
        # Kenny only uses the first list so we'll follow that convention.
        # has shape (len(data), 3), where col 1 is z_mean, 2 is z_log_var and z. This is by design of encoder.
        temp_score = np.zeros(len(data))
        for i in range(len(temp_score)):
            z_mean, z_log_var = dat_encoded[i][0], dat_encoded[i][1]
            temp_score[i] = AD_metric(z_mean, z_log_var)
        scores.append(temp_score)
    return scores