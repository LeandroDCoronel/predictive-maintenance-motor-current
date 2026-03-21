import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(signal_matrix):
    """
    signal_matrix: numpy array shape (N, 3) → [Ia, Ib, Ic]
    """

    Ia = signal_matrix[:, 0]
    Ib = signal_matrix[:, 1]
    Ic = signal_matrix[:, 2]

    features = []

    for signal in [Ia, Ib, Ic]:

        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.max(signal))
        features.append(np.min(signal))
        features.append(np.sqrt(np.mean(signal**2)))  # RMS
        features.append(skew(signal))
        features.append(kurtosis(signal))
        features.append(np.sum(signal**2))  # energy

    return np.array(features)