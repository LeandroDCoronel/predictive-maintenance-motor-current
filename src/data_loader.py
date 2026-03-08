from scipy.io import loadmat
import numpy as np


def load_motor_current_data(path):
    """
    Loads motor current dataset from MATLAB .mat file.

    Returns
    -------
    np.ndarray
        Array with shape (samples, 3) containing Ia, Ib, Ic
    """

    mat = loadmat(path)
    itsc = mat["itsc"]

    signals = []

    # iterate conditions
    for condition in itsc[0]:

        # iterate repetitions
        for repetition in condition.dtype.names:

            signal = condition[repetition][0][0]

            signals.append(signal)

    data = np.vstack(signals)

    return data