import numpy as np
from scipy.signal import hilbert


def pearson_corr_coef(x,y, abs=True):
    corr_coef = np.corrcoef(x,y)[0][1]
    if abs:
        return np.abs(corr_coef)
    else :
        return corr_coef


def plv(x, y):
    x_hill = hilbert(x)
    y_hill = hilbert(y)
    pdt = (np.inner(x_hill, np.conj(y_hill)) /
            (np.sqrt(np.inner(x_hill, np.conj(x_hill)) * np.inner(y_hill, np.conj(y_hill)))))
    return np.angle(pdt)