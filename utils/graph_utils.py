import numpy as np
import torch
from scipy.signal import hilbert


def pearson_corr_coef(x,y, abs=True):
    corr_coef = np.corrcoef(x,y)[0][1]
    if abs:
        return np.abs(corr_coef)
    else :
        return corr_coef


def plv(x, y, abs=True):
    x_hill = hilbert(x)
    y_hill = hilbert(y)
    pdt = (np.inner(x_hill, np.conj(y_hill)) /
            (np.sqrt(np.inner(x_hill, np.conj(x_hill)) * np.inner(y_hill, np.conj(y_hill)))))
    if abs:
        return np.abs(np.angle(pdt))
    else:
        return np.angle(pdt)



def threshold_graph(edge_index, edge_weights, density):

    sorted_idx = np.argsort(edge_weights)
    stop = int(density * len(edge_weights))

    mask = sorted_idx[:stop]
    edge_index_threshold = edge_index[:,mask]
    edge_weights_threshold = edge_weights[mask]
    
    return edge_index_threshold, edge_weights_threshold


def linearize(x, batch):
    features = []
    for sample in torch.unique(batch):
        rows = (batch == sample)
        features.append(x[rows].flatten())

    return torch.stack(features)