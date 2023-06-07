import os
import mne
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys

sys.path.append("../")
from utils.graph_utils import pearson_corr_coef
from multiprocessing import Pool
from functools import partial

from geopy.distance import geodesic
from itertools import combinations
import math

def read_subject_epochs(
    bids_dir, subject, tmin=-0.1, tmax=0.5, apply_proj=False, apply_baseline=True
):
    eeglab_dir = os.path.join(bids_dir, "derivatives", "eeglab-v14.1.1", subject, "eeg")
    epochs_fname = os.path.join(
        eeglab_dir, f"{subject}_task-faces_desc-preproc_eeg.set"
    )

    eeg = os.path.join(bids_dir, subject, "eeg")
    beh_fname = os.path.join(eeg, f"{subject}_task-faces_events.tsv")
    beh = pd.read_csv(beh_fname, sep="\t")
    beh = beh[beh.bad_epoch == 0]

    epochs = mne.read_epochs_eeglab(
        epochs_fname,
        events=None,
        event_id=None,
        eog=(),
        verbose='critical',
        uint16_codec=None,
    )

    epochs.events[:, 2] = list(beh.trial_type_id)
    epochs.event_id = {"Scrambled": 0, "Faces": 1}

    if apply_baseline:
        epochs.apply_baseline((-0.2, 0))

    epochs.set_eeg_reference(ref_channels="average", projection=True, verbose='critical')

    if tmin is not None:
        epochs.crop(tmin=tmin, tmax=tmax)

    if apply_proj:
        epochs.apply_proj()

    return epochs, beh.trial_type_id.values


def multiple_subjects_epochs(
    bids_dir, subjects, tmin=-0.2, tmax=0.5, apply_proj=False, apply_baseline=False
):
    epochs_multi = []
    trial_type_multi = []

    for sub in subjects:
        epochs, trial_types = read_subject_epochs(
            bids_dir, sub, tmin, tmax, apply_proj, apply_baseline
        )
        epochs_multi.append(epochs)
        trial_type_multi.append(trial_types)

    return epochs_multi, trial_type_multi


def aggregate_epochs(epochs_multi, y_multi, subjects):
    assert len(epochs_multi) == len(subjects)
    agg_epochs, agg_y, agg_subjects = [], [], []
    for epochs, y, sub in zip(epochs_multi, y_multi, subjects):
        trials = epochs.get_data()
        n_trials = len(trials)
        agg_epochs.append(trials)
        agg_y.append(y)
        agg_subjects.extend([sub] * n_trials)

    agg_epochs = np.concatenate(agg_epochs, axis=0)
    agg_y = np.concatenate(agg_y, axis=0)
    agg_subjects = np.array(agg_subjects, dtype=object)
    return agg_epochs, agg_y, agg_subjects


def reduce_trials(X, y, S, method="sample", n_samples=50, seed=0):
    """Pool trials according to method

    Parameters
    ----------
    X
        trials of shape : (n_trials, electrodes, time)
    y
        type of each trial (n_trials,)
    S
        subject of each trial (n_trials,)
    method, optional
        one of ['evoked', 'subject', 'sample'], by default 'evoked'
    """
    # set seed for reproducibility :
    np.random.seed(seed)
    # Get unique trial types and subjects
    unique_trial_types = np.unique(y)
    unique_subjects = np.unique(S)

    if method == "evoked":
        # Create empty arrays to store the averaged data
        averaged_data = np.zeros(
            (len(unique_trial_types) * len(unique_subjects), X.shape[1], X.shape[2])
        )

        # Iterate over trial types and subjects
        for i, subject in enumerate(unique_subjects):
            for j, trial_type in enumerate(unique_trial_types):
                # Find the indices where trial type and subject match
                indices = np.logical_and(S == subject, y == trial_type)

                # Average the data along the 0th axis (n_trials)
                averaged_data[i, j] = np.mean(X[indices], axis=0)

        return averaged_data

    if method == "sample":
        picked_trials = []
        for sub in unique_subjects:
            for trial_type in unique_trial_types:
                trial_indices = np.where(np.logical_and(S == sub, y == trial_type))[0]
                picked_trials.append(
                    np.random.choice(trial_indices, n_samples, replace=False)
                )
        picked_trials = np.sort(np.concatenate(picked_trials, axis=0))
        print("Picked {} trials out of {}".format(len(picked_trials), len(X)))
        return X[picked_trials], y[picked_trials], S[picked_trials]


def compute_graphs_multi(X, graph_fct, threshold):
    adjs = []

    with Pool() as pool:
        compute_adj_partial = partial(graph_fct, threshold=threshold)
        adjs = pool.map(compute_adj_partial, X)

    return np.stack(adjs, axis=0)


def corr_coef_graph(eeg, threshold=False, binarize=False):
    num_nodes = len(eeg)
    adj = np.zeros((num_nodes, num_nodes))

    # need to compute adj matrix to be able to threshold or binarize
    for i in range(0, num_nodes):
        for j in range(i + 1, num_nodes):
            adj[i][j] = pearson_corr_coef(eeg[i], eeg[j])

    adj += adj.T
    if threshold:
        threshold = calculate_threshold(adj, threshold)
        adj[adj < threshold] = 0

    if binarize:
        adj[adj != 0] = 1.0

    return adj

def threshold_graphs(adjs, density):
    thresholded_adjs = adjs.copy()
    for adj in thresholded_adjs:
        threshold = calculate_threshold(adj, density)
        adj[adj < threshold] = 0

    return thresholded_adjs

def calculate_threshold(adjacency_matrix, density):
    """computes threshold to get desired density

    Parameters
    ----------
    adjacency_matrix
        _description_
    density
        _description_

    Returns
    -------
        threshold
    """
    flattened_matrix = np.sort(adjacency_matrix.flatten())[::-1]
    threshold_index = int(len(flattened_matrix) * density)
    threshold = flattened_matrix[threshold_index]
    return threshold

def electrode_distances(file_path = '../EEGDataset/electrode_coordinates.csv'):
    elec_coord = pd.read_csv(file_path, usecols=['x','y','z'])

    positions = elec_coord.to_numpy()
    distances = squareform(pdist(positions))
    return distances

def electrode_x_distances(file_path = '../EEGDataset/electrode_coordinates.csv'):
    elec_coord = pd.read_csv(file_path, usecols=['x'])

    positions = elec_coord.to_numpy()
    distances = squareform(pdist(positions))
    return distances

def electrode_distances_geodesic(file_path = '../EEGDataset/electrode_geographic_coords.csv'):
    elec_coord = pd.read_csv(file_path)
    positions = elec_coord.to_numpy()
    n_points = len(positions)
    distances = np.zeros((n_points, n_points))

    # Compute geodesic distances between all pairs of points
    for i, j in combinations(range(n_points), 2):
        point1 = positions[i][::-1]
        point2 = positions[j][::-1]
        distance = geodesic(point1, point2).meters
        distances[i][j] = distance
        distances[j][i] = distance

    return distances/np.max(distances)

def get_node_ordering(with_hemi=False, fname='../EEGDataset/electrode_coordinates.csv'):
    df = pd.read_csv(fname)
    hemi_map ={-1 : 1, 1: 2, 0: 0}
    df['hemi'] = df['x'].apply(lambda x : hemi_map.get(np.sign(x)))
    df.sort_values(['hemi','y','x'], ascending=[True, False, True], inplace=True)
    if with_hemi:
        return df.index.to_numpy(), df['hemi'].to_numpy()

    return df.index.to_numpy()
