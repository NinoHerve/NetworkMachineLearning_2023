import os
import mne
import pandas as pd
import numpy as np

def read_subject_epochs(bids_dir, subject, tmin=-0.2, tmax=0.5, apply_proj=True):

    eeglab_dir = os.path.join(bids_dir, "derivatives", "eeglab-v14.1.1", subject, "eeg")
    epochs_fname = os.path.join(
        eeglab_dir, f"{subject}_task-faces_desc-preproc_eeg.set"
    )

    eeg = os.path.join(bids_dir, subject, "eeg")
    beh_fname = os.path.join(
        eeg, f"{subject}_task-faces_events.tsv"
    )
    beh = pd.read_csv(beh_fname, sep="\t")
    beh = beh[beh.bad_epoch == 0]

    epochs = mne.read_epochs_eeglab(
            epochs_fname,
            events=None,
            event_id=None,
            eog=(),
            verbose=0,
            uint16_codec=None,
        )

    epochs.events[:, 2] = list(beh.trial_type_id)
    epochs.event_id = {"Scrambled": 0, "Faces": 1}

    epochs.apply_baseline((-0.2, 0))
    epochs.set_eeg_reference(ref_channels="average", projection=True)

    if tmin is not None:
        epochs.crop(tmin=tmin, tmax=tmax)

    if apply_proj :
        epochs.apply_proj()

    return epochs, beh.trial_type_id.values

def multiple_subjects_epochs(bids_dir, subjects, tmin=-0.2, tmax=0.5, apply_proj=True):
    epochs_multi = []
    trial_type_multi = []

    for sub in subjects:
        epochs, trial_types = read_subject_epochs(bids_dir, sub, tmin, tmax, apply_proj)
        epochs_multi.append(epochs)
        trial_type_multi.append(trial_types)

    return epochs_multi, trial_type_multi

def aggregate_epochs(epochs_multi, y_multi, subjects):
    assert len(epochs_multi) == len(subjects)
    agg_epochs, agg_y, agg_subjects = [],[],[]
    for epochs, y, sub in zip(epochs_multi,y_multi, subjects):
        trials = epochs.get_data()
        n_trials = len(trials)
        agg_epochs.append(trials)
        agg_y.append(y)
        agg_subjects.extend([sub]*n_trials)
    
    agg_epochs = np.concatenate(agg_epochs, axis=0)
    agg_y = np.concatenate(agg_y, axis=0)
    agg_subjects = np.array(agg_subjects, dtype=object)
    return agg_epochs, agg_y, agg_subjects

def reduce_trials(X, y, S, method='sample', n_samples = 50):
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
    # Get unique trial types and subjects
    unique_trial_types = np.unique(y)
    unique_subjects = np.unique(S)

    if method == 'evoked':
        # Create empty arrays to store the averaged data
        averaged_data = np.zeros((len(unique_trial_types)* len(unique_subjects), X.shape[1], X.shape[2]))

        # Iterate over trial types and subjects
        for i, subject in enumerate(unique_subjects):
            for j, trial_type in enumerate(unique_trial_types):
                # Find the indices where trial type and subject match
                indices = np.logical_and(S == subject, y == trial_type)
            
                # Average the data along the 0th axis (n_trials)
                averaged_data[i, j] = np.mean(X[indices], axis=0)

        return averaged_data

    if method == 'sample':
        picked_trials = []
        for sub in unique_subjects:
            for trial_type in unique_trial_types:
                trial_indices = np.where(np.logical_and(S==sub,y==trial_type))[0]
                picked_trials.append(np.sort(np.random.choice(trial_indices, n_samples, replace=False)))
        picked_trials = np.concatenate(picked_trials, axis=0)
        print('Picked {} trials out of {}'.format(len(picked_trials), len(X)))
        return X[picked_trials], y[picked_trials], S[picked_trials]