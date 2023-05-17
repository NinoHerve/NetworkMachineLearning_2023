import numpy as np


def load_data_subject(eeg_dir, subject):
    """Loads eeg data for one subject
    
    Args:
        eeg_dir (Path): path to eeg dataset
        subject (str): name of the subject (e.g. 'sub-01')
        
    Returns:
        X (numpy.ndarray): array of shape (n_epochs, n_channels, n_time_points)
        y (numpy.ndarray): array of shape (n_epochs,) containing either 'faces' 
        or 'scrambled'
    """

    X = np.load(eeg_dir / subject / f'{subject}_data.npy', allow_pickle=True)
    y = np.load(eeg_dir / subject / f'{subject}_target.npy', allow_pickle=True)

    return X, y 


def load_data_multiple_subjects(eeg_dir, subjects):
    """Loads eeg data for multiple subjects
    
    Args:
        eeg_dir (Path): path to eeg dataset
        subject (list, str): name of the subjects (e.g. 'sub-01')
        
    Returns:
        (dictionary): first key selects the subject (e.g. ['sub-01']), 
        and second key selects either the data ['data'] or the targets 
        ['target']
    """
    # Change to list for iteration if string was given
    if isinstance(subjects, str):
        subjects = [subjects]

    data = dict()

    for sub in subjects:

        X, y = load_data_subject(eeg_dir, sub)

        data[sub] = dict()
        data[sub]['data']   = X
        data[sub]['target'] = y

    return data


