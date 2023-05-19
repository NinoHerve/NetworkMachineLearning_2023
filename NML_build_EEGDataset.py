from utils.readers import read_epochs, read_events
from pathlib import Path
import os
import numpy as np


# -------------------------------------------------------
""" Choose the parameters for building your dataset here

Only run NML_build_EEGDataset.py when the parameters here
suit you.
"""

# path to Vepcon dataset
bids_dir = Path('/home/admin/work/data/ds003505-download/')

# path where to save EEGDataset
save_dir = Path(os.getcwd())

# subjects: 1-20 without 5
subjects = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# task: 'faces' or 'motion'
task = 'faces'                      
# -------------------------------------------------------


def load_data_Vepcon(bids_dir, subject, task):
    """Loads eeg data of specific subject and task
    
    Args:
        bids_dir (Path): Path to data folder
        subject (str): name of subject
        task (str): name of taks. Valid options are 'faces' or 'motion'

    Returns:
        X (numpy.ndarray): EEG data of shape (n_epochs, n_channels, n_samples)
        y (numpy.ndarray): condition of each trial. Values are 0 if condition1 and 1 if condition2
    """

    epochs = read_epochs(bids_dir, subject, task)
    events = read_events(bids_dir, subject, task)

    X = epochs.get_data()
    y = events[events['bad_epoch']==0]['trial_type']

    return X, y.to_numpy()



if __name__ == '__main__':

    print('Creating EEG dataset for NML course')

    # Select names for labels
    if task == 'faces':
        labels = ['faces', 'scrambled']
    elif task == 'motion':
        labels = ['label1', 'label2']
    else:
        raise ValueError(f'task should be "faces" or "motion", not {task}')

    # Create EEGDataset directory with label subdirectories
    eeg_dir = save_dir / 'EEGDataset'
    os.makedirs(eeg_dir)
    for l in labels: 
        os.makedirs(eeg_dir / l)

    # Iterate through subjects
    for sub in subjects:
        
        # Subject name
        subject = f'sub-{str(sub).zfill(2)}'
        print(f'\nLoading and saving data from {subject}...')

        # Load data
        X, y = load_data_Vepcon(bids_dir, subject, task)

        # Save each epoch as sample in respective label folder
        for i, l in enumerate(y):
            label_dir = eeg_dir / l.lower()
            np.save(label_dir / f'{subject}_{i}.npy', X[i])
        
