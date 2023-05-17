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

    # Create EEGDataset directory
    eeg_dir = save_dir / 'EEGDataset'
    os.makedirs(eeg_dir)

    for sub in subjects:
        
        # Subject name
        subject = f'sub-{str(sub).zfill(2)}'
        print(f'Loading and saving data from {subject}...')

        # Create subject directory
        subject_dir = eeg_dir / subject 
        os.makedirs(subject_dir)

        # Load data then save data in subject directory
        X, y = load_data_Vepcon(bids_dir, subject, task)

        np.save(subject_dir / f'{subject}_data.npy', X)
        np.save(subject_dir / f'{subject}_target.npy', y)
