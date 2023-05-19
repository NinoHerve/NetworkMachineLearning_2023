import torch
from torch.utils.data import Dataset
import os
import numpy as np


class EEGDataset(Dataset):
    """EEG dataset"""

    def __init__(self, eeg_dir, subjects, transform=None):
        
        self.files = self.files_of_subjects(eeg_dir, subjects)     # list of files for the dataset
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load data of specified file name
        fn = self.files[idx]
        x = np.load(fn)
        y = int(fn.parts[-2] == 'faces')    # 1 = 'faces', 0 = 'scrambled'

        sample = {'eeg':x, 'label':y}

        # Transform data
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def files_of_subjects(self, eeg_dir, subjects):
        """Retrieve the file names of specified subjects
        
        Args:
            eeg_dir (Path): path to eeg data
            subjects (list): list of subject name
            
        Returns:
            (list): list of file names containing specified subject samples
        """

        labels = os.listdir(eeg_dir)    # first two folders seperate data into labels
        files  = []                                                  

        for l in labels:
            file_names = os.listdir(eeg_dir / l)
            for fn in file_names:
                if fn[:6] in subjects:              # fn[:6] corresponds to subject name
                    files.append(eeg_dir / l / fn)

        return files
