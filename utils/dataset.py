# make a method in data that returns a Standardizer
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from utils.transforms import StandardScaler

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
    

    def standard_scaler(self):
        """Builds a transform to standardize data to *this dataset"""

        mean = self.mean()
        var  = self.var()

        scaler = StandardScaler(mean, var)

        return scaler
    

    def mean(self):
        """Compute the average of the dataset"""

        # initialization (end size should be (n_channels,))
        fn   = self.files[0]
        eeg  = np.load(fn)              # (n_channel, n_time_points)
        sum_ = np.sum(eeg, axis=1)      # (n_channel,)

        for fn in self.files[1:]:       # skip first element
            eeg = np.load(fn)
            sum_ += np.sum(eeg, axis=1)

        N = self.__len__() * eeg.shape[1]   # n_samples * n_time_points
        mean = sum_ / N                     # normalize

        return mean
    

    def var(self, mean=None):
        """Compute the standard deviation of the dataset"""

        if mean is None:
            mean = self.mean()

        sum_ = np.zeros_like(mean)

        for fn in self.files:
            eeg = np.load(fn)                           # (n_channel, n_time_points)
            eeg = eeg - np.expand_dims(mean, axis=1)    # centralize
            sum_ += np.sum(eeg**2, axis=1)              # (n_channel,)

        N = self.__len__() * eeg.shape[1]   # n_samples * n_time_points
        var = sum_ / (N-1)                  # normalize

        return var

