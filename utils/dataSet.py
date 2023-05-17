import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.loaders import load_data_multiple_subjects


class EEGDataset(Dataset):
    '''EEG dataset, stores whole data on RAM (max 7GB)'''

    def __init__(self, eeg_dir, subjects):
        data = load_data_multiple_subjects(eeg_dir, subjects)
        
        # Concatenate data from all subjects
        self.eeg_data = []
        self.labels   = []

        for sub in subjects:
            self.eeg_data.append(data[sub]['data'])
            self.labels.append(data[sub]['target'])
           
        self.eeg_data = np.concatenate(self.eeg_data, axis=0)
        self.labels   = np.concatenate(self.labels, axis=0) 

    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, index):
        eeg_sample = self.eeg_data[index]
        label = self.labels[index]
        return eeg_sample, label
    

def collate_fn(batch):
    '''The collate function is responsible for combining multiple
    samples into a batch.'''

    eeg_batch, label_batch = zip(*batch)
    eeg_batch = pad_sequence(eeg_batch, batch_first=True)

    return eeg_batch, torch.tensor(label_batch)