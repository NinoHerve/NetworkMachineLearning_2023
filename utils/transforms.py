import torch
import numpy as np
from scipy import stats



class Compose:

    def __init__(self, transforms):
        self.transforms = transforms 

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample 
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"     {t}"
        format_sting += "\n)" 
        return format_string 

class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        eeg, label = sample['eeg'], sample['label']

        eeg   = torch.from_numpy(eeg).float()
        label = torch.tensor(label, dtype=torch.float)

        return {'eeg': eeg,
                'label': label}
    

class Flatten(object):
    """Flatten EEG to 1D array"""

    def __call__(self, sample):
        eeg, label = sample['eeg'], sample['label']

        eeg = eeg.flatten()

        return {'eeg': eeg,
                'label': label}
    

    
class TemporalShift(object):
    """Randomly shift eeg signal"""

    def __init__(self, max_shift):
        self.max_shift = max_shift 

    def __call__(self, sample):
        """Apply temporal shifting to EEG data.
        eeg_data (numpy.ndarray): EEG data of shape (n_channels, n_samples).
        shift_amount (int): Number of samples to shift the EEG data.
        
        Args:
            sample (dict): EEG data of shape (n_channels, n_time_points) and labels

        Return:
            (dict): Shifted EEG data and labels
        """
        eeg, label = sample['eeg'], sample['label']

        shift_amount = np.random.randint(-self.max_shift, self.max_shift)
        shifted_eeg = np.roll(eeg, shift_amount, axis=1)

        return {'eeg': shifted_eeg,
                'label': label}
    
class AmplitudeScaling(object):
    """Randomly apply amplitude scaling to EEG data"""

    def __init__(self, max_amplitude):
        self.max_amplitude = max_amplitude 

    def __call__(self, sample):
        """Apply amplitude scaling to EEG data.
        
        Args:
            sample (dict): EEG data of shape (n_channels, n_time_points) and labels

        Returns:
            (dict): Rescaled EEG data and labels
        """
        eeg, label = sample['eeg'], sample['label']

        scale_factor = np.random.rand()*self.max_amplitude
        scaled_eeg = eeg * scale_factor

        return {'eeg': scaled_eeg,
                'label': label}
    

class Resize(object):
    """Resize temporal axis of EEG data by pruning the extremities"""

    def __init__(self, new_size: int):
        self.new_size = new_size

    def __call__(self, sample):
        """Resize temporal axis of EEG data by pruning the extremities

        Args:
            sample (dict): EEG data of shape (n_channels, n_time_points) and labels

        Returns:
            (dict): Resized EEG data and labels
        """
        eeg, label = sample['eeg'], sample['label']

        pruning = eeg.shape[1]-self.new_size

        if pruning <= 0: 
            raise ValueError(f'new size of resizing ({self.new_size}) is bigger than original size ({eeg.shape[1]})')
        
        pruning_left  = np.ceil(pruning/2).astype(np.int32)
        pruning_right = np.floor(pruning/2).astype(np.int32)

        resized_eeg = eeg[:, pruning_left:-pruning_right]

        return {'eeg': resized_eeg,
                'label': label}
    

class EEGStandardScaler(object):
    """Standardize samples"""

    def __init__(self, dataset=None):
        self.mean, self.std = self.compute_metrics(dataset)

    def compute_metrics(self, dataset):
        eeg = [sample['eeg'] for sample in dataset]
        eeg = torch.stack(eeg)

        mean = torch.mean(eeg, axis=0)
        std  = torch.std(eeg, axis=0)

        return mean, std

    def __call__(self, sample):
        """Standardize data

        Args:
            sample (dict): EEG data of shape (n_channels, n_time_points) and labels

        Returns:
            (dict): Rescaled EEG data and labels
        """
        eeg, label = sample['eeg'], sample['label']

        scaled_eeg = (eeg - self.mean) / self.std

        return {'eeg': scaled_eeg,
                'label': label}

    
class GraphStandardScaler():
    def __init__(self, mean=0, std=0, dataset=None):
        self.mean = mean
        self.std = std
        if dataset is not None:
            self.fit(dataset)

    def fit(self, dataset):
        X = []
        for data in dataset:
            X.append(data.x)

        X = torch.stack(X)
        self.mean = torch.mean(X, axis=0)
        self.std  = torch.std(X, axis=0)

    def transform(self, dataset):
        if isinstance(dataset, torch.utils.data.dataset.ConcatDataset):
            datas = dataset.datasets
        else:
            datas = [dataset]

        for D in datas:    
            slices = D.slices['x'] 
            for i, _ in enumerate(slices[:-1]):
                idx1, idx2 = slices[i], slices[i+1]
                D._data.x[idx1:idx2] = (D._data.x[idx1:idx2] - self.mean) / (self.std+1e-20)

    def fit_transform(self, dataset):
        self.fit(dataset)
        self.transform(dataset)



# Feature transformation
def extract_features(X, features):

    X_feat = [feat(X) for feat in features]
    X_feat = np.stack(X_feat, axis=-1)

    return X_feat 


def mean(X):
    return np.mean(X, axis=-1)

def std(X):
    return np.std(X, axis=-1)

def ptp(X):
    return np.ptp(X, axis=-1)

def var(X):
    return np.var(X, axis=-1)

def minim(X):
    return np.min(X, axis=-1)

def maxim(X):
    return np.max(X, axis=-1)

def argminim(X):
    return np.argmin(X, axis=-1)

def argmaxim(X):
    return np.argmax(X, axis=-1)

def rms(X):
    return np.sqrt(np.mean(X**2, axis=-1))

def abs_diff_signal(X):
    return np.sum(np.abs(np.diff(X, axis=-1)), axis=-1)

def skewness(X):
    return stats.skew(X, axis=-1)

def kurtosis(X):
    return stats.kurtosis(X, axis=-1)
