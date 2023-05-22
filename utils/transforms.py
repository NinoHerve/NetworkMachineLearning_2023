import torch
import numpy as np



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

        return {'eeg': torch.from_numpy(eeg),
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
    

class StandardScaler(object):
    """Standardize samples"""

    def __init__(self, mean, std):
        self.mean_ = np.expand_dims(mean, axis=1)
        self.var_  = np.expand_dims(std, axis=1)

    def __call__(self, sample):
        """Standardize data

        Args:
            sample (dict): EEG data of shape (n_channels, n_time_points) and labels

        Returns:
            (dict): Rescaled EEG data and labels
        """
        eeg, label = sample['eeg'], sample['label']

        scaled_eeg = (eeg - self.mean_) / np.sqrt(self.var_)

        return {'eeg': scaled_eeg,
                'label': label}