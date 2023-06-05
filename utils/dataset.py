# make a method in data that returns a Standardizer
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from utils.transforms import StandardScaler
from utils.graph_utils import pearson_corr_coef
from torch_geometric.data import InMemoryDataset, Data
import shutil
import pandas as pd

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

        labels = [l for l in os.listdir(eeg_dir) if not l.startswith('.')]    # first two folders seperate data into labels
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
    

class GraphDataset(InMemoryDataset):
    """EEG dataset"""

    def __init__(self, root, subjects, feat_transform, graph_type = 'electrode_adjacency', reprocess=False):
        """__init__

        Args:
            root (Path): path to dataset folder
            subjects (list): list of subjects to process
            feat_transform (function): transformation to apply on features
            reprocess (bool): reprocess data, else data will be loaded from 
            previous processings
        """
        self.raw_files = self.files_of_subjects(root/'raw', subjects)
        self.feat_transform = feat_transform
        self.graph_type = graph_type

        if reprocess:
            if os.path.exists(root/'processed'):
                shutil.rmtree(root/'processed')

        super().__init__(root)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property 
    def raw_file_names(self):
        '''Get all raw file names'''
        rf = [f'{p.parts[-2]}/{p.parts[-1]}' for p in self.raw_files]
        return rf
    
    @property
    def processed_file_names(self):
        '''Get all processed file names'''
        return ['data.pt']
    
    def process(self):
        '''Read data into huge Data list'''
        X, y = self.load_data()

        if self.graph_type == 'electrode_adjacency' :
            # Transform data
            if self.feat_transform is not None:
                X = self.feat_transform(X)
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            # load graph    
            edge_index = self.load_graph()
            data_list = [Data(X[i], edge_index, y=y[i]) for i in range(len(X))]

        elif self.graph_type == 'corr_coef':
            list_of_edges_index, list_of_edge_weights = [],[]
            for s in range(len(X)):
                trial = X[s]
                edge_index, edge_weights = self.load_graph_corr_coef(trial, threshold=False, binarize=False)
                list_of_edges_index.append(edge_index)
                list_of_edge_weights.append(edge_weights)
            # Transform data
            if self.feat_transform is not None:
                X = self.feat_transform(X)
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            data_list = [Data(X[i], edge_index, edge_weights = edge_weights, y=y[i]) for i, edge_index, edge_weights in zip(range(len(X)), list_of_edges_index, list_of_edge_weights)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def files_of_subjects(self, eeg_dir, subjects):
        """Retrieve the file names of specified subjects
        
        Args:
            eeg_dir (Path): path to eeg data
            subjects (list): list of subject name
            
        Returns:
            (list): list of file names containing specified subject samples
        """

        labels = [l for l in os.listdir(eeg_dir) if not l.startswith('.')]    # first two folders seperate data into labels
        files  = []                                                  

        for l in labels:
            file_names = os.listdir(eeg_dir / l)
            for fn in file_names:
                if fn[:6] in subjects:              # fn[:6] corresponds to subject name
                    files.append(eeg_dir / l / fn)

        return files
    
    def load_data(self):
        """Loads node features and labels"""

        # Load data of all files
        X = []
        y = []
        for fn in self.raw_files:
            X.append(np.load(fn))
            y.append(int(fn.parts[-2] == 'faces'))    # 1 = 'faces', 0 = 'scrambled'  

        X = np.array(X)

        return X, y

    def load_graph(self):
        """Build graph from electrode coordinates"""

        elec_coord_file = self.root/'electrode_coordinates.csv'
        elec_coord = pd.read_csv(elec_coord_file, index_col=0)
        
        # create edge when distance between two nodes smaller than 30
        positions = elec_coord.to_numpy()
        distances = np.linalg.norm(positions[:,None,:]-positions[None,:,:], axis=-1)
        A = (distances<30).astype(int)

        # create edge index COO format
        edge_index = []
        for i in range(len(A)):
            for j in range(i, len(A)):
                if A[i,j] > 0:
                    edge_index.append([i,j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_index = torch.cat([edge_index,edge_index[[1,0],:]], dim=1)   # make undirected

        return edge_index
    
    def load_graph_corr_coef(self, eeg, threshold=.5, binarize=True):
        """ Build graph by computing correlation coefficient between electrodes"""

        num_nodes = len(eeg)
        adj = np.zeros((num_nodes, num_nodes))

        # need to compute adj matrix to be able to threshold or binarize
        for i in range(1,num_nodes):
            for j in range(i+1, num_nodes):
                adj[i][j] = pearson_corr_coef(eeg[i], eeg[j])
        
        adj += adj.T
        if threshold:
            adj[adj<threshold] = 0
        
        if binarize:
            adj[adj!=0] = 1.

        # create edge index COO format
        edge_index = []
        edge_weights = []
        for i in range(len(adj)):
            for j in range(i, len(adj)):
                if adj[i,j] > 0:
                    edge_index.append([i,j])
                    edge_weights.append(adj[i,j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_index = torch.cat([edge_index,edge_index[[1,0],:]] , dim=1)   # make undirected
        edge_weights = torch.tensor(edge_weights)
        edge_weights = torch.cat([edge_weights, edge_weights])

        return edge_index, edge_weights