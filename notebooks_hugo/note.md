## Data handling

In torchEEG, datasets are built on top of pytorch Dataset class
SEEDDataset <- BaseDataset <- torch.util.data.Dataset

idea : make our dataset inherit from BaseDataset to have necessary functions to work with torchGeom

SEEDDataset defines :
__init__ 
_load_data
_set_files
__getitem__
repr_body