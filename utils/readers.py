# Functions to read files from VEPCON data set

import mne
import nibabel as nb
import numpy as np
import pandas as pd

def read_evokeds(bids_dir, subject, task, condition):
    """
    Reads mne.Evoked eeg signals

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

        task: str [faces, motions]
            type of task

        condition: str [Faces, Scrambled, Coherent, Random]
            condition of task

    Returns:
        evoked: mne.evoked.Evoked
            average eeg signal of multiple trials
    """

    fileName = bids_dir / 'derivatives' / 'mne' / subject / f'{subject}_task-{task}_cond-{condition}_ave.fif'
    
    evoked = mne.read_evokeds(fileName)[0]

    return evoked


def read_cov(bids_dir, subject, task, condition):
    """
    Reads covariance matrix of evoked signals

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

        task: str [faces, motions]
            type of task

        condition: str [Faces, Scrambled, Coherent, Random]
            condition of task

    Returns:
        noiseCov: mne.cov.Covariance
            noise covariance matrix
    """

    fileName = bids_dir / 'derivatives' / 'mne' / subject / f'{subject}_task-{task}_cond-{condition}_cov.fif'

    noiseCov = mne.read_cov(fileName)

    return noiseCov


def read_bem_solution(bids_dir, subject):
    """
    Reads BEM (boundary element model) solution

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        bem: mne.bem.ConductorModel
            boundary element model solution
    """

    fileName = bids_dir / 'derivatives' / 'freesurfer-7.1.1' / subject / 'bem'/ f'{subject}_conductor_model.fif'
    
    bem = mne.read_bem_solution(fileName)

    return bem 


def read_fmri(bids_dir, subject, task):
    """
    Read fMRI timeseries

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        fmri: ndarray (n,1)
            fmri timeseries
    """

    fileNameLH = bids_dir / 'derivatives' / 'ants' / subject / task / f'lh.{subject}_con_0003_t1_space_surf.mgh'
    fmriSurf   = nb.freesurfer.mghformat.load(fileNameLH)
    lhData     = fmriSurf.get_fdata()[:,0,0]        # Why this indexing

    fileNameRH = bids_dir / 'derivatives' / 'ants' / subject / task / f'rh.{subject}_con_0003_t1_space_surf.mgh'
    fmriSurf = nb.freesurfer.mghformat.load(fileNameRH)
    rhData   = fmriSurf.get_fdata()[:,0,0]

    return np.concatenate((lhData, rhData)).reshape(-1, 1)


def read_tractography(bids_dir, subject):
    """
    Read tractography

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        tracks: nibabel.streamlines
            streamlines of the tractography
    """
    
    fileName = bids_dir / 'derivatives' / 'cmp-v3.0.3' / subject / 'dwi' / f'{subject}_model-CSD_desc-DET_tractogram.trk'

    tracks = nb.streamlines.load(fileName)

    return tracks


def read_source_space(bids_dir, subject):
    """
    Read source space

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        source: mne.SourceSpace
            source space
    """

    fileName = bids_dir / 'derivatives' / 'cscs' / subject / 'src' / f'{subject}-src.fif'

    source= mne.read_source_spaces(fileName)

    return source 


def read_forward(bids_dir, subject):
    """
    Read forward matrix (lead-field matrix)

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        forward: mne.
            forward matrix (lead-field matrix)
    """

    fileName = bids_dir / 'derivatives' / subject / f'{subject}_fwd.fif'

    forward = mne.read_forward_solution(fileName, verbose=False)

    return forward


def read_connectome(bids_dir, subject):
    """
    Read connectome

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        connectome: ndarray
            connectome
    """

    fileName = bids_dir / 'derivatives' / 'cscs' / subject / f'{subject}-sc.npy'

    connectome = np.load(fileName)

    return connectome


def read_ids(bids_dir, subject):
    """
    Read ids???

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        ids: ndarray
            ???
    """
    fileName = bids_dir / 'derivatives' / 'cscs' / subject / f'{subject}-ids.npy'

    ids = np.load(fileName)

    return ids


def read_trans(bids_dir, subject):
    """
    Read transformation operator for eeg<->MRI coregistration. Applying trans on eeg
    channel positions will correctly position the channels in the MRI space.

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        trans: ndarray
            transformation matrix
    """
    fileName = bids_dir / 'derivatives' / 'mne' / subject / f'{subject}_trans.fif'

    trans = mne.read_trans(fileName)

    return trans


def read_epochs(bids_dir, subject, task):
    """Read raw eeg files
    
    Args:
        bids_dir (Path): path to data folder
        subject (str): subject name
        task (str): task name

    Return:
        mne.raw: concatenated eeg data
    """

    fileName = bids_dir / 'derivatives' / 'eeglab-v14.1.1' / subject / 'eeg' / f'{subject}_task-{task}_desc-preproc_eeg.set'

    epochs = mne.read_epochs_eeglab(
            fileName,
            events=None,
            event_id=None,
            eog=(),
            verbose=0,
            uint16_codec=None,
        )

    return epochs


def read_events(bids_dir, subject, task):
    """Read event file
    
    Args:
        bids_dir (Path): path to data folder
        subject (str): subject name
        task (str): task name

    Return:
        pd.DataFrame: event informations
    """

    fileName = bids_dir / subject / 'eeg' / f'{subject}_task-{task}_events.tsv'

    events = pd.read_table(fileName)

    return events