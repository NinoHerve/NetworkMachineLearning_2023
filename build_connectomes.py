import argparse 
from pathlib import Path
from utils.connectome import build_subject_connectome
import os
import numpy as np
import mne


# -------------------------------------------------------

""" Choose the parameters for building your connectomes here

Only run build_connectomes.py when the parameters here
suit you.
"""

# path to Vepcon dataset
bids_dir = Path('/home/admin/work/data/ds003505-download/')

# path where to save EEGDataset
save_dir = Path(os.getcwd())

# build source space from scratch ? (20 min per subject)
rebuildSources = False

# subjects: 1-20 but never 5
subjects = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]                  

# -------------------------------------------------------



if __name__ == '__main__':

    # Build directories to save data
    folderName = Path(save_dir) / 'connectomes'
    os.makedirs(folderName)

    if rebuildSources:
        src_dir = None 
    else:
        src_dir = Path('/home/admin/work/data/vepcon_derivatives')    # external derivatives


    for sub in subjects:

        # Name of subject
        subject = f'sub-{str(sub).zfill(2)}'

        print(f'\n*** Building connectome for {subject}...')

        # build connectome for subject
        source, connectome, ids = build_subject_connectome(bids_dir, subject, src_dir)

        # create file names
        subjectFolderName = folderName / subject
        srcFileName = subjectFolderName / f'{subject}-src.fif'
        scFileName  = subjectFolderName / f'{subject}-sc'
        idsFileName = subjectFolderName / f'{subject}-ids'

        # save files
        print(f'Saving files for {subject} in {subjectFolderName}')
        os.makedirs(subjectFolderName)
        mne.write_source_spaces(srcFileName, source, overwrite=True)
        np.save(scFileName, connectome)
        np.save(idsFileName, ids)