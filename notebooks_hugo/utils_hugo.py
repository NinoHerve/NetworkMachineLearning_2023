import os
import mne
import pandas as pd

def read_subject_epochs(bids_dir, subject, tmin=-0.2, tmax=0.5, apply_proj=True):
    eeglab_dir = os.path.join(bids_dir, "derivatives", "eeglab-v14.1.1", subject, "eeg")
    epochs_fname = os.path.join(
        eeglab_dir, f"{subject}_task-faces_desc-preproc_eeg.set"
    )

    eeg = os.path.join(bids_dir, subject, "eeg")
    beh_fname = os.path.join(
        eeg, f"{subject}_task-faces_events.tsv"
    )
    beh = pd.read_csv(beh_fname, sep="\t")
    beh = beh[beh.bad_epoch == 0]

    epochs = mne.read_epochs_eeglab(
            epochs_fname,
            events=None,
            event_id=None,
            eog=(),
            verbose=0,
            uint16_codec=None,
        )

    epochs.events[:, 2] = list(beh.trial_type_id)
    epochs.event_id = {"Scrambled": 0, "Faces": 1}

    epochs.apply_baseline((-0.2, 0))
    epochs.set_eeg_reference(ref_channels="average", projection=True)

    epochs.crop(tmin=tmin, tmax=tmax)

    if apply_proj :
        epochs.apply_proj()

    return epochs, beh.trial_type_id.values