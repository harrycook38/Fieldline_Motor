import os

import numpy as np
import pandas as pd

import mne
from mne_bids import (
    BIDSPath,
    make_dataset_description,
    print_dir_tree,
    read_raw_bids,
    write_meg_calibration,
    write_meg_crosstalk,
    write_raw_bids,
)
#%% Load Data
data_path = 'Y:\Harry_TMS\Motor\Motor\sub-P450-ania-2' #CHANGE
file_name = '20241211_161104_sub-Pilot2_file-Motor1_raw.fif'#CHANGE
raw_fname = os.path.join(data_path, file_name)

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw_resampled_fname = raw_fname.replace('.fif', f'_rs.fif')

event_fname = raw_fname.replace('.fif', f'_event.fif')

bids_folder = os.path.join(data_path, "sub-P450_BIDS") #CHANGE

#%% Lowpass and resample
desired_sfreq = 1000
current_sfreq = raw.info['sfreq']

lowpass_freq = desired_sfreq / 4.0
raw_resampled = raw.copy().filter(l_freq=None, h_freq=lowpass_freq)

raw_resampled.resample(sfreq=desired_sfreq)

# %% Save data
raw_resampled.save(raw_resampled_fname, overwrite=True)

#Verify save

print(raw_resampled.info)
# %% Convert to BIDs

del raw, raw_resampled
raw = mne.io.read_raw(raw_resampled_fname)

#Find stim

events = mne.find_events(raw, stim_channel="di32")
event_dict = {
     "stim_on": 64,
     "stim_off": 208
}

# Keep only the events whose IDs are in the event_dict
valid_event_ids = set(event_dict.values())
filtered_events = np.array([ev for ev in events if ev[2] in valid_event_ids])

# Save only the filtered events
mne.write_events(event_fname, filtered_events, overwrite=True)

raw.info["line_freq"] = 50

subject = '01'
session = '01'
task = 'MotorEvoked'
run = '01'

bids_path = BIDSPath(
    subject=subject, 
    session=session, 
    task=task, 
    run=run, 
    datatype="meg", 
    root=bids_folder
)

write_raw_bids(
    raw=raw,
    bids_path=bids_path,
    events=event_fname,
    event_id=event_dict,
    overwrite=True,
)

print_dir_tree(bids_folder)
# %%
