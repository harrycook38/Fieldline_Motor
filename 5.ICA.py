
import mne
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import array
from pprint import pprint
import json
import os.path as op
from mne_bids import (
    BIDSPath,
    make_dataset_description,
    print_dir_tree,
    read_raw_bids,
    write_meg_calibration,
    write_meg_crosstalk,
    write_raw_bids,
)
from mne_bids.stats import count_events
from mne.preprocessing import ICA

subject = '01'
session = '01'
task = 'MotorEvoked'
run = '01'
meg_suffix = 'meg'
hfc_suffix = 'raw_hfc'
ann_suffix = 'ann'
ica_suffix = 'ica'

data_path='Y:\Harry_TMS\Motor\Motor\sub-P450-ania-2'
deriv_root = op.join(data_path, "sub-P450_BIDS/derivatives/preprocessing")

bids_path = BIDSPath(subject=subject, session=session, datatype='meg',
            task=task, run=run, suffix=ann_suffix, 
            root=deriv_root, extension='.fif', check=False)

deriv_fname_fif = bids_path.basename.replace(ann_suffix, ica_suffix) # fif output filename
deriv_fname_fif_1 = op.join(bids_path.directory, deriv_fname_fif)

print(bids_path)
print(deriv_fname_fif_1)

#%% Resamp and filter
# Read the raw data from the BIDS path
raw = read_raw_bids(
    bids_path=bids_path, 
    extra_params={'preload': True}, 
    verbose=True
)

# Process the raw data
raw_resmpl = raw.copy().pick('meg')  # Keep only MEG channels
raw_resmpl.filter(3, 30)  # Band-pass filter from 3 to 30 Hz
raw_resmpl.resample(200)  # Downsample to 200 Hz
# %% ICA algo

raw_resmpl_all = raw_resmpl
ica = ICA(method='fastica',
    random_state=96,
    n_components=30,
    verbose=True)

ica.fit(raw_resmpl_all,
    verbose=True)

#%% Identify comps

%matplotlib inline
ica.plot_sources(raw_resmpl_all, title='ICA');

#%% Topo
%matplotlib inline
ica.plot_components();

# %% Exclude components

# Set the components to exclude
ica.exclude = [2,4,7,9,14,17,18]

raw_ica = read_raw_bids(bids_path=bids_path,
                        extra_params={'preload':True},
                        verbose=True)
ica.apply(raw_ica)

raw_ica.save(deriv_fname_fif_1, overwrite=True)

# %%

chs = ['L102_bz-s73', 'L211_bz-s46', 'R112_bz-s39', 'R211_bz-s34']
chan_idxs = [raw.ch_names.index(ch) for ch in chs]

%matplotlib inline
raw.plot(order=chan_idxs, duration=5)

%matplotlib inline
raw_ica.plot(order=chan_idxs, duration=5)

# %%
