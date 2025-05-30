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

#%% Inspect Data

subject = '01'  
session = '01'  
task = 'MotorEvoked'
run = '01'  
meg_suffix = 'meg'
meg_extension = '.fif'
events_suffix = 'events'
events_extension = '.tsv'

data_path='Y:\Harry_TMS\Motor\Motor\sub-P450-ania-2'  # CHANGE
bids_root = op.join(data_path, "sub-P450_BIDS")
bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=meg_suffix, extension=meg_extension, root=bids_root)

raw = read_raw_bids(bids_path=bids_path, verbose=False, 
                     extra_params={'preload':True})
print(raw)
print(raw.info)

# %% FFT
%matplotlib qt
n_fft = 2000
raw_PSD = raw.compute_psd(method="welch", fmin=1, fmax=120, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
raw_PSD.plot();

# %%
