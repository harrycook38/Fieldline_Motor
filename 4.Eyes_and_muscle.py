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
from mne.preprocessing import annotate_muscle_zscore

subject = '01'  
session = '01'  
task = 'MotorEvoked'
run = '01'  
meg_suffix = 'meg'
hfc_suffix = 'raw_hfc'
ann_suffix = 'ann'

subject = '01'  
session = '01'  
task = 'MotorEvoked'
run = '01'  
meg_suffix = 'meg'
meg_extension = '.fif'
events_suffix = 'events'
events_extension = '.tsv'

data_path='Y:\Harry_TMS\Motor\Motor\sub-P450-ania-2'

deriv_root = op.join(data_path, "sub-P450_BIDS/derivatives/preprocessing")


bids_path = BIDSPath(subject=subject, session=session, datatype='meg',
            task=task, run=run, suffix=hfc_suffix, 
            root=deriv_root, extension='.fif', check=False)

deriv_fname_fif = bids_path.basename.replace(hfc_suffix, ann_suffix) # fif output filename
deriv_fname_fif_1 = op.join(bids_path.directory, deriv_fname_fif)
deriv_fname_csv_1 = deriv_fname_fif_1.replace('fif', 'csv') # csv output filename

print(bids_path)
print(deriv_fname_fif_1)
print(deriv_fname_csv_1)


#%% read in

raw = read_raw_bids(bids_path=bids_path, 
                     extra_params={'preload':True},
                     verbose=True)
# %%
eog_events = mne.preprocessing.find_eog_events(raw, ch_name='s16_bz') 
# %%
n_blinks = len(eog_events)
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
description = ['blink'] * n_blinks
orig_time = raw.info['meas_date']
annotations_blink = mne.Annotations(onset, duration, description, orig_time)
# %%
threshold_muscle = 10  
annotations_muscle, scores_muscle = annotate_muscle_zscore(
    raw, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
    filter_freq=[110, 140])
%matplotlib inline
fig1, ax = plt.subplots()
ax.plot(raw.times, scores_muscle);
ax.axhline(y=threshold_muscle, color='r')
ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity (threshold = %s)' % threshold_muscle)

annotations_event = raw.annotations 
raw.set_annotations(annotations_event + annotations_blink + annotations_muscle)

#%%
%matplotlib qt
raw.plot(start=50)
# %%
raw.save(deriv_fname_fif_1, overwrite=True)
raw.annotations.save(deriv_fname_csv_1, overwrite=True)
# %%
