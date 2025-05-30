import os.path as op
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath, read_raw_bids

subject = '01'
session = '01'
task = 'MotorEvoked'
run = '01'
meg_suffix = 'meg'
ica_suffix = 'ica'
epo_suffix = 'epo'

data_path='Y:\Harry_TMS\Motor\Motor\sub-P450-ania-2'
deriv_root1 = op.join(data_path,"sub-P450_BIDS/derivatives/preprocessing")
deriv_root2 = op.join(data_path,"sub-P450_BIDS/derivatives/analysis")

bids_path_preproc = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=ica_suffix, datatype='meg',
            root=deriv_root1, extension='.fif', check=False)

bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=epo_suffix, datatype='meg',
            root=deriv_root2, extension='.fif', check=False).mkdir()

deriv_file = bids_path.basename.replace('run-01', 'run-12')  
deriv_fname = op.join(bids_path.directory, deriv_file)

print(bids_path_preproc.fpath)
print(deriv_fname)

#%%

raw = read_raw_bids(bids_path=bids_path_preproc, 
            extra_params={'preload':False},
            verbose=True)

# Reading the events from the raw file
events, events_id = mne.events_from_annotations(raw,event_id='auto')

%matplotlib inline
raw.plot(start=50)

%matplotlib inline
plt.stem(events[:,0][250:280], events[:,2][250:280])
plt.xlabel('samples')
plt.ylabel('Trigger value (di32)')
plt.show()
# %%
bids_path_preproc.update(run='01')
raw = read_raw_bids(bids_path=bids_path_preproc, 
                    extra_params={'preload': True}, 
                    verbose=True)
events, events_id = mne.events_from_annotations(raw, event_id='auto')

raw_list = [raw]
events_list = [events]


### Define event ids we are interested in
events_picks_id = {k: v for k, v in events_id.items() if k.startswith('stim')}
# %%
reject = dict(mag=9e-12)
## Make epochs
epochs = mne.Epochs(raw,
            events, events_picks_id,
            tmin=-0.5 , tmax=2,
            baseline=(-0.2, 0),
            proj=False,
            picks = 'all',
            detrend = 1,
            reject=reject,
            reject_by_annotation=True,
            preload=True,
            verbose=True)
## Show Epoch Details
epochs

# %%
%matplotlib inline
epochs.plot_drop_log();

#%%
%matplotlib qt
raw.plot_sensors(kind='3d', show_names=True)


# %%

print(deriv_fname)
epochs.save(deriv_fname, overwrite=True)

# epochs['stim_on'].plot(n_epochs=10, picks=['mag'])
# epochs['stim_on'].plot(n_epochs=1, picks=['stim'])

%matplotlib inline
epochs['stim_on'].filter(1,30).crop(-0.5,2).plot_image(picks = ['L207_bz-s75'],vmin=-2000, vmax=2000)
# %%


