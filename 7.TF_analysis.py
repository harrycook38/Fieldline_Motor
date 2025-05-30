import os.path as op
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import mne
from mne_bids import BIDSPath, read_raw_bids

subject = '01'
session = '01'
task = 'MotorEvoked'  # name of the task
run = '12'  # both runs compbined
meg_suffix = 'meg'
epo_suffix = 'epo'

data_path='Y:\Harry_TMS\Motor\Motor\sub-P450-ania-2'
deriv_root=op.join(data_path,"sub-P450_BIDS/derivatives/analysis")
#deriv_root=op.join(data_path,"BIDSRec_main/derivatives/analysis")
bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)
print(bids_path.basename,bids_path.fpath)

epochs = mne.read_epochs(bids_path.fpath,
                         proj = False,
                         preload=True,
                         verbose=False)

#%%
# Pick MEG sensors (magnetometers and gradiometers)
meg_picks = mne.pick_types(epochs.info, meg=True, eeg=False, stim=False, exclude=[])

# Collect names of MEG channels at the origin
bad_sensors = []
for idx in meg_picks:
    ch = epochs.info['chs'][idx]
    if np.allclose(ch['loc'][:3], [0., 0., 0.]):
        bad_sensors.append(ch['ch_name'])

print(f"MEG sensors at origin: {bad_sensors}")

# Drop just those
# Drop bad sensors
epochs.drop_channels(bad_sensors)

#%%

freqs = np.arange(1, 31, 1)
n_cycles = freqs / 3
time_bandwidth = 2.

tfr_stim_on =  mne.time_frequency.tfr_multitaper(
    epochs['stim_on'], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'mag', 
    use_fft=True, 
    return_itc=False,
    average=True,
    decim=2,
    n_jobs = -1,
    verbose=True)

# tfr_stim_off = mne.time_frequency.tfr_multitaper(
#     epochs['stim_off'], 
#     freqs=freqs, 
#     n_cycles=n_cycles,
#     time_bandwidth=time_bandwidth, 
#     picks = 'mag', 
#     use_fft=True, 
#     return_itc=False,
#     average=True,
#     decim=2,
#     n_jobs = -1,
#     verbose=True)


tfr_stim_on.apply_baseline(baseline=(-0.2,0.),mode='logratio', verbose=True)

# Plot the TFR with the percentage change already applied
tfr_stim_on.plot(
    picks=['L209_bz-s52'],  # Specify the channel to plot
    tmin=-0.5, tmax=1.5,    # Time range for the plot
    title='L209_bz-s52'     # Title of the plot
)
# %%

# Get the global min and max of the TFR data
vmin = tfr_stim_on.data.min()*0.8
vmax = tfr_stim_on.data.max() *1.1

print(f"vmin: {vmin:.2e}, vmax: {vmax:.2e}")

#%%

def quick_topo(tfr,name):
# plot tfrs for every channel at the location of the channel
    picks, pos, merge_channels, names, _, sphere, clip_origin = mne.viz.topomap._prepare_topomap_plot(
        tfr, 'mag', sphere=None
    )
    posx = pos.copy()
    posx = posx - pos.min(axis=0)
    posx = posx / posx.max(axis=0)
    posx = (0.8 * (posx - 0.5)) + 0.5

    plt.figure()
    for ii in range(pos.shape[0]):
        plt.axes([posx[ii, 0], posx[ii, 1], 0.05, 0.05])
        extent = (tfr.times[0], tfr.times[-1], tfr.freqs[0], tfr.freqs[-1])
        plt.imshow(tfr.data[ii, :, :], aspect="auto", cmap='RdBu_r',
                    origin='lower', extent=extent, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])

    plt.axes([0.1, 0.1, 0.1, 0.1])
    plt.imshow(tfr.data.mean(axis=0), aspect="auto", cmap='RdBu_r',
                origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    plt.title('Average')
    plt.savefig(name)

quick_topo(tfr_stim_on,'sensor map for beta desynch')
# %%
