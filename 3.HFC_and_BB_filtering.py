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

subject = '01'  
session = '01'  
task = 'MotorEvoked'
run = '01'  
meg_suffix = 'meg'
meg_extension = '.fif'
events_suffix = 'events'
events_extension = '.tsv'

data_path='Y:\Harry_TMS\Motor\Motor\sub-P450-ania-2'
bids_root = op.join(data_path, "sub-P450_BIDS")
bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=meg_suffix, extension=meg_extension, root=bids_root)

#Create internal folder to save processed data
deriv_root = op.join(bids_root, 'derivatives/preprocessing')  # output path

deriv_path = BIDSPath(subject=subject, session=session, datatype='meg',
            task=task, run=run, suffix=meg_suffix, root=deriv_root).mkdir()

deriv_fname = bids_path.basename.replace('meg', 'raw_hfc') # output filename
deriv_file_1 = op.join(deriv_path.directory, deriv_fname)

raw = read_raw_bids(bids_path=bids_path, verbose=False,extra_params={'preload':True})

#%% Identify bad sensors

n_fft = 2000
raw_PSD = raw.compute_psd(method="welch", fmin=0.1, fmax=120, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
psds = raw_PSD.get_data() # units are in T^2/Hz
freqs = raw_PSD.freqs 
psd_db = [10 * np.log10(psd * 1e30) for psd in psds] # Ref power: 1fT^2=1e-30T^2
average_power = np.mean(psd_db, axis=1)

raw_PSD.plot();

#%% Histogram of sensor power

plt.clf()
plt.figure(figsize=(10, 6))
plt.hist(average_power, bins=100, color='skyblue', edgecolor='black')
plt.xlabel('Average Power (dB) [fT^2/Hz]')
plt.ylabel('Number of Sensors')
plt.title('Histogram of Average Power Across Sensors')

sensor_names = raw_PSD.ch_names
sensors_above_threshold = [sensor_names[i] for i, avg in enumerate(average_power) if avg > 37.5]
print(sensors_above_threshold)

raw1=raw.copy()
raw1.info['bads']=sensors_above_threshold
#raw.info["bads"].extend(["L507_bz-s70"])
# %% PSD minus bad sensors

plt.clf()
raw1_PSD = raw1.compute_psd(method="welch", fmin=0.1, fmax=120, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2)).plot(exclude='bads')
plt.ylim(20, 90)
# Save the plot
output_file = "PSD_before_HFC.png"
plt.savefig(output_file, dpi=600, bbox_inches="tight")
plt.show()

# %% Raw timecourse

picks = mne.pick_types(raw1.info, meg=True,exclude='bads')

amp_scale = 1e12  # Converting to pico Tesla(pT)
stop = len(raw1.times) - 300
step = 500
data_ds, time_ds = raw1[picks[::5], :stop]
data_ds, time_ds = data_ds[:, ::step]* amp_scale , time_ds[::step]

plt.clf()
fig, ax = plt.subplots(layout="constrained")
plot_kwargs = dict(lw=1, alpha=1)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(
    ylim=(-3000, 3000), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)
ax.set(title="Before Applying HFC", **set_kwargs)

# %% HFC
raw2=raw1.copy()

picks = mne.pick_types(
    raw2.info,
    meg=True,
    eeg=False,
    eog=False,
    stim=False,
    exclude=['s16_bz']
)
# Now compute HFC projections only on the valid channels
projs = mne.preprocessing.compute_proj_hfc(raw2.info, order=2, picks=picks)
raw2.add_proj(projs).apply_proj(verbose="error")

# plot
data_ds, _ = raw2[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale

fig, ax = plt.subplots(layout="constrained")
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True, ls=":")
ax.set(title="After HFC", **set_kwargs)

plt.clf()

raw2_PSD = raw2.compute_psd(method="welch", fmin=1, fmax=120, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2)).plot(show=False,exclude='bads')
# Add title and labels using Matplotlib
plt.title("(PSD after HFC)")
#plt.show()
for ax in raw2_PSD.axes:
    ax.set_ylim(20, 90) 

# Save the plot
output_file = "PSD_after_HFC.png"
plt.savefig(output_file, dpi=600, bbox_inches="tight")
plt.show()

# %% Construct and apply bandpass filter
# notch
raw2.notch_filter(np.arange(50, 251, 50), notch_widths=4)
# bandpass
raw3=raw2.copy()
raw3.filter(3, 120, picks="meg")
# plot
data_ds, _ = raw3[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale

%matplotlib inline
plt.clf()
fig, ax = plt.subplots(layout="constrained")
plot_kwargs = dict(lw=1, alpha=0.5)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(
    ylim=(-10, 10), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)
ax.set(title="After HFC and bandpass(3-120Hz)", **set_kwargs)
# %%
raw3_PSD = raw3.compute_psd(method="welch", 
                            fmin=1, 
                            fmax=120, 
                            picks="mag", 
                            n_fft=n_fft, 
                            n_overlap=int(n_fft/2))
raw3_PSD.plot(exclude='bads')
plt.title("HFC+ Bandpass (3-120Hz)")


# %% sensor-by-sensor PSD
""" n_fft = 2000
raw_PSD = raw3.compute_psd(method="welch", fmin=1, fmax=120, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
psds = raw_PSD.get_data() # units are in T^2/Hz
freqs = raw_PSD.freqs 


# Convert to db
avg_psd = [10 * np.log10(psd * 1e30) for psd in psds] # Ref power: 1fT^2=1e-30T^2
average_power = np.mean(avg_psd, axis=1)
## Get Only MEG channels
mag_channels = mne.pick_types(raw3.info, meg=True, stim=False)
channel_names = [raw3.info['ch_names'][i] for i in mag_channels]

ch_power_data = list(zip(channel_names, average_power, avg_psd))
ch_power_data_s = sorted(ch_power_data, key=lambda x: x[1], reverse=True) # Sorting by average power

s_channel_names = [channel for channel, power, psd in ch_power_data_s] #Sorted Channel names
s_average_power = [power for channel, power, psd in ch_power_data_s]
s_psds = [psd for channel, power, psd in ch_power_data_s]


n_channels = len(s_channel_names)
n_cols = 4  
n_rows = 17

ymin = 20 
ymax = 70 
grand_avg_psd = np.mean(s_psds, axis=0)

plt.clf()

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for rank, (channel, avg_power, psd) in enumerate(ch_power_data_s):
    ax = axes[rank]
    ax.plot(freqs, psd, label=f'{channel}')
    ax.plot(freqs, grand_avg_psd, label='Average', color='red')
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('Power dB [fT^2/Hz]')
    avg_psd_value = np.mean(psd)
    ax.set_title(f'Rank {rank + 1}, Mean PSD: {avg_psd_value:.2f} dB')
    #ax.set_ylim(ymin, ymax)
    ax.legend()
    ax.grid(True)

# Delete remaining unused subplots
for ax in axes[len(s_channel_names):]:
    ax.axis('off')
plt.tight_layout()
plt.show() """ #shift+alt+A to undo large comment. 

#%% save to bids directory 

raw_hfc=raw3.copy()
raw_hfc.save(deriv_file_1, overwrite=True) 