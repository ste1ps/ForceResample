
import sys
import os
#sys.path.insert(0, '/home/ficarella.s/PycharmProjects/ForceResample/')
import matplotlib.pyplot as plt
import mne
import numpy as np

#os.getcwd()
#current directory


path_dir = u'envau/work/comco/ficarella.s/etudeMEG/forFrioul/'

sbj_types={'sb1':'dm','sb2':'dm','sb3':'dm','sb4':'dm','sb5':'dm','sb6':'bv','sb7':'bv','sb8':'bv',
           'sb9':'bv','sb10':'bv','sb11':'bv','sb12':'bv','sb15':'dm','sb16':'dm','sb17':'dm',
           'sb18':'dm','sb19':'dm'}
subject=['sb1','sb2','sb3','sb4','sb5','sb6','sb7','sb8','sb9','sb10','sb11',\
'sb12','sb15','sb16','sb17','sb18','sb19']
os.chdir('./../../../../../')
os.getcwd()
os.chdir(path_dir)
#list_dir = os.listdir(path_dir)



for s in subject:
    if s[2:] == '12':
        totrun = 5
    else:
        totrun = 4
        # subject number without sb s[2:]
        #for run in np.arange(totrun):
s='sb4'
run=0
megDir = './' + s + '/MEG/' + str(run + 1) + '/'
os.chdir(megDir)
raw = mne.io.read_raw_bti('c,rfDC', sort_by_ch_name=True, preload=True)
print('Processing file ' + s +'block' + str(run + 1) )

#raw_resampled = raw.resample(200,npad='auto')

events = mne.find_events(raw)
resp_events = mne.find_events(raw, stim_channel='STI 013')
print ('Found %s resp_events ' % len(resp_events))
print np.unique(resp_events[:, 2])
print('Found %s events, first five:' % len(events))
print(events[:5])

print(raw.info['chs'][0])
(raw.copy().pick_channels(['RFG 001', 'RFM 009', 'VEOG', 'EEG 001', 'EEG 002', 'EXT 002', 'EEG 003', 'EXT 001'])
           .plot())


#Artifact detection
from mne.datasets import sample
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
raw._data = mne.filter.high_pass_filter(raw._data, Fs=raw.info['sfreq'], Fp=1)
(raw.copy().pick_types(meg='mag')
           .plot(duration=60, n_channels=100, remove_dc=False))

raw.plot_psd(tmax=np.inf, fmax=45)


# Define Stim-trigger codes
Trig_id = [522, 532, 542, 552]

# Define Cue-trigger codes
Cue_id = [514, 516, 518, 520]

# get stimuli events
stim_events = events[np.logical_or.reduce([events[:, -1] == _id for _id in Trig_id])]
Trig_id_size = stim_events.shape[0]

if Trig_id_size != 160:
    raise ValueError('stimuli events are not 160!')

# get cue events
cue_events = events[np.logical_or.reduce([events[:, -1] == _id for _id in Cue_id])]
Cue_id_size = cue_events.shape[0]

Trig_respid = [2432, 2944, 3456]
response_events = resp_events[np.logical_or.reduce([resp_events[:, -1] == _id for _id in Trig_respid])]
Trig_respid_size = response_events.shape[0]
# low-pass filtered to 250 Hz
raw._data = mne.filter.low_pass_filter(raw._data, Fs=raw.info['sfreq'], Fp=45,picks=None, method = 'fft')
raw.copy().pick_types(meg='mag')
raw.plot_psd(tmax=100, fmax=250)

# The subsequent decoding analyses only capture evoked responses, so we can
# low-pass the MEG data. Usually a value more like 40 Hz would be used,
# but here low-pass at 20 so we can more heavily decimate, and allow
# the examlpe to run faster.
#raw_lowpass=raw.filter(None, 20., method='fft')



# Keep only MEG data
meg = raw.pick_types(meg=True)
ch_names=meg.ch_names
ch_indices = meg._get_channel_positions()
for i in np.arange(len(ch_names)):
    ch_names[i]=ch_names[i][3:7]
mne.viz.plot_sensors(meg.info, kind='topomap',show_names=True)
for i in np.arange(len(ch_names)):
    ch_names[i]='MEG' + ch_names[i]


# create epochs stim locked
epochs_stim = mne.Epochs(meg, stim_events, event_id=Trig_id, tmin=-2.5,
                         tmax=2.5, baseline=(-2.5, -1), preload=True)
print epochs_stim


# create epochs cue locked
epochs_cue = mne.Epochs(meg, stim_events, event_id=Trig_id, tmin=-1.5,
                         tmax=3.5, baseline=(-1.5, 0), preload=True)
print epochs_cue
#epochs_stim.plot_sensors(kind='topomap', ch_type='mag')
#epochs_stim.plot(n_channels=20)
evoked_square = epochs_stim['522'].average()
evoked_triangle = epochs_stim['532'].average()
evoked_circle = epochs_stim['542'].average()
evoked_diamond = epochs_stim['552'].average()
evoked_square.plot(spatial_colors=True, gfp=True)
#evoked_square.plot_sensors(kind='topomap', ch_type='mag')

epoch_resp = mne.Epochs(meg, response_events, event_id=Trig_respid, tmin=-3.5,
                        tmax=1, baseline=(-3.5, -3), preload=True)
print epoch_resp
#epoch_resp.plot_sensors(kind='topomap', ch_type='mag')
#epoch_resp.plot(n_channels=20)
evoked_resp_right = epoch_resp['2944'].average()
evoked_resp_left = epoch_resp['2432'].average()
evoked_noresp = epoch_resp['3456'].average()



tmin, tmax = [0, 60]  # use the first 60s of data
fmin, fmax = [2, 45]  # look at frequencies between 2 and 250Hz
n_fft = 1000

# Add SSP projection vectors to reduce EOG and ECG artifacts
#projs = read_proj(proj_fname)  ???
#raw.add_proj(projs, remove_existing=True)

plt.figure()
ax = plt.axes()
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
             n_jobs=1, proj='False', ax=ax, color=(0, 0, 1),  picks='mag',
             show='False')