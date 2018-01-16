import mne
import os
import numpy as np
import matplotlib.pyplot as plt

subject = ['sb1', 'sb2', 'sb3', 'sb4', 'sb5', 'sb6', 'sb7', 'sb8', 'sb9', 'sb10', 'sb11', \
           'sb12', 'sb15', 'sb16', 'sb17', 'sb18', 'sb19']
path = ('F:/MEG_data/proACT/')
os.chdir(path)
for s in subject:
    if s[2:] == '12':
        totrun = 5
    else:
        totrun = 4
        # subject number without sb s[2:]
    for run in np.arange(totrun):
        megDir = (path + s + '/MEG/' + str(run + 1) + '/')
        fullFileName = os.path.join(megDir, "c,rfDC")
        raw = mne.io.read_raw_bti(fullFileName, preload=True)
        print('Processing file ' + fullFileName)
        events = mne.find_events(raw)
        resp_events = mne.find_events(raw, stim_channel='STI 013')
        print('Found %s events, first five:' % len(events))
        print(events[:5])

        # Define Stim-trigger codes
        Trig_id = [522, 532, 542, 552]
        # get stimuli events
        stim_events = events[np.logical_or.reduce([events[:, -1] == _id for _id in Trig_id])]
        Trig_id_size = stim_events.shape[0]

        if Trig_id_size != 160:
            raise ValueError('stimuli events are not 160!')
        # low-pass filtered to 250 Hz
        # raw._data = mne.filter.low_pass_filter(raw._data, Fs=1000, Fp=250,picks=None, method = 'fft')
        raw._data = mne.filter.filter_data(raw._data, sfreq=raw.info['sfreq'], h_freq=0.01, l_freq=250, picks=None,
                                           method='fft')
        # Keep only MEG data
        # meg = raw.pick_types(meg=True)
        # create epochs
        epochs_stim = mne.Epochs(raw, stim_events, event_id=Trig_id, tmin=-1.5,
                                 tmax=1.5, baseline=(-1.5, -1), preload=True)
        epochs_stim.plot_sensors(kind='topomap', ch_type='mag')
        epochs_stim.plot(events=stim_events, n_channels=20)
