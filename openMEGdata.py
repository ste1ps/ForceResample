import sys
import os
#sys.path.insert(0, '/home/ficarella.s/PycharmProjects/ForceResample/')

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
os.chdir('./../../../')

list_dir = os.listdir(path_dir)

os.chdir(path_dir)
for s in subject:
    if s[2:] == '12':
        totrun = 5
    else:
        totrun = 4
        # subject number without sb s[2:]
    for run in np.arange(totrun):
        megDir = './' + s + '/MEG/' + str(run + 1) + '/'
        os.chdir(megDir)
        raw = mne.io.read_raw_bti('c,rfDC', preload=True)
        print('Processing file ' + s +'block' + str(run + 1) )
        events = mne.find_events(raw)
        resp_events = mne.find_events(raw, stim_channel='STI 013')
        print ('Found %s resp_events ' % len(resp_events))
        print np.unique(resp_events[:, 2])
        print('Found %s events, first five:' % len(events))
        print(events[:5])

        # Define Stim-trigger codes
        Trig_id = [522, 532, 542, 552]
        # get stimuli events
        stim_events = events[np.logical_or.reduce([events[:, -1] == _id for _id in Trig_id])]
        Trig_id_size = stim_events.shape[0]

        if Trig_id_size != 160:
            raise ValueError('stimuli events are not 160!')

        Trig_respid = [2432, 2944, 3456]
        response_events = resp_events[np.logical_or.reduce([resp_events[:, -1] == _id for _id in Trig_respid])]
        Trig_respid_size = response_events.shape[0]
        # low-pass filtered to 250 Hz
        raw._data = mne.filter.low_pass_filter(raw._data, Fs=1000, Fp=250,picks=None, method = 'fft')


        # Keep only MEG data
        # meg = raw.pick_types(meg=True)
        # create epochs
        epochs_stim = mne.Epochs(raw, stim_events, event_id=Trig_id, tmin=-1.5,
                                 tmax=1.5, baseline=(-1.5, -1), preload=True)
        print epochs_stim
        epochs_stim.plot_sensors(kind='topomap', ch_type='mag')
        epochs_stim.plot(n_channels=20)
        evoked_square = epochs_stim['522'].average()
        evoked_triangle = epochs_stim['532'].average()
        evoked_circle = epochs_stim['542'].average()
        evoked_diamond = epochs_stim['552'].average()

        epoch_resp = mne.Epochs(raw, response_events, event_id=Trig_respid, tmin=-3.5,
                                tmax=1, baseline=(-3.5, -3), preload=True)
        print epoch_resp
        epoch_resp.plot_sensors(kind='topomap', ch_type='mag')
        epoch_resp.plot(events=response_events, n_channels=20)
        evoked_resp_right = epoch_resp['2944'].average()
        evoked_resp_left = epoch_resp['2432'].average()
        evoked_noresp = epoch_resp['3456'].average()


