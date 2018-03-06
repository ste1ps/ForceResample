
import os
import mne
import numpy as np

subject='sb18'

subject_dir = '/envau/work/comco/ficarella.s/etudeMEG/forFrioul/{0}/MEG/1'.format(subject)

fname_bti = subject_dir + '/c,rfDC'
fname_config = subject_dir + '/config'
fname_hs = subject_dir + '/hs_file'

raw = mne.io.read_raw_bti(fname_bti, fname_config, fname_hs, rename_channels=False, sort_by_ch_name=True, preload=True)

events = mne.find_events(raw)
Trig_id = [522, 532, 542, 552]
stim_events = events[np.logical_or.reduce([events[:, -1] == _id for _id in Trig_id])]
Trig_id_size = stim_events.shape[0]
meg = raw.pick_types(meg=True)

epochs_stim = mne.Epochs(meg, stim_events, event_id=Trig_id, tmin=-1.5,
                         tmax=1.5, baseline=(-1.5, -1), preload=True)

epochs_stim.save('/hpc/comco/ficarella.s/etudeMEG/db_mne/proACT/subject_18/trans/{0}_temp_-epo.fif'.format(subject))