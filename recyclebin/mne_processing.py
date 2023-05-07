import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import mne
mne.set_log_level(verbose='ERROR')

'''
001: LOADMAT
raw = loadmat('/media/FastData3/douke/DrownessDetector/tutor/dataset.mat')
print(raw.keys())
print(np.unique(raw['substate']))
002: EEG CHANNEL CHECK
standard 10-20:
LPA RPA Nz 
Fp1 Fpz Fp2 
AF9 AF7 AF5 AF3 AF1 AFz AF2 AF4 AF6 AF8 AF10 
F9 F7 F5 F3 F1 Fz F2 F4 F6 F8 F10 
FT9 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 FT10 
T9 T7 C5 C3 C1 Cz C2 C4 C6 T8 T10
TP9 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 TP10
P9 P7 P5 P3 P1 Pz P2 P4 P6 P8 P10
PO9 OPO7 PO5 PO3 PO1 POz PO2 PO4 PO6 PO8 PO10
O1 Oz O2
O9 Iz O10
T3 T5 T4 T6
M1 M2
A1 A2
Original 40 channel names
'HEOL', 'HEOR', 'FP1':'Fp1', 'FP2':'Fp2', 'VEOU', 'VEOL',
'F7', 'F3', 'FZ':'Fz', 'F4', 'F8', 
'FT7', 'FC3', 'FCZ':'FCz', 'FC4', 'FT8', 
'T3', 'C3', 'CZ':'Cz', 'C4', 'T4', 
'TP7', 'CP3', 'CPZ':'CPz', 'CP4', 'TP8', 
'A1', 'T5', 'P3', 'PZ':'Pz', 'P4', 'T6', 'A2', 
'O1', 'OZ':'Oz', 'O2', 
'FT9', 'FT10',
'PO1', 'PO2'
003: OS_FOR_FOLDER_ANALYZE
subj_num = len(next(os.walk(dataset_root))[1])
file_num = len([name for name in os.listdir(dataset_root) if os.path.isfile(os.path.join(dataset_root, name))])
file_num = len(next(os.walk(dataset_root))[2])
file_num_recursive = sum(len(files) for _, _, files in os.walk(dataset_root))
004: ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=0)
ica.fit(raw.copy())
# ica.plot_components(outlines='skirt')
ica.exclude = [list]
# bad_idx, scores = ica.find_bads_eog(raw, 'S02', threshold=2)
ica.apply(raw.copy(), exclude=ica.exclude).plot()
005: CHANNEL_SELECTION
MANUAL
all_ch = raw.ch_names  
good_ch = ['FCZ', 'PZ', 'CZ', 'C3', 'C4','O1', 'O2', 'OZ',]
good_ch = set(good_ch)
bad_ch = []
for x in all_ch:
    if x not in good_ch:
        bad_ch.append(x)
picks = mne.pick_channels(all_ch, good_ch, bad_ch)
raw.plot(order=picks, n_channels=len(picks))
for x in bad_ch:
    raw.info['bads'].append(x)
AUTO
eeg_channel_indices = mne.pick_types(raw.info, meg=False, eeg=True)
eeg_data, times = raw[eeg_channel_indices]
print(eeg_data.shape)
006: CREATE_EPOCHS_FROM_EVENTS
events = mne.find_events(raw, stim_channel='***')
epochs = mne.Epochs(raw, events=events, event_id=2, tmax=6, tmin=1, baseline=(1, 6))'''
# https://mne.tools/stable/index.html
dataset_root = '/media/FastData3/douke/data'
data_from = 'dataset_figshare'
data_to = 'preprocessed'
subj_num = len(os.listdir(os.path.join(dataset_root, data_from)))
state = ['Fatigue', 'Normal']

for i in range(subj_num):
    for s in state:
        raw = mne.io.read_raw_cnt(os.path.join(dataset_root, data_from, '%d/%s state.cnt' %(i+1, s)),  
                                    eog=('HEOL', 'HEOR', 'VEOU', 'VEOL'), preload=True, data_format='int16')
        print('preprocessing ', os.path.join(dataset_root, data_from, '%d/%s state.cnt' %(i+1, s)))

        mapping = {'FP1':'Fp1', 'FP2':'Fp2', 'FZ':'Fz', 'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz', 'PZ':'Pz', 'OZ':'Oz'}
        raw.rename_channels(mapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)  # , on_missing='ignore')

        # ref.: Mikko A. Uusitalo and Risto J. Ilmoniemi. 
        # Signal-space projection method for separating MEG or EEG into components. 
        # Medical & Biological Engineering & Computing, 35(2):135–140, 1997. doi:10.1007/BF02534144.
        eog_projs, _ = mne.preprocessing.compute_proj_eog(raw, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True, eog_h_freq=10)
        if len(eog_projs):
            eog_evoked = mne.preprocessing.create_eog_epochs(raw).average(picks='all')
            eog_evoked.apply_baseline((None, None))
            mne.viz.plot_projs_joint(eog_projs, eog_evoked, 'eog')
            plt.savefig('./fig/%d_%s_proj.jpg' %(i+1, s))
            plt.close()
            raw.add_proj(eog_projs)
            raw.apply_proj()
        
        epochs = mne.make_fixed_length_epochs(raw, duration=4, preload=True)
        # if len(eog_projs):
        #     epochs.add_proj(eog_projs)
        #     epochs.apply_proj()

        epochs.filter(l_freq=4, h_freq=45)
        epochs = epochs.resample(sfreq=250)
        epochs.set_eeg_reference(['A1', 'A2'])  #  All channel data were referenced to two electrically linked mastoids at A1 and A2
        epochs.pick_types(eeg=True)
        epochs.plot_psd(fmax=60)
        plt.savefig('./fig/%d_%s_psd.jpg' %(i+1, s))
        plt.close()
        epochs.plot(n_epochs=7, n_channels=36)
        plt.savefig('./fig/%d_%s.jpg' %(i+1, s))
        plt.close()
        epoched_data = epochs.get_data()
        
        epoched_data = np.nan_to_num(epoched_data)  # wipe off NaN
        epoched_data = epoched_data[25:125, :]  # Keep only EEG channels
        if s == 'Normal':
            label = np.zeros(epoched_data.shape[0])
        else:
            label = np.ones(epoched_data.shape[0])

        mat = {'data': epoched_data, 'label': label}
        if not os.path.exists(os.path.join(dataset_root, data_to, '%d' %(i+1))):
            os.makedirs(os.path.join(dataset_root, data_to, '%d' %(i+1)))
        savemat(os.path.join(dataset_root, data_to, '%d/%s state.mat' %(i+1, s)), mat)


        '''
        raw.get_data().shape (40, 600880)
        raw.plot(start=2, duration=6, block=True), raw.plot_psd(), raw.plot_projs_topomap(), raw.plot_sensor() 

        
        mne.channels.find_layout(raw.info, ch_type='eeg').plot()
        plt.show()

        raw.plot_psd()
        plt.show()
        raw.notch_filter(50)
        raw.filter(l_freq=4, h_freq=45)
        raw.plot(start=0, duration=6, n_channels=40)
        raw.plot_psd()  # fmin=30, fmax=70, tmin=40,tmax=50.0,average=True)
        plt.show()

        # ref.: Mikko A. Uusitalo and Risto J. Ilmoniemi. 
        # Signal-space projection method for separating MEG or EEG into components. 
        # Medical & Biological Engineering & Computing, 35(2):135–140, 1997. doi:10.1007/BF02534144.
        eog_evoked = mne.preprocessing.create_eog_epochs(raw).average(picks='all')
        eog_evoked.apply_baseline((None, None))
        # eog_evoked.plot_joint()
        eog_projs, _ = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, reject=None, no_proj=True)
        raw.add_proj(eog_projs)
        mne.viz.plot_projs_topomap(eog_projs, info=raw.info)
        fig = mne.viz.plot_projs_joint(eog_projs, eog_evoked, 'eog')
        fig.suptitle('EOG projectors')
        eog_evoked = mne.preprocessing.create_eog_epochs(raw, h_freq=10).average(picks='all')
        eog_evoked.apply_baseline((None, None))
        eog_projs, _ = mne.preprocessing.compute_proj_eog(raw, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True, eog_h_freq=4)
        
        raw.pick_types(eeg=True)
        raw = raw.resample(sfreq=250)
        raw.set_eeg_reference(['A1', 'A2'])  #  All channel data were referenced to two electrically linked mastoids at A1 and A2
        raw.filter(l_freq=1, h_freq=45)
        eog_projs, _ = mne.preprocessing.compute_proj_eog(raw, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True, eog_h_freq=4)
        epochs.add_proj(eog_projs)
        epochs.apply_proj()
        
        '''



