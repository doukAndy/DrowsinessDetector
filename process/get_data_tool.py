import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import mne
mne.set_log_level(verbose='ERROR')

dataset_root = './data'
data_from = 'dataset_figshare'
data_to = 'raw_cnt2mat'
subj_num = len(next(os.walk(os.path.join(dataset_root, data_from)))[1])
state = ['Fatigue', 'Normal']

for i in range(subj_num):
    for s in state:
        raw = mne.io.read_raw_cnt(os.path.join(dataset_root, data_from, '%d/%s state.cnt' %(i+1, s)),  
                                    eog=('HEOL', 'HEOR', 'VEOU', 'VEOL'), preload=True, data_format='int16')
        print('preprocessing ', os.path.join(dataset_root, data_from, '%d/%s state.cnt' %(i+1, s)), '...')

        mapping = {'FP1':'Fp1', 'FP2':'Fp2', 'FZ':'Fz', 'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz', 'PZ':'Pz', 'OZ':'Oz'}
        raw.rename_channels(mapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage) 

        # if wanting to get only EEG channels and a uniform length of 600,000:
        # raw.pick_types(eeg=True)
        # raw_data = raw.get_data()[:, :600000]

        raw_data = raw.get_data() 
        print(raw_data.shape)  
        raw_data = np.nan_to_num(raw_data)
        if s == 'Normal': label = np.zeros(raw_data.shape[1])
        else: label = np.ones(raw_data.shape[1])
        
        mat = {'data': raw_data, 'label': label}
        if not os.path.exists(os.path.join(dataset_root, data_to, '%d' %(i+1))):
            os.makedirs(os.path.join(dataset_root, data_to, '%d' %(i+1)))
        savemat(os.path.join(dataset_root, data_to, '%d/%s state.mat' %(i+1, s)), mat)
        print('successfully saved transformed dataset to ', os.path.join(dataset_root, data_to, '%d/%s state.mat\n' %(i+1, s)))

# check
raw = loadmat(os.path.join(dataset_root, data_to, '1/Normal state.mat'))
print(raw['data'].shape, raw['label'].shape)  
# (40, 600880) (1, 600880) label = raw['label'][0].transpose() -> (600880,)
        


'''
CHANNEL REFERENCE TABLE

**standard 10-20:

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

**original 40 channel names
'HEOL', 'HEOR', 'FP1':'Fp1', 'FP2':'Fp2', 'VEOU', 'VEOL',
'F7', 'F3', 'FZ':'Fz', 'F4', 'F8', 
'FT7', 'FC3', 'FCZ':'FCz', 'FC4', 'FT8', 
'T3', 'C3', 'CZ':'Cz', 'C4', 'T4', 
'TP7', 'CP3', 'CPZ':'CPz', 'CP4', 'TP8', 
'A1', 'T5', 'P3', 'PZ':'Pz', 'P4', 'T6', 'A2', 
'O1', 'OZ':'Oz', 'O2', 
'FT9', 'FT10',
'PO1', 'PO2'
'''
