import os
import mne
import numpy as np
import scipy.io as scio
from scipy.signal import butter, lfilter, cheby2

def bandpass_filter(signal, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = cheby2(order, 60, [low, high], btype='band')  # butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    
    return y


session_types = ['T', 'E']
SUBJECTS_BCI_IV = ['A{0:02d}'.format(i) for i in range(1, 10)]
data_root = '/media/storage/data/dataset_2a'
gdf_name = 'BCICIV_2a_gdf'
label_name = 'true_labels'

processed_name = './BCICIV_2a_processed'
filted_name = './BCICIV_2a_fbcsp'

if(not os.path.exists(os.path.join(data_root, processed_name))):
    os.makedirs(os.path.join(data_root, processed_name))
if(not os.path.exists(os.path.join(data_root, filted_name))):
    os.makedirs(os.path.join(data_root, filted_name))

for name in SUBJECTS_BCI_IV:
    for session_type in session_types:
        label = scio.loadmat(os.path.join(data_root, label_name, name + session_type + '.mat'))['classlabel']
        data = mne.io.read_raw_gdf(os.path.join(data_root, gdf_name, name + session_type + '.gdf'))

        events = mne.events_from_annotations(data)
        epochs = mne.Epochs(data, events[0], event_id=events[1]['768'], tmin=2, tmax=6, baseline=None, detrend=None, preload=True) # Create epochs with start event (code=768) as trigger
        epoched_data = epochs.get_data()

        assert epoched_data.shape[0] == label.shape[0], "Trials and label counts do not match"

        epoched_data = np.nan_to_num(epoched_data)  # wipe off NaN
        epoched_data = epoched_data[:, :22, :1000]  # Keep only EEG channels
        epoched_data = np.transpose(epoched_data, (2, 1, 0)) # Expected shape for downstream code
       
        mat = {'data': epoched_data, 'label': label}
        scio.savemat(os.path.join(data_root, processed_name, name + session_type + '.mat'), mat)

        filter_data = bandpass_filter(epoched_data, 4, 40, 250)
        mat = {'data': filter_data, 'label': label}
        scio.savemat(os.path.join(data_root, filted_name, name + session_type + '.mat'), mat)

        print("Finished processing", name + session_type)

print("Finished processing all subjects.")