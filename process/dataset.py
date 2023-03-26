import os
from scipy.io import loadmat, savemat
import mne
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset


class EEGdata():
    def __init__(self, nSub, p) -> None:
        super().__init__()
        self.dataset_root = '/media/FastData3/douke/data'
        self.data_from = 'dataset_figshare'
        self.data_to = 'preprocessed_%s'%p
        self.subj_num = len(os.listdir(os.path.join(self.dataset_root, self.data_from)))
        self.state = ['Normal', 'Fatigue']
        self.nSub = nSub

    
    def standardize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        return data
    
    
    def preprocess(self, nsub, s, epoch_duration=4, filt=True, proj=False, visualize=False, save_mat=False):
        try:
            load = loadmat(os.path.join(self.dataset_root, self.data_to, '%d/%s state.mat' %(nsub, s)))
            return load['data'], load['label'][0].transpose()
        except:
            raw = mne.io.read_raw_cnt(os.path.join(self.dataset_root, self.data_from, '%d/%s state.cnt' %(nsub+1, s)),  
                                        eog=('HEOL', 'HEOR', 'VEOU', 'VEOL'), preload=True, data_format='int16')
            print('preprocessing ', os.path.join(self.dataset_root, self.data_from, '%d/%s state.cnt' %(nsub+1, s)), '...')

            # set montage
            mapping = {'FP1':'Fp1', 'FP2':'Fp2', 'FZ':'Fz', 'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz', 'PZ':'Pz', 'OZ':'Oz'}
            raw.rename_channels(mapping)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)  
            
            # get epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)
            epochs = epochs.resample(sfreq=1000//epoch_duration)

            # set reference channels
            epochs.set_eeg_reference(['A1', 'A2'])  #  All channel data were referenced to two electrically linked mastoids at A1 and A2

            # FIR
            if filt:
                epochs.filter(l_freq=4, h_freq=45)

            # EOG SSP
            if proj:
                '''
                ref.: Mikko A. Uusitalo and Risto J. Ilmoniemi. 
                Signal-space projection method for separating MEG or EEG into components. 
                Medical & Biological Engineering & Computing, 35(2):135–140, 1997. doi:10.1007/BF02534144.
                '''
                eog_projs, _ = mne.preprocessing.compute_proj_eog(raw, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True)
                if len(eog_projs):
                    eog_evoked = mne.preprocessing.create_eog_epochs(raw).average(picks='all')
                    eog_evoked.apply_baseline((None, None))
                    mne.viz.plot_projs_joint(eog_projs, eog_evoked, 'eog')
                    plt.savefig('./fig/%d_%s_proj.jpg' %(nsub+1, s))
                    plt.close()
                    raw.add_proj(eog_projs)
                    raw.apply_proj()
                    epochs.add_proj(eog_projs)
                    epochs.apply_proj()
            
            # get data
            epochs.pick_types(eeg=True)
            epoched_data = epochs.get_data()
            epoched_data = np.nan_to_num(epoched_data)  
            epoched_data = epoched_data[25:125, :]  # 600/epoch_duration

            # visualization
            if visualize:
                epochs.plot_psd(fmax=60)
                plt.savefig('./fig/%d_%s_psd.jpg' %(nsub+1, s))
                plt.close()
                epochs.plot(n_epochs=7, n_channels=36)
                plt.savefig('./fig/%d_%s.jpg' %(nsub+1, s))
                plt.close()

            # get label
            if s == 'Normal':
                label = np.zeros(epoched_data.shape[0])
            else:
                label = np.ones(epoched_data.shape[0])

            # save to .mat
            if save_mat:
                mat = {'data': epoched_data, 'label': label}
                if not os.path.exists(os.path.join(self.dataset_root, self.data_to, '%d' %(nsub+1))):
                    os.makedirs(os.path.join(self.dataset_root, self.data_to, '%d' %(nsub+1)))
                savemat(os.path.join(self.dataset_root, self.data_to, '%d/%s state.mat' %(nsub+1, s)), mat)

            return epoched_data, label


    def get_data(self):
        
        self.train_data, self.test_data = [], []
        self.train_label, self.test_label = [], []
        
        for s in self.state:
            for nsub in range(self.subj_num):
                epoched_data, label = self.preprocess(nsub+1, s)
                if (nsub+1) == self.nSub:
                    self.test_data.append(epoched_data)
                    self.test_label.append(label)
                else:
                    self.train_data.append(epoched_data)
                    self.train_label.append(label)

        self.train_data = np.expand_dims(np.concatenate(self.train_data), axis=1)
        self.train_label = np.concatenate(self.train_label)

        sample_num = len(self.train_label)
        shuffle = np.random.permutation(sample_num)
        self.train_data = self.train_data[shuffle, :, :]
        self.train_label = self.train_label[shuffle]

        self.test_data = np.expand_dims(np.concatenate(self.test_data), axis=1)
        self.test_label = np.concatenate(self.test_label)
        # test dataset doesnot need to shuffle
        # sample_num = len(self.test_label)
        # shuffle = np.random.permutation(sample_num)
        # self.test_data = self.test_data[shuffle, :, :]
        # self.test_label = self.test_label[shuffle]

        data = torch.from_numpy(self.standardize(self.train_data))
        label = torch.from_numpy(self.train_label)
        train_dataset = TensorDataset(data, label)

        data = torch.from_numpy(self.standardize(self.test_data))
        label = torch.from_numpy(self.test_label)
        test_dataset = TensorDataset(data, label)
        
        return train_dataset, test_dataset


    def interaug(self, batch_size):  
        '''
        Segmentation and Reconstruction (S&R) data augmentation

        return: augmented np.array of shape (int(batch_size / 2), 1, 36, 1000), adn its label
        '''
        aug_data = []
        aug_label = []
        for cls4aug in range(2):
            cls_idx = np.where(self.train_label == cls4aug)
            tmp_data = self.train_data[cls_idx]
            tmp_label = self.train_label[cls_idx]

            tmp_aug_data = np.zeros((int(batch_size / 2), 1, 36, 1000))
            for ri in range(int(batch_size / 2)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(batch_size / 2)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label
    
