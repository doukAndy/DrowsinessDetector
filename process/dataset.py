import os
from scipy.io import loadmat, savemat
import mne
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
from experiment.config import Config
cfg = Config()
mne.set_log_level(verbose=cfg.verbose)


class EEGdata():
    def __init__(self) -> None:
        super().__init__()
        

    def standardize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        # whether to add noise 
        return data
    
    
    def preprocess(self, nsub, s):
        try:
            load = loadmat(os.path.join(cfg.dataset_root, cfg.data_to, '%d/%s state.mat' %(nsub, s)))
            return load['data'], load['label'][0].transpose()
        except:
            raw = mne.io.read_raw_cnt(os.path.join(cfg.dataset_root, cfg.data_from, '%d/%s state.cnt' %(nsub, s)),  
                                        eog=('HEOL', 'HEOR', 'VEOU', 'VEOL'), preload=True, data_format='int16')
            print('preprocessing ', os.path.join(cfg.dataset_root, cfg.data_from, '%d/%s state.cnt' %(nsub, s)), '...')

            # set montage
            mapping = {'FP1':'Fp1', 'FP2':'Fp2', 'FZ':'Fz', 'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz', 'PZ':'Pz', 'OZ':'Oz'}
            raw.rename_channels(mapping)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)  
            # mne.channels.find_layout(raw.info, ch_type='eeg').plot()
            # get epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=cfg.epoch_duration, preload=True)
            epochs = epochs.resample(sfreq=cfg.samples//cfg.epoch_duration)

            # set reference channels
            epochs.set_eeg_reference(['A1', 'A2'])  #  All channel data were referenced to two electrically linked mastoids at A1 and A2

            # FIR
            if cfg.filt:
                epochs.filter(l_freq=cfg.lfreq, h_freq=cfg.hfreq)

            # EOG SSP
            if cfg.proj:
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
                    plt.savefig('./fig/%d_%s_proj.jpg' %(nsub, s))
                    plt.close()
                    raw.add_proj(eog_projs)
                    raw.apply_proj()
                    epochs.add_proj(eog_projs)
                    epochs.apply_proj()
            
            # get data
            epochs.pick_types(eeg=True)
            epoched_data = epochs.get_data()
            epoched_data = np.nan_to_num(epoched_data) 

            baseline_num = min(5, cfg.get_time[0])
            baseline = np.abs(epoched_data[(cfg.get_time[0] - baseline_num): cfg.get_time[0], :])
            ampmax = np.max(np.max(baseline, axis=-1, keepdims=True), axis=0)
            
            epoched_data = epoched_data[cfg.get_time, :] / ampmax  # 600/epoch_duration

            # visualization
            if cfg.visualize:
                epochs.plot_psd(fmax=60)
                plt.savefig('./fig/%d_%s_psd.jpg' %(nsub, s))
                plt.close()
                epochs[cfg.get_time].plot(n_epochs=5, n_channels=36)
                plt.savefig('./fig/%d_%s.jpg' %(nsub, s))
                plt.close()

            # get label
            if s == 'Normal':
                label = np.zeros(epoched_data.shape[0])
            else:
                label = np.ones(epoched_data.shape[0])

            # save to .mat
            if cfg.save_mat:
                mat = {'data': epoched_data, 'label': label}
                if not os.path.exists(os.path.join(cfg.dataset_root, cfg.data_to, '%d' %(nsub))):
                    os.makedirs(os.path.join(cfg.dataset_root, cfg.data_to, '%d' %(nsub)))
                savemat(os.path.join(cfg.dataset_root, cfg.data_to, '%d/%s state.mat' %(nsub, s)), mat)

            return [epoched_data, label]


    def get_train_val_set(self):
        
        train_val_data, train_val_label = [], []
        
        for s in cfg.state:
            for i in range(2, cfg.subj_num):
                train_val_data.append(self.preprocess(i+1, s)[0])
                train_val_label.append(self.preprocess(i+1, s)[1])

        train_val_data = np.expand_dims(np.concatenate(train_val_data), axis=1)
        train_val_label = np.concatenate(train_val_label)

        sample_num = len(train_val_label)
        shuffle = np.random.permutation(sample_num)
        train_val_data = train_val_data[shuffle, ...]
        train_val_label = train_val_label[shuffle]

        spliter = sample_num // 4
        valid_data, train_data = train_val_data[: spliter, ...], train_val_data[spliter: , ...]
        valid_label, train_label = train_val_label[: spliter], train_val_label[spliter:]

        self.train_data = self.standardize(train_data)
        train_label = torch.from_numpy(train_label)
        train_label = torch.stack((1-train_label, train_label), dim=1)
        self.train_label = train_label
        
        valid_label = torch.from_numpy(valid_label)
        valid_label = torch.stack((1-valid_label, valid_label), dim=1)

        train_dataset = TensorDataset(
            torch.from_numpy(self.train_data), 
            self.train_label
            )

        valid_dataset = TensorDataset(
            torch.from_numpy(self.standardize(valid_data)), 
            valid_label
            )
                
        return train_dataset, valid_dataset
    

    def get_test_set(self, nsub):

        test_data = [self.preprocess(nsub, s)[0] for s in cfg.state]
        test_data = np.expand_dims(np.concatenate(test_data), axis=1)

        test_label = [self.preprocess(nsub, s)[1] for s in cfg.state]
        test_label = np.concatenate(test_label)
        test_label = torch.from_numpy(test_label)
        test_label = torch.stack((1-test_label, test_label), dim=1)

        test_dataset = TensorDataset(
            torch.from_numpy(self.standardize(test_data)), 
            test_label
            )

        return test_dataset


    def interaug(self):  
        '''
        Segmentation and Reconstruction (S&R) data augmentation

        return: augmented np.array of shape (int(batch_size / 2), 1, 36, 1000), adn its label
        '''
        batch_size = cfg.batch_size

        aug_data = []
        aug_label = []
        for cls4aug in range(2):
            cls_idx = np.where(self.train_label[:, 1] == cls4aug)[0]
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
    
