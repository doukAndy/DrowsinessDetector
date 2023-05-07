from experiment.config_simplified import Config
cfg = Config()
import os
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset


class DD_Dataset():
    def __init__(self) -> None:
        super().__init__()
        

    def standardize(self, data):
        mean = torch.mean(data, axis=-1, keepdims=True)
        std = torch.std(data, axis=-1, keepdims=True)
        data = (data - mean) / std
        # whether to add noise 
        return data
    
            
    def get_dataset(self):
        
        load = loadmat(os.path.join(cfg.dataset_root, 'dataset.mat'))
        sample = np.expand_dims(load['EEGsample'], axis=1)
        sample = torch.from_numpy(sample)
        sample_len = sample.shape[0]

        label = torch.from_numpy(load['substate'].transpose()[0])
        label = torch.stack((1-label, label), dim=1)

        test_len = np.where(load['subindex'] < 3)[0].shape[0]
        test_data = sample[:test_len]  # (320, 30, 384)
        test_label = label[:test_len]

        shuffle = np.random.permutation(sample_len - test_len)
        self.train_data = self.standardize(sample[test_len:][shuffle])
        self.train_label = label[test_len:][shuffle]
        
        train_dataset = TensorDataset(
            self.train_data, 
            self.train_label
            )

        test_dataset = TensorDataset(
            self.standardize(test_data), 
            test_label
            )
                
        return train_dataset, test_dataset


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

            tmp_aug_data = np.zeros((int(batch_size / 2), 1, cfg.channels, cfg.samples))
            for ri in range(int(batch_size / 2)):
                for rj in range(6):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 6)
                    tmp_aug_data[ri, :, :, rj * 64:(rj + 1) * 64] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 64:(rj + 1) * 64]

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
    
