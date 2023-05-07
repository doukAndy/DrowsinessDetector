from scipy.io import loadmat, savemat
import os
import numpy as np
import torch
from experiment.config import Config
cfg = Config()


'''
load.keys():
    dict_keys(['__header__', '__version__', '__globals__', 'EEGsample', 'subindex', 'substate'])

load['EEGsample'].shape: nparray(2022, 30, 384)
load['subindex'].shape:  nparray(2022, 1) int from 1 to 11
load['substate'].shape:  nparray(2022, 1) 0 or 1

np.where(load['subindex'] == 1)[0].shape (188,)
np.where(load['subindex'] == 2)[0].shape (132,)
np.where(load['subindex'] == 3)[0].shape (150,)
np.where(load['subindex'] == 4)[0].shape (148,)
np.where(load['subindex'] == 5)[0].shape (224,)
np.where(load['subindex'] == 6)[0].shape (188,)
np.where(load['subindex'] == 7)[0].shape (102,)
np.where(load['subindex'] == 8)[0].shape (264,)
np.where(load['subindex'] == 9)[0].shape (314,)
np.where(load['subindex'] == 10)[0].shape (108,)
np.where(load['subindex'] == 10)[0].shape (226,)

num_subj = np.unique(load['subindex']).shape[0]  # 11

'''
load = loadmat(os.path.join(cfg.dataset_root, 'dataset.mat'))

sample = torch.from_numpy(load['EEGsample'])
sample_len = sample.shape[0]

label = torch.from_numpy(load['substate'].transpose()[0])
label = torch.stack((1-label, label), dim=1)

test_len = np.where(load['subindex'] < 3)[0].shape[0]
test_data = sample[:test_len]  # (320, 30, 384)
test_label = label[:test_len]

shuffle = np.random.permutation(sample_len - test_len)
sample = sample[test_len:][shuffle]
label = label[test_len:][shuffle]
spliter = (sample_len - test_len) // 8
valid_data, train_data = sample[:spliter], sample[spliter:]
valid_label, train_label = label[:spliter], label[spliter:]
