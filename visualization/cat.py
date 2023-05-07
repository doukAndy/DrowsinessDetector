import os
import sys
sys.path.append(os.getcwd())
gpus = [2]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

import scipy.io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from einops import rearrange, reduce, repeat
from torch.backends import cudnn
import mne
from matplotlib import mlab as mlab
from utils import GradCAM, show_cam_on_image

from model.eegnet import EEGNet
from model.hier_xfmr import HierXFMR
from experiment.config_simplified import Config

cfg = Config()

cudnn.benchmark = False
cudnn.deterministic = True


def reshape_transform(tensor):
    result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return result

# montage.plot(show_names=True, show=False)  # topo layout

test_data = np.load('test_data.npy')
test_data_sample = np.squeeze(np.mean(test_data, axis=0))
test_data_channel = np.mean(test_data_sample, axis=1)

# model = HierXFMR()
# model.load_state_dict(torch.load('HierXFMR.pth', map_location=torch.device("cpu")))
# target_layers = [model.highlevel.transformer.layers[0]]  
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform) 

model = EEGNet()
model.load_state_dict(torch.load('EEGNet.pth', map_location=torch.device("cpu")))  # cfg.device
target_layers = [model.blocks[1]]  # , model.blocks[1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

test_data = torch.as_tensor(test_data, dtype=torch.float32)
test_data = torch.autograd.Variable(test_data, requires_grad=True)

# test_data = torch.as_tensor(test_data)
# test_data = torch.autograd.Variable(test_data.cuda().type(torch.cuda.FloatTensor))
# _, _, weight = model(test_data)
# # weight = np.load('channel_weight_hierXFMR.npy')
# w_mean = 0
# plt.style.use('seaborn-pastel')
# colors = [list(plt.rcParams['axes.prop_cycle'])[0]['color'], list(plt.rcParams['axes.prop_cycle'])[1]['color']]
# cmap = LinearSegmentedColormap.from_list('my_attn', colors, N=255)
# for i in range(6):
#     weight[i] = weight[i].mean(axis=0)
#     weight[i] = (weight[i] - weight[i].min()) /(weight[i].max() - weight[i].min())
#     w_mean += weight[i]
#     plt.imshow(weight[i].detach().numpy(), cmap=cmap)
#     plt.savefig('ts_attn1_%d.png' %i)
#     plt.close

# test_cam_sample = weight.mean(axis=0)
# mean_hyb_all = test_data_channel * test_cam_sample
# mean_hyb_all = (mean_hyb_all - mean_hyb_all.min()) /(mean_hyb_all.max() - mean_hyb_all.min())

grayscale_cam = cam(input_tensor=test_data)
test_cam_sample = np.mean(grayscale_cam, axis=0)
test_cam = np.mean(test_cam_sample, axis=1)
test_cam = (test_cam -test_cam.min()) /(test_cam.max()-test_cam.min())

hyb_all = test_data_sample * test_cam_sample
mean_hyb_all = np.mean(hyb_all, axis=1)
mean_hyb_all = (mean_hyb_all - mean_hyb_all.min()) /(mean_hyb_all.max() - mean_hyb_all.min())

montage = mne.channels.make_standard_montage('standard_1020')  # , head_size=0.095)
montage.ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8','T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz','O2']
index = [0, 1, 2, 3, 5, 18, 20, 22, 24, 26, 29, 31, 33, 35, 37, 89, 42, 44, 46, 91, 51, 53, 55, 57, 59, 90, 64, 66, 68, 92, 83, 84, 85]
montage.dig = [montage.dig[i] for i in index]
for i in range(3):
    montage.dig[i]['r'][1] += 0.02
info = mne.create_info(ch_names=montage.ch_names, sfreq=128., ch_types='eeg')
info.set_montage(montage)

plt.style.use('seaborn-pastel')
colors = [list(plt.rcParams['axes.prop_cycle'])[2]['color'], list(plt.rcParams['axes.prop_cycle'])[4]['color']]
cmap = LinearSegmentedColormap.from_list('my_attn', colors, N=255)
mne.viz.plot_topomap(mean_hyb_all, info, show=False, res=1200, size=5, cmap=cmap)
plt.savefig('topo_en_weight.png')
plt.close()
