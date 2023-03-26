"""
Class activation topography (CAT) for EEG model visualization, combining class activity map and topography
Code: Class activation map (CAM) and then CAT

refer to high-star repo on github: 
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/grad_cam

"""
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import scipy.io

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

import matplotlib.pyplot as plt

from einops import rearrange, reduce, repeat
from torch.backends import cudnn
import mne
from matplotlib import mlab as mlab
from utils import GradCAM, show_cam_on_image

from model.model import Conformer
from experiment.config import Config

cfg = Config()

cudnn.benchmark = False
cudnn.deterministic = True

# ! A crucial step for adaptation on Transformer
# reshape_transform  b 61 40 -> b 40 1 61
def reshape_transform(tensor):
    result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return result


preprocessing = ['raw', 'proj']
device = torch.device("cpu")
model = Conformer()
# set the class (class activation mapping)
target_category = 2

montage = mne.channels.make_standard_montage('standard_1020')
index = [0, 2, 15, 17, 19, 21, 23, 26, 28, 30, 32, 34, 86, 39, 41, 43, 88, 48, 50, 52, 54, 56, 92, 87, 61, 63, 65, 89, 93, 80, 81, 82, 25, 35, 73, 75]
montage.ch_names = [montage.ch_names[i] for i in index]
montage.dig = [montage.dig[i+3] for i in index]
info = mne.create_info(ch_names=montage.ch_names, sfreq=250., ch_types='eeg')


for nSub in range(12):
    for p in preprocessing:
        # # used for cnn model without transformer
        # model.load_state_dict(torch.load('./model/model_cnn.pth', map_location=device))
        # target_layers = [model[0].projection]  # set the layer you want to visualize, you can use torchsummary here to find the layer index
        # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        model.load_state_dict(torch.load(os.path.join(cfg.model_dir, 'sub%d_%s.pth'%(nSub+1, p), map_location=device)))
        target_layers = [model[0].shallownet, model[1]]  # set the target layer 
        cam1 = GradCAM(model=model, target_layers=target_layers[0], use_cuda=False)
        cam2 = GradCAM(model=model, target_layers=target_layers[1], use_cuda=False, reshape_transform=reshape_transform)

        data = []
        for s in cfg.state:
            load = scipy.io.loadmat(os.path.join(cfg.dataset_root, 'preprocessed_%s'%p, '%d/%s state.mat' %(nSub+1, s)))
            data.append(load['data'])
            # load['label'][0].transpose()
        data = np.expand_dims(np.concatenate(data), axis=1)

        
        # TODO: Class Activation Topography (proposed in the paper)
        all_cam = []
        # this loop is used to obtain the cam of each trial/sample
        for i in range(200):
            test = torch.as_tensor(data[i:i+1, :, :, :], dtype=torch.float32)
            test = torch.autograd.Variable(test, requires_grad=True)

            grayscale_cam = cam1(input_tensor=test)
            grayscale_cam = grayscale_cam[0, :]
            all_cam.append(grayscale_cam)

        test_all_data = np.squeeze(np.mean(data, axis=0))
        mean_all_test = np.mean(test_all_data, axis=1)
        test_all_cam = np.mean(all_cam, axis=0)
        mean_all_cam = np.mean(test_all_cam, axis=1)
        hyb_all = test_all_data * test_all_cam
        mean_hyb_all = np.mean(hyb_all, axis=1)

        evoked = mne.EvokedArray(test_all_data, info)
        evoked.set_montage(montage)

        fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5)
        plt.subplot(151)
        im1, cn1 = mne.viz.plot_topomap(mean_all_test, evoked.info, show=False, axes=ax1, res=1200)
        plt.subplot(152)
        im1, cn1 = mne.viz.plot_topomap(mean_all_cam, evoked.info, show=False, axes=ax2, res=1200)
        plt.subplot(153)
        im1, cn1 = mne.viz.plot_topomap(mean_hyb_all, evoked.info, show=False, axes=ax3, res=1200)

        all_cam = []
        # this loop is used to obtain the cam of each trial/sample
        for i in range(200):
            test = torch.as_tensor(data[i:i+1, :, :, :], dtype=torch.float32)
            test = torch.autograd.Variable(test, requires_grad=True)

            grayscale_cam = cam2(input_tensor=test)
            grayscale_cam = grayscale_cam[0, :]
            all_cam.append(grayscale_cam)

        test_all_cam = np.mean(all_cam, axis=0)
        mean_all_cam = np.mean(test_all_cam, axis=1)
        hyb_all = test_all_data * hyb_all
        mean_hyb_all = np.mean(hyb_all, axis=1)

        plt.subplot(154)
        im1, cn1 = mne.viz.plot_topomap(mean_all_cam, evoked.info, show=False, axes=ax4, res=1200)
        plt.subplot(155)
        im1, cn1 = mne.viz.plot_topomap(mean_hyb_all, evoked.info, show=False, axes=ax5, res=1200)

        plt.savefig('%d_%s.jpg'%(nSub+1, p))
        plt.close()



