import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.nn.modules.module import _addindent
import numpy as np
import matplotlib.pyplot as plt
import itertools
from einops import rearrange
from pytorch_grad_cam import GradCAM
import torchcam
from torchcam.utils import overlay_mask
from torch.utils.data import DataLoader

from config import Config
cfg = Config()

import os
from model.eegnet import EEGNet
from model.conformer import Conformer
from model.hier_xfmr import HierXFMR
from process.dataset import EEGdata
from torch.cuda import FloatTensor, LongTensor
from torch.autograd import Variable


def torch_summarize(model, show_weights=True, show_parameters=True):
        """Summarizes torch model by showing trainable parameters and weights."""
        tmpstr = model.__class__.__name__ + ' (\n'
        for key, module in model._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = torch_summarize(module)
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr +=  ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        return tmpstr


def confusion_matplotter(cms, subtitles, flag=1, cmap=plt.cm.Blues):
    classes = cfg.state
    tick_marks = np.arange(len(classes))
    num_subplot = len(subtitles)
    fig, axes = plt.subplots(1, num_subplot, figsize=(9, 3), layout='constrained')
        
    for k in range(num_subplot):
        im = axes[k].imshow(cms[k], interpolation='nearest', cmap=cmap)
        axes[k].set_title(subtitles[k], fontsize=10)
        axes[k].tick_params(labelsize=7) 
        axes[k].set_xticks(tick_marks, classes, rotation=90) 
        axes[k].set_yticks(tick_marks, classes if k==0 else [])
        
        threshold = cms[k].max() / 2.
        for i, j in itertools.product(range(cms[k].shape[0]), range(cms[k].shape[1])):
            axes[k].text(j, i, cms[k][i, j],
                    horizontalalignment="center",
                    color="white" if cms[k][i, j] > threshold else "black",
                    fontsize=12)
    
    fig.suptitle('confusion matrix', fontsize=12)
    fig.supxlabel('PREDICT', fontsize=10, c='r')
    fig.supylabel('GROUND TRUTH', fontsize=10, c='r')
    fig.colorbar(im, ax=axes, shrink=0.5)
    plt.savefig('./result/figures/confusion matrix_%d.pdf' %flag, dpi=300)  # pdf
    plt.close()



def reshape_transform(tensor):
    result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return result


def cam_ploter(img, model, nsub=3):
    
    if model.name == 'EEGNet':
        target_layers_bag = [[model.blocks[0]], [model.blocks[1]], [model.blocks[0], model.blocks[1]]]
    elif model.name == 'Conformer':
        target_layers_bag = [[model[0].shallownet], [model[1][0]], [model[1][1]], [model[1][2]]]  # , [model[1][3]], [model[1][4]], [model[1][5]]]  # set the target layer   
    else: target_layers_bag = [[model.lowlevel.transformer], [model.highlevel.transformer]]  

    len_bag = len(target_layers_bag)
    plt.figure()
    for i, target_layers in enumerate(target_layers_bag):
        if model.name == 'EEGNet':
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)  
        elif model.name == 'Conformer':
            if i == 0:
                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)           
            else: cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
        else: cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
        cam_map = cam(input_tensor=img, targets=None, aug_smooth=True, eigen_smooth=True)[0]
        
        plt.subplot(len_bag, 1, i+1)
        plt.imshow(cam_map)
    plt.savefig('./result/figures/%s_%d.png' %(model.name, nsub))
    plt.close()
    



if __name__ == "__main__":
    model = HierXFMR()  # Conformer()  # EEGNet()  # HierXFMR()
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, '%s.pth' % model.name), map_location=cfg.device))
    model = model.cuda()
    
    cam_ploter(model)