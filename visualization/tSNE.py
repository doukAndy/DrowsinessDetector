# use t-SNE to show the feature distribution

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from einops import reduce
import scipy.io
import torch
from torch.autograd import Variable
from torch import FloatTensor, LongTensor
from model.eegnet import EEGNet
from process.dataset_simplified import DD_Dataset
from torch.utils.data import DataLoader
from experiment.config_simplified import Config
cfg = Config()

def plt_tsne(data, label, per):
    
    data = data.detach().cpu().numpy()
    # data = reduce(data, 'b n e -> b e', reduction='mean')
    label = label.numpy()[:, 0]

    tsne = manifold.TSNE(n_components=2, perplexity=2, init='pca', random_state=166)
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    plt.style.use('seaborn-pastel')
    colors = [list(plt.rcParams['axes.prop_cycle'])[0]['color'], list(plt.rcParams['axes.prop_cycle'])[1]['color']]
    color = [colors[label[i]] for i in range(len(label))]
    plt.figure()
    plt.scatter(X_norm[:, 0], X_norm[:, 1], s=120, color=color) 
    plt.xticks([])
    plt.yticks([])
    plt.savefig('test.png', dpi=600)
    plt.close()


if __name__ == "__main__":  

    model = EEGNet()
    name = model.name
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, '%s.pth' % name), map_location=torch.device("cpu")))
    model.eval()
    dataset = DD_Dataset()
    _, test_dataset = dataset.get_dataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=500)
    for (img, label) in test_loader:
        img = img.type(FloatTensor)
        outputs = model(img)  # torch.Size([320, 1, 30, 384]) --> torch.Size([320, 2])
        rawdata = reduce(img, 'b c n e -> (b c) n', reduction='mean')  # torch.Size([320, 30])
    plt_tsne(outputs, label, outputs.shape[-1])
