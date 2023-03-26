import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch
from torch.cuda import FloatTensor, LongTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

from config import Config
from model.model import Conformer
from process.dataset import EEGdata


cfg = Config()

class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        
        self.nSub = nsub
        self.log_write = open(os.path.join(cfg.log_dir, "log_sub_%d.txt" % nsub), "w")
        self.dataset = EEGdata(nsub)

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

        
    def train(self):
        
        train_dataset, test_dataset = self.dataset.get_data()

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=200, shuffle=True)

        # Optimizers
        
        test_data = Variable(test_data.type(FloatTensor))
        test_label = Variable(test_label.type(LongTensor))

        
        curr_lr = cfg.lr
        bestAcc,  averAcc = 0, 0
        Y_true, Y_pred = 0, 0
        for e in range(cfg.n_epochs):

            if (e + 1) % 10 == 0: curr_lr *= 0.1
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=curr_lr, betas=(self.b1, self.b2))

            self.model.train()
            for i, (img, label) in enumerate(self.train_loader):

                img = Variable(img.cuda().type(FloatTensor))
                label = Variable(label.cuda().type(LongTensor))

                # data augmentation
                aug_data, aug_label = self.dataset.interaug()
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))


                tok, outputs = self.model(img)
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                loss = cfg.cross_entropy(outputs, label) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            self.model.eval()
            for (test_data, test_label) in self.test_loader:
                Tok, Cls = self.model(test_data)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))

                loss_test = cfg.cross_entropy(Cls, test_label)

                msg = 'Epoch: %4d' %e +\
                        '  Train loss: %.3f' % loss.detach().cpu().numpy() +\
                        '  Test loss: %.3f' % loss_test.detach().cpu().numpy() +\
                        '  Train accuracy %.3f' % train_acc +\
                        '  Test accuracy is %.3f' % acc
    
                print(msg)
                self.log_write.write(msg + '\n')

                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred
        averAcc = averAcc / cfg.n_epochs


        torch.save(self.model.module.state_dict(), os.path.join(cfg.model_dir, 'model_subj_%d' % self.nSub))
        
        print('avg acc:  ', averAcc)
        print('best acc: ', bestAcc)
        self.log_write.write('avg acc:  ' + str(averAcc) + "\n")
        self.log_write.write('best acc: ' + str(bestAcc) + "\n")
        self.log_write.close()

        return bestAcc, averAcc, Y_true, Y_pred
        

def main():

    seed_n = 2023
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    result_write = open(os.path.join(cfg.result_dir, 'result.txt'), "w")

    best, aver = 0, 0
    for i in range(cfg.subj_num):
        starttime = datetime.datetime.now()

        print('Subject %d' % (i+1))

        exp = ExP(i + 1)
        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' :   best acc %.3f' %(bestAcc) + '      avg acc %.3f' %(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / cfg.subj_num
    aver = aver / cfg.subj_num

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('**The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))