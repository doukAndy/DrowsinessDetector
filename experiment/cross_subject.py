import os
import sys
sys.path.append(os.getcwd())

gpus = [0]
os.environ['_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import random
import time
import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.cuda import FloatTensor, LongTensor
from torch.nn.modules.module import _addindent
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

from config import Config
from model.conformer import Conformer
from model.eegnet import EEGNet
from model.hier_xfmr import HierXFMR
from process.dataset import EEGdata
from utils import confusion_matplotter, torch_summarize, cam_ploter



cfg = Config()


class ExP():
    def __init__(self, models):
        super(ExP, self).__init__()
        
        self.models = models 
        self.dataset = EEGdata() 
        self.report_writer = open(os.path.join(cfg.result_dir, 'report.txt'), "w")

        
    def train(self):
        
        train_dataset, valid_dataset = self.dataset.get_train_val_set()

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=500, shuffle=True)
        num_train_iter = len(self.train_loader)
        num_valid_iter = len(self.valid_loader)

        for k, model in enumerate(self.models):
            name = model.name
            print('training %s ...\n' %name)
            log_writer = open(os.path.join(cfg.log_dir, '%s.txt' % name), "w")
            model = nn.DataParallel(model.cuda(), device_ids=[i for i in range(len(gpus))]).()
            curr_lr = cfg.lr
            max_valid_acc = 0
            max_acc_e = 0
            for e in range(cfg.n_epochs):
                ## TRAIN
                model.train()

                if (e + 1) % cfg.milestone == 0: curr_lr *= 0.1
                self.optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr, betas=(cfg.b1, cfg.b2))

                train_acc, train_loss = 0, 0
                for (img, label) in self.train_loader:
                    
                    matrix, loss = self.evaluate_iter(img, label, model, aug=True)
                                       
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.detach().cpu().numpy()

                    [[tn, fp], [fn, tp]] = matrix
                    train_acc += tp / (tp + fp + 1e-10)
                    
                train_loss /= num_train_iter
                train_acc /= num_train_iter

                ## VALID
                model.eval()

                valid_acc, valid_loss = 0, 0
                for (img, label) in self.valid_loader:

                    matrix, loss = self.evaluate_iter(img, label, model)
                    
                    [[tn, fp], [fn, tp]] = matrix
                    valid_acc += tp / (tp + fp)
                    valid_loss += loss

                valid_loss /= num_valid_iter
                valid_acc /= num_valid_iter

                msg = 'Epoch: %4d' %(e+1) +\
                        '  train loss: %.3f' % (train_loss) +\
                        '  valid loss: %.3f' % (valid_loss) +\
                        '  train accuracy: %.3f' % (train_acc) +\
                        '  valid accuracy: %.3f' % (valid_acc)
        
                print(msg)
                log_writer.write(msg + '\n')
                
                if valid_acc > max_valid_acc:
                    max_valid_acc = valid_acc
                    max_acc_e = e
                    torch.save(model.module.state_dict(), os.path.join(cfg.model_dir, '%s.pth' % name))
            log_writer.write('\n** max_valid_acc = %.3f when epoch = %d' %(max_valid_acc, max_acc_e))
            self.report_writer.write('\n** %s: \n\tmax_valid_acc = %.3f when epoch = %d' %(name, max_valid_acc, max_acc_e))
        log_writer.close()
        

    def test(self, nsub, visualization=True):
        test_dataset = self.dataset.get_test_set(nsub)
        test_loader = DataLoader(dataset=test_dataset, batch_size=200, shuffle=True)

        test_loss = 0
        confusion = np.zeros((3, 2, 2))
        subtitles = list()
        for k, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(cfg.model_dir, '%s.pth' % model.name), map_location=cfg.device))
            model = model.cuda()
            self.report_writer = open(os.path.join(cfg.result_dir, 'report.txt'), "w")
            self.report_writer.write('\n' + torch_summarize(model))
            for (img, label) in test_loader:
                matrix, loss = self.evaluate_iter(img, label, model, nsub=nsub)
                confusion[k] += matrix
                test_loss += loss
                
            test_loss /= len(test_loader)
            confusion[k] /= len(test_loader)
            subtitles.append(model.name)
            
            print('\t%s: ' % model.name + str(confusion[k]))
            self.report_writer.write('\t%s: ' % model.name + str(confusion[k]))
        confusion_matplotter(confusion, subtitles, flag=nsub)
        self.report_writer.close()
        return confusion


    def evaluate_iter(self, img, label, model, aug=False, nsub=None):
        img = Variable(img.cuda().type(FloatTensor))
        # label = Variable(label.().type(LongTensor))
        # label = torch.stack((1-label, label), dim=1)
        label = Variable(label.cuda().type(FloatTensor))
        if nsub: cam_ploter(img, model, nsub)
        if aug == True:
            aug_data, aug_label = self.dataset.interaug()
            img = torch.cat((img, aug_data))
            label = torch.cat((label, aug_label))
        
        outputs = model(img)
        y_pred = torch.max(outputs, 1)[1].cpu().numpy().astype(int)
        matrix = confusion_matrix(label[:, 1].cpu().numpy(), y_pred)
        loss = cfg.criterion_cls(outputs, label)

        return matrix, loss



def main():

    seed_n = 2023
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    

    models = [EEGNet(), Conformer(), HierXFMR()]  # 
    exp = ExP(models)

    # exp.train()

    for i in range(2):       
        print('Subject %d' % (i+1))
        confusion = exp.test(i+1)
 


if __name__ == "__main__":
    main()
    # print(time.asctime(time.localtime(time.time())))
    # starttime = datetime.datetime.now()
    # endtime = datetime.datetime.now()
    # add noise