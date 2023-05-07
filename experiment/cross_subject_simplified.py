import os
import sys
sys.path.append(os.getcwd())
gpus = [2]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

from config_simplified import Config
cfg = Config()
import numpy as np
import random
import time
import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

from model.conformer import Conformer
from model.eegnet import EEGNet
from model.hier_xfmr import HierXFMR
from process.dataset_simplified import DD_Dataset
from utils import confusion_matplotter, torch_summarize, cam_ploter
from einops import repeat
from sklearn import manifold


class ExP():
    def __init__(self, models):
        super(ExP, self).__init__()
        
        self.models = models 
        self.dataset = DD_Dataset() 
        
        
    def training(self):
        
        train_dataset, test_dataset = self.dataset.get_dataset()

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=500)
        num_train_iter = len(self.train_loader)
        
        for k, model in enumerate(self.models):
            name = model.name
            print('training %s ...\n' %name)
            log_np = list()
            # model.load_state_dict(torch.load(os.path.join(cfg.model_dir, '%s.pth' % name), map_location=cfg.device))
            model = nn.DataParallel(model.cuda(), device_ids=[i for i in range(len(gpus))]).cuda()
            curr_lr = cfg.lr
            self.optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr, betas=(cfg.b1, cfg.b2))
            # self.optimizer.load_state_dict(torch.load(os.path.join(cfg.model_dir, '%s_opt.pth' % name), map_location=cfg.device))
                
            for e in range(cfg.n_epochs):
                ## TRAIN
                model.train()
                # if (e + 1) % cfg.milestone == 0: curr_lr *= 0.8
                # self.optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr, betas=(cfg.b1, cfg.b2))
            
                train_acc, train_loss = 0, 0
                for (img, label) in self.train_loader:
                    
                    matrix, loss = self.evaluate_iter(img, label, model)  # , aug=True)
                                       
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    self.optimizer.step()
                    train_loss += loss.detach().cpu().numpy()

                    [[tn0, fp0], [fn0, tp0]] = matrix
                    train_acc += (tp0 + tn0) / (tp0 + fp0 + tn0 + fn0)
                    
                train_loss /= num_train_iter
                train_acc /= num_train_iter

                # msg = 'Epoch: %4d' %(e+1) +\
                #         '  train: %.3f,' % (train_loss) +' [%2d, %2d, %2d, %2d], %.3f' % (tn0, fp0, fn0, tp0, train_acc) +\
                # print(msg)

                # TEST
                if (e + 1) % 1 == 0: 
                    model.eval()
                    test_acc, test_loss = 0, 0
                    for (img, label) in self.test_loader:

                        matrix, loss = self.evaluate_iter(img, label, model)
                        
                        [[tn2, fp2], [fn2, tp2]] = matrix
                        test_acc += (tp2 + tn2) / (tp2 + fp2 + tn2 + fn2)
                        test_loss += loss.detach().cpu().numpy()

                    msg = 'Epoch: %4d' %(e+1) +\
                            '  train: %.3f,' % (train_loss) +' [%2d, %2d, %2d, %2d], %.3f' % (tn0, fp0, fn0, tp0, train_acc) +\
                            '  test: %.3f,' % (test_loss) +' [%3d, %3d, %3d, %3d], %.3f' % (tn2, fp2, fn2, tp2, test_acc)      
                    print(msg)

                    log_np.append([[train_loss, test_loss], [train_acc, test_acc]])

            torch.save(model.module.state_dict(), os.path.join(cfg.model_dir, '%s.pth' % name))
            torch.save(self.optimizer.state_dict(), os.path.join(cfg.model_dir, '%s_opt.pth' % name))
            training_curve = np.array(log_np)
            np.save('%s.npy'%name, training_curve)

            loss_curve, acc_curve = training_curve[:, 0], training_curve[:, 1]
            plt.style.use('seaborn-pastel')
            x = np.linspace(1, 500, 500)
            fig, axes = plt.subplots(1, 2,  figsize=(21, 9), layout='constrained')
            ax1 = axes[0]
            ax1.plot(x, loss_curve[:, 0], label='train', linewidth=2, 
                        color=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
            ax1.set_ylabel('training loss')
            ax1.set_xlabel('epoch')
            ax1.set_title("LOSS")
            ax2 = ax1.twinx()  # this is the important function
            line2 = ax2.plot(x, loss_curve[:, 1], label='test', linewidth=2, 
                        color=list(plt.rcParams['axes.prop_cycle'])[1]['color'])
            ax2.set_ylabel('testing loss')

            ax1 = axes[1]
            ax1.plot(x, acc_curve[:, 0], label='train', linewidth=2, 
                        color=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
            ax1.set_ylabel('training accuracy')
            ax1.set_title("ACCURACY")
            ax1.legend(loc=2, frameon=False)
            ax2 = ax1.twinx()  # this is the important function
            ax2.plot(x, acc_curve[:, 1], label='test', linewidth=2, 
                        color=list(plt.rcParams['axes.prop_cycle'])[1]['color'])
            ax2.set_ylabel('testing accuracy')
            ax2.set_xlabel('epoch')
            ax2.legend(loc=4, frameon=False)

            fig.suptitle('%s'%name)

            plt.savefig('%s_check.png'%name)
            plt.close()


            img = Variable(img.cuda().type(FloatTensor))
            _, outputs, score = model(img)

            plt.style.use('seaborn-pastel')
            colors = [list(plt.rcParams['axes.prop_cycle'])[2]['color'], list(plt.rcParams['axes.prop_cycle'])[4]['color']]
            cmap = LinearSegmentedColormap.from_list('my_attn', colors, N=255)

            plt.imshow(model.module.lowlevel.position_embedding.position_embedding[0].squeeze().unsqueeze(0).expand(30, -1).cpu().detach().numpy(), cmap=cmap)
            plt.savefig("position_embedding_low.png")
            plt.close()
            

            plt.style.use('seaborn-pastel')
            colors = [list(plt.rcParams['axes.prop_cycle'])[0]['color'], list(plt.rcParams['axes.prop_cycle'])[1]['color']]
            cmap = LinearSegmentedColormap.from_list('my_attn', colors, N=255)
            channel_weight = np.array([score[i].mean(dim=0)[0, 1:].cpu().detach().numpy() for i in range(1, 6)])
            np.save('channel_weight_hierXFMR.npy', channel_weight)
            fig = plt.figure(figsize=(15, 3))
            plt.imshow(channel_weight, cmap=cmap, alpha=0.9)
            plt.xticks(list(range(30)), ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8','T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'Oz','O2'])
            plt.yticks(list(range(5)), list(range(1, 6)))
            plt.xlabel('channels')
            plt.ylabel('layers')

            fig.savefig("attn_high.png", dpi=600)
            plt.close()

            data = outputs.detach().cpu().numpy()
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

                

    def evaluate_iter(self, img, label, model, aug=False, nsub=None):
        img = Variable(img.cuda().type(FloatTensor))
        label = Variable(label.cuda().type(FloatTensor))

        # llt, _ = model(img)
        # y_pred = torch.max(llt, 1)[1].cpu().numpy().astype(int)
        # label_llt = repeat(label, 'h c -> (h r) c', r=cfg.seq_len_hlt)
        # matrix = confusion_matrix(label_llt[:, 1].cpu().numpy(), y_pred)
        # loss_llt = cfg.criterion_cls(llt, label_llt)
        
        _, outputs, score = model(img)
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

    

    models = [HierXFMR()]  # , Conformer(), HierXFMR()
    exp = ExP(models)

    exp.training()

    for i in range(2):       
        print('Subject %d' % (i+1))
        cms = exp.comparation(i+1)
 


if __name__ == "__main__":
    main()
    # print(time.asctime(time.localtime(time.time())))
    # starttime = datetime.datetime.now()
    # endtime = datetime.datetime.now()
    # add noise