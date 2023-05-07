import os
import torch
import torch.nn as nn

class Config:
    def __init__(self):
        # set path:
        self.result_dir = './result/simplified'
        self.log_dir = './result/simplified/log'
        self.model_dir = './model/simplified'
        self.dataset_root = './data'
        
        if not os.path.exists(self.result_dir): os.mkdir(self.result_dir)
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        
        # data discription:
        self.state = ['Normal', 'Fatigue']
        self.samples = 384              
        self.epoch_duration = 3          
        self.subj_num = 11
        self.fs = 128
        self.channels = 30
        self.n_classes = 2
        
        # training:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.batch_size = 64
        
        # self.start_epoch = 0
        self.n_epochs = 500
        
        self.lr = 0.00001
        self.milestone = 500
        
        self.b1 = 0.5
        self.b2 = 0.999

        self.criterion_l1 = nn.L1Loss().cuda()
        self.criterion_l2 = nn.MSELoss().cuda()
        self.criterion_cls = nn.BCEWithLogitsLoss().cuda()  # nn.CrossEntropyLoss(label_smoothing=0.3).cuda()  #   # 

        # configuration of EEGNet
        self.dropout_e = 0.5
        self.kernel_len1 = int(0.5 * self.fs)
        self.kernel_len2 = 16
        self.pool_1 = 4
        self.pool_2 = 8
        self.F1 = 8
        self.F2 = 16
        self.D = 2

        # configuration of Conformer
        self.dropout_c = 0.5
        self.emb_size = 48
        self.depth = 3
        self.kernel_len = 64
        self.n_kernels = 8
        self.pool_len = 64
        self.pool_stride = 64
        self.num_heads = 2
        self.forward_expansion = 2
        self.num_tokens = 36

        # configuration of Hierarchical Transformer
        self.seq_len_hlt = 1  # 47
        self.patch_len = 1
        self.num_layers = 6
        self.sliding_window = 384  # 16
        self.d_model = 30
        self.nhead = 3
        self.dropout = 0.5
        self.norm = None
        self.mode1 = 'global'
        self.mode2 = 'class'
        self.train_mode = 'llt'
        self.how_train = 'fixed'
        self.use_HT = True
        self.eval_subject = 1

        # # configuration of EEG-Inception
        # self.scales_samples = self.fs * [0.5, 0.25, 0.125]
        # self.filter_per_branch = 8
        # self.dropout_rate = 0.25

        # compared to the saved one, I changed the batchsize fome 64 to 32, sliding window from 32 to 16
        # further: add milestone back (not imply yet)