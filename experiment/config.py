import os
import torch
import torch.nn as nn


class Config:
    def __init__(self):
        # set path:
        self.result_dir = './result'
        self.log_dir = './result/log'
        self.model_dir = './model/saved_models'
        self.dataset_root = './data'
        self.data_from = 'dataset_figshare'
        self.data_to = 'preprocessed_raw'
        
        if not os.path.exists(self.result_dir): os.mkdir(self.result_dir)
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        if not os.path.exists(os.path.join(self.dataset_root, self.data_from)): 
            os.mkdir(os.path.join(self.dataset_root, self.data_from))
            raise RuntimeError('HAVE NOT PUT RAW DATASET INTO RIGHT PLACE!!! Go to download raw dataset and put it into %s. '
                           'Then try to run again.' %(os.path.join(self.dataset_root, self.data_from)))
        if not os.path.exists(os.path.join(self.dataset_root, self.data_to)): 
            os.mkdir(os.path.join(self.dataset_root, self.data_to))

        # preprocessing configuration:
        self.samples = 1000              # samples per epoch
        self.epoch_duration = 4          # duration(4s) per epoch, ths downsampling to 250Hz
        self.get_time = list(range(25, 125))   # total: 600s, getting 600/epoch_duration = 150 epochs
        self.lfreq, self.hfreq = 4, 45
        self.filt = True
        self.proj = False
        self.visualize = True
        self.save_mat = True
        self.verbose = 'ERROR'

        # data discription:
        self.state = ['Normal', 'Fatigue']
        self.subj_num = len(os.listdir(os.path.join(self.dataset_root, self.data_from)))
        self.fs = 250  # self.samples/self.epoch_duration
        self.channels = 36
        self.n_classes = 2
        
        # training:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.batch_size = 32
        
        self.start_epoch = 0
        self.n_epochs = 100
        
        self.lr = 0.0002
        self.milestone = 10
        
        self.b1 = 0.5
        self.b2 = 0.999

        self.criterion_l1 = nn.L1Loss().cuda()
        self.criterion_l2 = nn.MSELoss().cuda()
        self.criterion_cls = nn.BCEWithLogitsLoss().cuda()  # nn.CrossEntropyLoss(label_smoothing=0.1).cuda()  # nn.BCEWithLogitsLoss().cuda()

        # configuration of EEGNet
        self.dropout_e = 0.5
        self.kernel_len1 = int(0.5 * self.fs)
        self.kernel_len2 = 25
        self.F1 = 8
        self.F2 = 16
        self.D = 2
        self.pool_1 = 5
        self.pool_2 = 25

        # configuration of Conformer
        self.dropout_c = 0.5
        self.emb_size = 16
        self.depth = 3
        self.kernel_len = 25
        self.n_kernels = 8
        self.pool_len = 25
        self.pool_stride = 25
        self.num_heads = 2
        self.forward_expansion = 2
        self.num_tokens = 61

        # configuration of Hierarchical Transformer
        self.seq_len_hlt = 40
        self.patch_len = 1
        self.num_layers = 2
        self.sliding_window = 25
        self.d_model = 36
        self.nhead = 2
        self.dropout = 0.1
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
