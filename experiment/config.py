import os
import torch.nn as nn

class Config:
    def __init__(self):
        
        self.batch_size = 64
        self.start_epoch = 0
        self.n_epochs = 50
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999

        self.criterion_l1 = nn.L1Loss().cuda()
        self.criterion_l2 = nn.MSELoss().cuda()
        self.cross_entropy = nn.CrossEntropyLoss().cuda()

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
            
        self.state = ['Normal', 'Fatigue']
        self.subj_num = len(os.listdir(os.path.join(self.dataset_root, self.data_from)))
 
        # preprocessing config:
        self.epoch_duration = 4  # downsample to 250Hz
        self.filt = True
        self.lfreq, self.hfreq = 4, 45
        self.proj = False
        self.visualize = False
        self.save_mat = True
        self.verbose = 'ERROR'
        
        
