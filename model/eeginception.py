import torch
from torch.nn import init
from torch.nn import Module, Sequential
from torch.nn import Conv2d, AvgPool2d 
from torch.nn import BatchNorm2d
from torch.nn import Linear, ELU, Softmax, Dropout
import numpy as np
from experiment.config import Config
cfg = Config()


class Conv2dWithConstraint(Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs) -> None:
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class Inception_1(Sequential):
    def __init__(self):
        super().__init__()
        self.sub_blocks = list()
        for i in range(len(cfg.scales_samples)):
            block = Sequential(
                Conv2d(1, cfg.filter_per_branch, (1, cfg.scales_samples[i]), stride=1, padding=(0, cfg.scales_samples[i]//2), bias=False),
                BatchNorm2d(cfg.filter_per_branch, momentum=0.01, affine=True, eps=1e-3),
                ELU(),
                Dropout(p=cfg.dropout_rate),

                Conv2dWithConstraint(cfg.filter_per_branch, cfg.filter_per_branch * 2, (cfg.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                    groups=cfg.filter_per_branch, bias=False),
                BatchNorm2d(cfg.filter_per_branch * 2, momentum=0.01, affine=True, eps=1e-3),
                ELU(),
                Dropout(p=cfg.dropout_rate)
            )
            self.sub_blocks.append(block)

        self.avgpool = AvgPool2d((1, 4), stride=4)
        
    def multi_blocks(self, x):
        sub_xs = [self.sub_blocks[i](x) for i in range(len(cfg.scales_samples))]
        torch.cat(sub_xs, 1)
        return x

    def forward(self, x):
        x = self.multi_blocks(x)
        x = self.avgpool(x)
        return x


class Inception_2(Sequential):
    def __init__(self, scales_samples, filter_per_branch, dropout_rate, channels):
        super().__init__()
        cfg.scales_samples = scales_samples
        cfg.filter_per_branch = filter_per_branch
        cfg.dropout_rate = dropout_rate
        cfg.channels = channels

        self.sub_blocks = list()
        for i in range(len(cfg.scales_samples)):
            block = Sequential(
                Conv2d(1, cfg.filter_per_branch, (1, cfg.scales_samples[i]//4), stride=1, padding=(0, cfg.scales_samples[i]//8), bias=False),
                BatchNorm2d(cfg.filter_per_branch, momentum=0.01, affine=True, eps=1e-3),
                ELU(),
                Dropout(p=cfg.dropout_rate)
            )
            self.sub_blocks.append(block)

        self.avgpool = AvgPool2d((1, 2), stride=2)
        
    def multi_blocks(self, x):
        sub_xs = [self.sub_blocks[i](x) for i in range(len(cfg.scales_samples))]
        torch.cat(sub_xs, 1)
        return x

    def forward(self, x):
        x = self.multi_blocks(x)
        x = self.avgpool(x)
        return x

    
class Output(Sequential):
    def __init__(self):
        super().__init__()

        self.classhead = Sequential(
            Conv2d(1, cfg.filter_per_branch * len(cfg.scales_samples)//2, (1, 8), stride=1, padding=(0, 4), bias=False),
            BatchNorm2d(cfg.filter_per_branch * len(cfg.scales_samples)//2, momentum=0.01, affine=True, eps=1e-3),
            ELU(),
            AvgPool2d((1, 2), stride=2),
            Dropout(p=cfg.dropout_rate), 

            Conv2d(cfg.filter_per_branch * len(cfg.scales_samples)//2, cfg.filter_per_branch * len(cfg.scales_samples)//4, (1, 4), stride=1, padding=(0, 2), bias=False),
            BatchNorm2d(cfg.filter_per_branch * len(cfg.scales_samples)//4, momentum=0.01, affine=True, eps=1e-3),
            ELU(),
            AvgPool2d((1, 2), stride=2),
            Dropout(p=cfg.dropout_rate), 
        )
        self.fc = Sequential(
            Linear(cfg.filter_per_branch * len(cfg.scales_samples)//4, self.n_classes, bias=False),
            Softmax(dim=1)
        )

    
    def forward(self, x):
        x = self.classhead(x)
        x = x.contiguous().view(x.size(0), -1)  # x.view(x.size()[0], -1)
        x = self.fc(x)
        return x



class EEGInception(Sequential):
    def __init__(self, scales_samples, filter_per_branch, dropout_rate, channels):
        cfg.scales_samples = scales_samples
        cfg.filter_per_branch = filter_per_branch
        cfg.dropout_rate = dropout_rate
        cfg.channels = channels
        super().__init__(
            Inception_1(),
            Inception_2(),
            Output()
        )
