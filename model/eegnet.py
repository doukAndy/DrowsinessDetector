import torch
from torch.nn import Module, Sequential
from torch.nn import Conv2d, AvgPool2d 
from torch.nn import BatchNorm2d
from torch.nn import Linear, ELU, Softmax, Dropout
from torch.nn.modules.module import _addindent
import numpy as np

from experiment.config_simplified import Config
cfg = Config()


class Conv2dWithConstraint(Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs) -> None:
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(Module):
    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = Sequential(
            Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            ELU(),
            AvgPool2d((1, cfg.pool_1), stride=cfg.pool_1),
            Dropout(p=dropoutRate))
        block2 = Sequential(
            # SeparableConv2D =======================
            Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            ELU(),
            AvgPool2d((1, cfg.pool_2), stride=cfg.pool_2),
            Dropout(p=dropoutRate))
        return Sequential(block1, block2)


    def ClassifierBlock(self, inputSize, n_classes):
        return Sequential(
            Linear(inputSize, n_classes, bias=False),
            # Softmax(dim=1)
            )

    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, n_classes=cfg.n_classes, channels=cfg.channels, samples=cfg.samples,
                 dropoutRate=cfg.dropout_e, kernelLength= cfg.kernel_len1, kernelLength2=cfg.kernel_len2, F1=cfg.F1,
                 D=cfg.D, F2=cfg.F2):
        super(EEGNet, self).__init__()
        self.name = 'EEGNet'
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropoutRate

        self.blocks = self.InitialBlocks(dropoutRate)
        self.blockOutputSize = self.CalculateOutSize(self.blocks, channels, samples)
        self.classifierBlock = self.ClassifierBlock(self.F2 * self.blockOutputSize[1], n_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.classifierBlock(x)

        return x

def categorical_criterion_cls(y_pred, y_true):
    # y_pred = y_pred.cuda()
    # y_true = y_true.cuda()
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

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

