from experiment.config_simplified import Config
cfg = Config()
import os
# import sys
# sys.path.append(os.getcwd())
from typing import Optional, Tuple
import copy

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np
import wandb
from model.MHA import multi_head_attention_forward


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 batch_first=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        xavier_uniform_(self.in_proj_weight)
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        constant_(self.in_proj_bias, 0.)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs) 
        constant_(self.out_proj.bias, 0.)     

        
    def forward(self, x, layer) -> Tuple[Tensor, Optional[Tensor]]:

        if self.batch_first:
            x = x.transpose(1, 0)
    

        attn_output, attn_output_weights = multi_head_attention_forward(
            x, layer, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, 
            self.dropout, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
            training=self.training)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dropout, layer_norm_eps=1e-5, batch_first=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, 2*d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2*d_model, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(0)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, layer):
        src2, score = self.self_attn(src, layer)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, score


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, num_layers, d_model, nhead, device, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, device=device)

        self.layers = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src: Tensor) -> Tensor:

        outputs = src
        scores = []

        for i in range(self.num_layers):
            outputs, score = self.encoder_layer(outputs, i)
            scores.append(score)
        # for mod in self.layers:
        #     outputs, score = mod(outputs)
        #     scores.append(score)

        return outputs, scores


class Head(nn.Module):
    def __init__(self, d_model, num_classes, mode, dropout=0.5):
        super(Head, self).__init__()
        self.mode = mode
        self.dropout = dropout
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        if self.mode == 'global':
            x = x.mean(dim=1)  # [992, 90]
            x = nn.Linear(in_features=30, out_features=2)
        elif self.mode == 'class':
            x = self.norm(x)[:, 0]  # [32, 32, 90] [64, 31, 384] [64, 384]
            x = self.mlp(x)  # [64, 2]
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, input_seq, output_seq, mode): 
        super().__init__()
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.mode = mode
        if mode == 'sequence':
            self.position_embedding = nn.Parameter(torch.zeros(1, input_seq, 1))
        else:
            self.position_embedding = nn.Parameter(torch.zeros(1, input_seq, output_seq))

    def forward(self, x):
        if self.mode == 'sequence':
            x = x + self.position_embedding.expand(-1, -1, self.output_seq)
        else:
            x = x + self.position_embedding
        # import matplotlib.pyplot as plt
        # pe = self.position_embedding[:, :cfg.seq_len_hlt].detach().cpu().numpy()[0]
        # plt.imshow(pe)
        # plt.colorbar()
        # plt.savefig('pe.png')
        # plt.close()
        return x


class Patching(nn.Module):
    def __init__(self):
        super(Patching, self).__init__()

        self.patch_1D = Rearrange('b h c (s d) -> b (h s) (d c)', d=cfg.patch_len)

    def forward(self, x):
        shape = x.shape     # [32, 1, 30, 384]
        x = self.patch_1D(x)
        return x, shape[0]  # [32, 384, 30]


class Tokenizer(nn.Module):
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.patching = Patching()

    def forward(self, x):
        x, b = self.patching(x)  # [32, 128, 90]
        x = x.unfold(step=cfg.sliding_window//2, size=cfg.sliding_window, dimension=1)  # [32, 31, 90, 8]
        cfg.seq_len_hlt = x.shape[1]  # 47
        x = Rearrange('b d s c -> (b d) c s', b=b)(x)  # [992, 8, 90]
        return x, b


class LowLevelTransformer(nn.Module):
    def __init__(self):
        super(LowLevelTransformer, self).__init__()
        self.transformer = TransformerEncoder(num_layers=cfg.num_layers, d_model=cfg.d_model, nhead=cfg.nhead, device=torch.device("cpu"), dropout=0.3)
        self.position_embedding = PositionEmbedding(input_seq=cfg.sliding_window, output_seq=cfg.d_model, mode='sequence')
        self.head = Head(d_model=cfg.d_model, num_classes=cfg.n_classes, dropout=cfg.dropout, mode=cfg.mode1)

    def forward(self, tuples):
        x, b = tuples      
        x = self.position_embedding(x)          # [992, 8, 90]
        x, score = self.transformer(x) 
        out = self.head(x)           # [992, 2]
        # out = Rearrange('(b d) s -> b d s', b=b)(out)
        # out = out.mean(dim=1)
        return out, b, x, score


class HighLevelTransformer(nn.Module):
    def __init__(self):
        super(HighLevelTransformer, self).__init__()
        self.transformer = TransformerEncoder(num_layers=6, d_model=cfg.sliding_window, nhead=1, device=torch.device("cpu"), dropout=0.7) 
        self.head = Head(d_model=cfg.sliding_window, num_classes=cfg.n_classes, dropout=cfg.dropout, mode=cfg.mode2)

        self.class_embedding = nn.Parameter(torch.zeros(1, 1, cfg.sliding_window))
        self.position_embedding = PositionEmbedding(input_seq=cfg.d_model, output_seq=cfg.sliding_window, mode='sequence')
        self.pool = nn.AvgPool2d((64, 1))
        
    def forward(self, tuples):
        x, b = tuples      # [992, 8, 90] [64, 384, 30]
        # x = self.pool(x)
        # x = x.mean(dim=1)  # [992, 90]
        x = Rearrange('b c s -> b s c', b=b)(x)  # [32, 31, 90] # [64, 30, 384]

        x = self.position_embedding(x)  # [64, 30, 384]
        x = torch.cat((self.class_embedding.expand(b, -1, -1), x), dim=1)  # [64, 31, 384]
        
        x, score = self.transformer(x)
        out = self.head(x)
        return out, score


class HierXFMR(nn.Module):
    def __init__(self):
        super(HierXFMR, self).__init__()
        self.name = 'HierXFMR'
        self.tokenizer = Tokenizer()
        self.lowlevel = LowLevelTransformer()
        self.highlevel = HighLevelTransformer()

    def forward(self, x):
        x = self.tokenizer(x)
        llt, batch, x, score = self.lowlevel(x)
        hlt, _ = self.highlevel((x, batch))    

        return llt, hlt, score  # llt, hlt, score 

    def llt_freeze(self):
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        for param in self.lowlevel.parameters():
            param.requires_grad = False

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
