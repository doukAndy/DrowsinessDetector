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

from experiment.config import Config
cfg = Config()


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiHeadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, score = self.self_attn(src, src, src, attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, score


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, num_layers, d_model, nhead, device, dropout=0.1, norm=None):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model,
                                                     dropout=dropout, device=device)

        self.layers = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        outputs = src
        scores = []

        for mod in self.layers:
            outputs, score = mod(outputs, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            scores.append(score)

        if self.norm is not None:
            outputs = self.norm(outputs)

        return outputs, scores


class Head(nn.Module):
    def __init__(self, d_model, num_classes, mode, dropout=0.):
        super(Head, self).__init__()
        self.mode = mode
        self.dropout = dropout
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp = nn.Linear(in_features=d_model, out_features=num_classes)

    def forward(self, x):
        if self.mode == 'global':
            x = x.mean(dim=1)
        if self.mode == 'class':
            x = self.norm(x)[:, 0]
        x = self.mlp(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, input_seq, d_model): 
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, input_seq, d_model))

    def forward(self, x):
        x = x + self.position_embedding[:cfg.seq_len_hlt]
        return x


class Patching(nn.Module):
    def __init__(self):
        super(Patching, self).__init__()

        self.patch_1D = Rearrange('b h c (s d) -> (b h s) (d c)', d=cfg.patch_len)

    def forward(self, x):
        shape = x.shape
        x = self.patch_1D(x)
        return x, shape[0]


class Tokenizer(nn.Module):
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.patching = Patching()

    def forward(self, x):
        x, b = self.patching(x)
        x = Rearrange('(b d) s -> b d s', b=b)(x)
        x = Rearrange('b (d n) s -> b d (n s)', n=1)(x)
        x = x.unfold(step=cfg.sliding_window//2, size=cfg.sliding_window, dimension=1)
        cfg.seq_len_hlt = x.shape[1]
        x = Rearrange('b d s c -> (b d) c s', b=b)(x)
        return x, b


class LowLevelTransformer(nn.Module):
    def __init__(self):
        super(LowLevelTransformer, self).__init__()
        self.transformer = TransformerEncoder(num_layers=cfg.num_layers, d_model=cfg.d_model,
                                              nhead=cfg.nhead,
                                              device=cfg.device,
                                              dropout=cfg.dropout, norm=cfg.norm)

        self.head = Head(d_model=cfg.d_model, num_classes=cfg.n_classes, dropout=cfg.dropout,
                         mode=cfg.mode1)

    def forward(self, tuples):
        x, b = tuples
        x, _ = self.transformer(x)
        out = self.head(x)
        return out, b, x


class HighLevelTransformer(nn.Module):
    def __init__(self):
        super(HighLevelTransformer, self).__init__()
        self.transformer = TransformerEncoder(num_layers=cfg.num_layers, d_model=cfg.d_model,
                                              nhead=1,
                                              device=cfg.device,
                                              dropout=cfg.dropout, norm=cfg.norm)
        self.head = Head(d_model=cfg.d_model, num_classes=cfg.n_classes, dropout=cfg.dropout,
                         mode=cfg.mode2)

        self.class_embedding = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.position_embedding = PositionEmbedding(input_seq=(cfg.seq_len_hlt + 1), d_model=cfg.d_model)
        
    def forward(self, tuples):
        # wandb.watch(model, criterion, log='all', log_freq=10)
        x, b = tuples
        x = x.mean(dim=1)
        x = Rearrange('(b c) s -> b c s', b=b)(x)
        # x = x[:, :cfg.seq_len_hlt, :]
        x = torch.cat((self.class_embedding.expand(b, -1, -1), x), dim=1)
        # x = self.position_embedding(x)
        x = x + nn.Parameter(torch.zeros(1, x.shape[1], x.shape[2])).to(cfg.device)   # positional embedding
        
        x, score = self.transformer(x)
        out = self.head(x)

        # wandb.log({"hlt_out":out})
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
        llt, batch, x = self.lowlevel(x)
        hlt, score = self.highlevel((x, batch))    

        return hlt  # llt, hlt, score 

    def llt_freeze(self):
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        for param in self.lowlevel.parameters():
            param.requires_grad = False

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
