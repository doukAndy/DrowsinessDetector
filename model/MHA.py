from typing import Callable, List, Optional, Tuple
import math
import warnings
import torch.nn.functional as F
import torch
from torch import _VF
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes

Tensor = torch.Tensor

def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    drop_key: Optional[Tensor] = True,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))

    if dropout_p > 0:
        if drop_key == True:
            m_r = torch.ones_like(attn) * dropout_p
            attn += torch.bernoulli(m_r) * -1e12
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=dropout_p)
    else:
        attn = F.softmax(attn, dim=-1)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(
    x: Tensor,
    layer: int,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:

    # set up shape vars
    tgt_len, bsz, embed_dim = x.shape
    src_len = tgt_len
    assert embed_dim == embed_dim_to_check, f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    
    q, k, v = F._in_projection_packed(x, x, x, in_proj_weight, in_proj_bias)
    
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, drop_key=False, dropout_p=dropout_p-0.05*layer)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    return attn_output, attn_output_weights.sum(dim=1) / num_heads
    
