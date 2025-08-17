import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


import torch
import torch.nn as nn
from einops import rearrange, repeat

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        q = self.query_projection(queries)
        k = self.key_projection(keys)
        v = self.value_projection(values)
        out, _ = self.inner_attention(q, k, v, attn_mask=attn_mask)
        return self.out_projection(out)

class ThreeStageAttentionLayer(nn.Module):
    """
    Three Stage attention: Space Time Dim
    输入输出 shape: [space_dim, ts_d, seg_num, d_model]
    Link_matrix: [num_edges, 2] Tensor
    """
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.space_attention = AttentionLayer(d_model, n_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP3 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

        self.seg_num = seg_num

    def forward(self, x, Link_matrix):
        batch = x.shape[0]
        ts_d = x.shape[1]
        seg_num = x.shape[2]

        ### 1. Time attention ###
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        ### 2. Dimension attention ###
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        ### 3. Space attention ###
        # create mask
        space_mask = torch.zeros(batch, seg_num, seg_num, device=x.device)
        Link_matrix = Link_matrix.long()
        for edge in Link_matrix:
            u, v = edge[0].item(), edge[1].item()
            space_mask[:, u, v] = 1
            space_mask[:, v, u] = 1

        space_in = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        space_in_flat = rearrange(space_in, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')

        space_out = self.space_attention(space_in_flat, space_in_flat, space_in_flat, attn_mask=None)
        space_out = rearrange(space_out, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b=batch)
        final_out = space_in + self.dropout(space_out)
        final_out = self.norm5(final_out)
        final_out = final_out + self.dropout(self.MLP3(final_out))

        return final_out

