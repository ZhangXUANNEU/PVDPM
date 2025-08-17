import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed

class DES_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DES_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        # 输入: [space_num, time_num, dim]
        space_num, time_num, dim = x.shape
        num_seg = time_num // self.seg_len

        # rearrange: [s, t, d] -> [s * d * num_seg, seg_len]
        x = rearrange(x, 's (n seg_len) d -> (s d n) seg_len', seg_len=self.seg_len)
        x_embed = self.linear(x)  # [s * d * num_seg, d_model]

        # reshape回去: [s * d * n, d_model] -> [s, d, n, d_model]
        x_embed = rearrange(x_embed, '(s d n) d_model -> s d n d_model', s=space_num, d=dim)

        return x_embed
