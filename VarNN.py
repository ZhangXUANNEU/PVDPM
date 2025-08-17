import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from einops import repeat
from math import ceil

from var_models.var_encoder import Encoder
from var_models.var_decoder import Decoder
from var_models.var_embed import DES_embedding

class VarNN(nn.Module):
    def __init__(self, node_num, in_len, out_len, edge_index,
                 data_dim, seg_len, win_size,
                 factor, d_model, d_ff, n_heads,
                 e_layers, dropout=0.1, baseline=False,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(VarNN, self).__init__()

        self.node_num = node_num
        self.in_len = in_len
        self.out_len = out_len
        self.edge_index = edge_index
        self.device = device
        self.baseline = baseline

        self.seg_len = seg_len
        self.pad_in_len = ceil(in_len / seg_len) * seg_len
        self.pad_out_len = ceil(out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - in_len

        # Value Embedding
        self.enc_value_embedding = DES_embedding(seg_len, d_model)
        self.pre_norm = nn.LayerNorm(d_model)

        # Positional Embedding
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, data_dim, self.pad_out_len // seg_len, d_model)
        )

        # Encoder
        self.encoder = Encoder(
            e_layers, win_size, d_model, n_heads, d_ff, block_depth=1,
            dropout=dropout, in_seg_num=self.pad_in_len // seg_len, factor=factor
        )

        # Decoder
        self.decoder = Decoder(
            seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout,
            out_seg_num=self.pad_out_len // seg_len, factor=factor
        )

    def forward(self, Graph):


        input, edge_index, = Graph.x, Graph.edge_index
        P_input = input[:,7:27].unsqueeze(-1)
        Q_input = input[:,27:47].unsqueeze(-1)
        A_input = input[:,47:67].unsqueeze(-1)
        Link_matrix = Graph.edge_index
        Other_input = input[:,0:6]
        space_size = P_input.shape[0]
        # input [space number, time number, dim number]
        Physics_input = torch.cat([A_input, P_input, Q_input], dim=-1).float()

        if (self.baseline):
            base = Physics_input.mean(dim=1, keepdim=True)
        else:
            base = 0

        if (self.in_len_add != 0):
            Physics_input = torch.cat((Physics_input[:, :1, :].expand(-1, self.in_len_add, -1), Physics_input), dim=1)
        x_seq = self.enc_value_embedding(Physics_input)
        enc_out = self.encoder(x_seq,Link_matrix)
        # enc_out: [s d ts_d l]
        dec_in = repeat(self.dec_pos_embedding, 's ts_d l d -> (repeat s) ts_d l d', repeat=space_size)
        predict_y = self.decoder(dec_in, enc_out, Link_matrix)  # [B, T, ts_d]
        Q, P, A = torch.chunk(predict_y, 3, dim=-1)
        return Q, P, A

