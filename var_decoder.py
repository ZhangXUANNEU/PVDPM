import torch
import torch.nn as nn
from einops import rearrange
from var_models.attn import FullAttention, AttentionLayer, ThreeStageAttentionLayer

class DecoderLayer(nn.Module):
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num=10, factor=10):
        super(DecoderLayer, self).__init__()
        # 使用带空间注意力的 TwoStageAttentionLayer
        self.self_attention = ThreeStageAttentionLayer(out_seg_num, factor, d_model, n_heads, d_ff, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross, Link_matrix):
        batch = x.shape[0]
        # self_attention 传入 Link_matrix
        x = self.self_attention(x, Link_matrix)

        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')

        tmp = self.cross_attention(x, cross, cross)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        return dec_output, layer_predict


class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 20, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, \
                                        out_seg_num, factor))

    def forward(self, x, cross, Link_matrix):
        final_predict = None
        i = 0
        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc, Link_matrix)  # 传入 Link_matrix
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)
        final_predict = torch.sigmoid(final_predict)
        return final_predict


