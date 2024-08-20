import torch
import torch.nn as nn

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        #Initialization of the attention weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None,is_causal=False):
        #capture the attention weights
        src = src.permute(1,0,2)
        #src2 shape: (t, batch, nd)
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        #src shape: (t, batch, nd)
        src = self.norm2(src)
        self.attn_weights = attn_weights
        src = src.permute(1,0,2)
        return src

    def get_attention_weights(self):

        return self.attn_weights
