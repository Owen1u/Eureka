'''
Descripttion: 本文件的encoder和decoder模块，不含输入的embed层；也不含decoder输出的全连接和softmax层；
version: 
Contributor: Minjun Lu
Source: https://github.com/hyunwoongko/transformer
LastEditTime: 2022-09-10 19:52:53
'''

import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    '''
    单个Encoder层
    embed_size: patch_size^2 * channel
    num_heads:
    dropout:
    expend: feed_forward层的dim = embed_size * expend
    '''
    def __init__(self,embed_size=786,num_heads=8,dropout=0.1,expend=4):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, expend * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expend * embed_size, embed_size),
        )
        self.layernorm2 = nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,input):
        # (N,seq_len,embed_size)->(seq_len,N,embed_size)
        # Multi_head_Attention's old version needs batch to be the dim1
        # x = x.transpose(0,1)

        # Q,K,V
        x ,x_w = self.self_attention(input, input, input)
        x = self.layernorm1(x + input)
        x = self.dropout1(x)

        out = self.feed_forward(x)
        out = self.layernorm2(out + x)
        out  = self.dropout2(out)
        return out

class Encoder(nn.Module):
    '''
    num_layers: EncoderLayer的层数
    '''
    def __init__(self, embed_size=768, num_heads=8, dropout=0.1, expend=4, num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(embed_size,num_heads,dropout,expend) for _ in range(num_layers)
        ])
    def forward(self,input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out
        
class  DecoderLayer(nn.Module):
    def __init__(self,embed_size=768,num_heads=8,dropout=0.1,expend=4):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)
        self.layernorm1=nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)

        self.enc_dec_attention = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)
        self.layernorm2=nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, expend * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expend * embed_size, embed_size),
        )
        self.layernorm3 = nn.LayerNorm(embed_size)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,dec,enc):
        _x = dec
        # Q,K,V
        x ,x_w = self.self_attention(dec, dec, dec)
        x = self.layernorm1(x+_x)
        x = self.dropout1(x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(x,enc,enc)

            x = self.layernorm2(x+_x)
            x = self.dropout2(x)

        out = self.feed_forward(x)
        out = self.layernorm3(out+x)
        out = self.dropout3(out)
        return out

class Decoder(nn.Module):
    def __init__(self,embed_size=768, num_heads=8, dropout=0.1, expend=4, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size,num_heads,dropout,expend) for _ in range(num_layers)
        ])
    def forward(self,input,enc):
        out = input
        for layer in self.layers:
            out = layer(out,enc)
        return out


if __name__ =='__main__':
    x = torch.rand([2,65,192])
    encoder = Encoder(embed_size=192, num_heads=8, dropout=0.1, expend=4, num_layers=6)
    print(encoder(x).size())