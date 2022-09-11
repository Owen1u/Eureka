'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: https://gitee.com/stdu-jtxy_liwx/vision-transformer.git
LastEditTime: 2022-09-11 00:37:15
'''

import sys
sys.path.append('../')
from typing import List
import torch
from torch import nn
from Eureka.utils import PatchEmbedding
from einops.layers.torch import Rearrange, Reduce

class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000, isseq2seq = True):
        super().__init__()
        self.net = nn.Sequential()
        if isseq2seq:
            self.net.add_module(
                'reduce_seq2seq',Reduce('b n e -> b e', reduction='mean')
            )
        self.net.add_module('LayerNorm',nn.LayerNorm(emb_size))
        self.net.add_module('Linear',nn.Linear(emb_size, n_classes))
    def forward(self,x):
        return self.net(x)

def int2list(x:int):
    return x if isinstance(x,List) else [x,x]

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

        self.layernorm1 = nn.LayerNorm(embed_size)
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, expend * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expend * embed_size, embed_size),
        )
    def forward(self,input):
        x = self.layernorm1(input)
        # (N,seq_len,embed_size)->(seq_len,N,embed_size)
        # Multi_head_Attention's old version needs batch to be the dim1
        # x = x.transpose(0,1)

        # Q,K,V
        x ,x_w = self.self_attention(x, x, x)
        x = x + input
        out = self.feed_forward(x)
        out = out + x
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


class VisionTransformer(nn.Module):
    """VisionTransformer
    """
    def __init__(self,
            in_channels: int = 3,
            patch_size = [16,16],
            img_size = [224,224],
            num_heads: int = 8,
            num_layers: int = 12,
            dropout: float = 0.1,
            expand_forward: int = 4,
            n_classes: int = 100,
            isseq2seq = True
            ):
        super().__init__()
        self.n_classes = n_classes
        patch_size = int2list(patch_size)
        pH,pW = patch_size
        iH,iW = int2list(img_size)
        embed_size = pH * pW * in_channels

        # embed_size = patch_size**2 * in_channels
        self.patch_embed = PatchEmbedding(in_channels,patch_size,embed_size,img_size)
        self.encoder = Encoder(embed_size=embed_size,num_heads=num_heads,dropout=dropout,expend=expand_forward,num_layers=num_layers)
        if self.n_classes>0:
            self.outputLayer = ClassificationHead(emb_size=embed_size,n_classes=n_classes,isseq2seq=isseq2seq)

    def forward(self,input):
        x = self.patch_embed(input)
        x = self.encoder(x)
        if self.n_classes>0:
            x = self.outputLayer(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda')
    x = torch.rand([2, 3, 32, 320]).to(device)

    vit = VisionTransformer(
        in_channels=3,
        patch_size=[8,16],
        img_size=[32,320],
        num_heads=6,
        num_layers=6,
        dropout=0.1,
        expand_forward=4,
        n_classes=4000,
        isseq2seq = False
    ).to(device)

    print(vit(x).shape)