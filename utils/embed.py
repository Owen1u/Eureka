from ast import List
import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    '''
    in_channels: 通道数
    patch_size: 打patch的单边size，patch大小 patch_size * patch_size
    emb_size: patch_size^2 * channel 算好再填
    img_size: 输入图片大小（单边）
    '''
    def __init__(self,in_channels: int = 3, patch_size= [16,16], emb_size: int = 768, img_size= [224,224]):
        super().__init__()
        self.patch_size = patch_size
        self.patch_cnn = nn.Conv2d(in_channels, emb_size, kernel_size=tuple(patch_size), stride=tuple(patch_size))
        self.flaten = Rearrange('b e h w -> b (h w) e')
        # 生成一个维度为emb_size的向量当做cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # 位置编码信息，一共有(img_size / patch_size)**2 + 1(cls token)个位置向量
        self.positions = nn.Parameter(torch.randn(int((img_size[0]* img_size[1]/ patch_size[0]/patch_size[1]) + 1), emb_size))

    def forward(self,x):
        '''
        x: (N,C,H,W)
        output:(N,emb_size,img_size[0]*img_size[1]/patch_size[0]/patch_size[1]) + 1)
        '''
        N = x.size(0)
        # 将cls_token 扩展成 N个: [1, 1, emb_size] => [N, 1, emb_size]
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=N)
        # x->(N, emb_size, H//patch_size, W//patch_size)
        x = self.patch_cnn(x)
        # x->(N,H*W//patch_size^2,emb_size)
        x = self.flaten(x)
        # x->(N,H*W//patch_size^2+1,emb_size)
        x = torch.cat([cls_tokens,x],dim=1)
        x = x + self.positions
        return x

if __name__ == '__main__':
    torch.manual_seed(2022)
    x = torch.rand([2, 3, 64, 80])
    patchEmbedding = PatchEmbedding(3, [8,10], 192, [64,80])
    print(patchEmbedding(x).shape)