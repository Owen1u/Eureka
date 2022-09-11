'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2022-09-11 00:18:58
'''
import torch
import torch.nn as nn

class InitWeight(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if  m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        elif isinstance(m,nn.Parameter):
            nn.init.trunc_normal_(m,std=.02)