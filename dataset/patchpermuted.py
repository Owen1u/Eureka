'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-20 23:14:26
'''

import sys
sys.path.append('./')
import torch
import random
from dataset.base import ImageDataset
from torchvision import transforms
from collections.abc import Iterable

class PermutedDataset(ImageDataset):
    def __init__(self, path, size: Iterable, patch_size:Iterable, imagenet: bool = False, ext='.jpg',aug_tfs=None, *args, **kw) -> None:
        super().__init__(path, size, imagenet, ext, *args, **kw)
        
        self.channel,self.img_h,self.img_w = size
        self.patch_size = patch_size
        assert self.img_h%patch_size[0]==0 and self.img_w%patch_size[1]==0,'imagesize must be devided by patchsize'
        self.patch_num_h = self.img_h//patch_size[0]
        self.patch_num_w = self.img_w//patch_size[1]
        self.patch_num = self.patch_num_h * self.patch_num_w

        self.augment_tfs = aug_tfs
        
    def __getitem__(self, index: int):
        image = super().__getitem__(index)
        if self.augment_tfs is not None:
            image = self.augment_tfs(image)
        h,w = self.shape(image)
        # image = self.crop(image,int(0.5*(w-self.size[-1])),int(0.5*(h-self.size[-2])))
        image = self.keep_ratio_crop_center(image)
        image = self.resize(image)
        image = self.norm(self.totensor(image)) 
        images_shuffle,idx = self.shuffle_one(image)
        
        return images_shuffle,idx,image
    
    def shuffle_one(self,images:torch.Tensor):
        _images = images.reshape(self.channel,self.patch_num_h,self.patch_size[0],self.patch_num_w,self.patch_size[1])
        _images = _images.permute(0,1,3,2,4)
        _images = _images.reshape(self.channel,self.patch_num,self.patch_size[0],self.patch_size[1])

        idx=torch.randperm(_images.size(1))

        _images=_images[:,idx,:,:]

        images_shuffle = _images.reshape(self.channel,self.patch_num_h,self.patch_num_w,self.patch_size[0],self.patch_size[1])
        images_shuffle = images_shuffle.permute(0,1,3,2,4)
        images_shuffle = images_shuffle.reshape(self.channel,self.img_h,self.img_w)
    
        idx = idx.long()
        return images_shuffle,idx
        