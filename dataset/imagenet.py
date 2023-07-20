'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-20 23:17:49
'''
import os
import sys
sys.path.append('./')
from PIL import Image
from collections.abc import Iterable
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.base import BaseDataset
import pandas as pd
import re

class ImageNet(Dataset,BaseDataset):
    def __init__(self,root,
                 mode='train',
                 size:Iterable=[3,224,224],
                 crop:bool=False,
                 imagenet:bool=True,
                 nsample:int=None) -> None:
        super().__init__(size=size,imagenet=imagenet)
        self.root = root
        self.mode = mode
        self._crop = crop
        self.imagenet = imagenet
        
        self.loc_solution_path = os.path.join(root,'LOC_'+mode+'_solution.csv')
        self.loc_solution = pd.read_csv(os.path.join(root,'LOC_'+mode+'_solution.csv'))
        
    def get_loc_solution(self):
        return self.loc_solution

    def get_image(self,filename,label):
        if self.mode=='train':
            img = Image.open(os.path.join(self.root,'ILSVRC','Data','CLS-LOC','train',label,filename+'.JPEG'))
        elif self.mode=='val':
            img = Image.open(os.path.join(self.root,'ILSVRC','Data','CLS-LOC','val',filename+'.JPEG'))

        return img
    
    def __len__(self):
        return self.loc_solution.shape[0]-1
    
    def __getitem__(self, index: int):
        loc_solution = pd.read_csv(self.loc_solution_path,skiprows=index,nrows=1)
        filename,prediction = loc_solution.values[0]
        label,x0,y0,x1,y1,_ = prediction.split(' ',maxsplit=5)
        
        img = self.get_image(filename,label)

        if self._crop:
            box = (x0,y0,x1,y1)
            box = (int(i) for i in box)
            img = img.crop(box)

        img = self.norm(self.totensor(self.resize(img))) 
        return img,label

if __name__=='__main__':

    # from mocov3 import loader
    # augmentation1 = transforms.Compose([
    #         transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
    #         transforms.RandomApply([
    #             transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    #         ], p=0.8),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=1.0),
    #         transforms.RandomHorizontalFlip()
    #     ])
    dataset = ImageNet('/server19/dataset/ImageNet1k/data',crop=False)

    img,label = dataset[4]
    print(img,label)

