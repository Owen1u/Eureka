'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-20 20:46:17
'''

import os
from glob import glob
from PIL import Image
import cv2
import numpy as np
from collections.abc import Iterable
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BaseDataset():
    def __init__(self,size:Iterable,imagenet:bool=False,*args,**kw) -> None:
        self.size=size
        self.imagenet = imagenet
        
        self.totensor = transforms.ToTensor()
        if imagenet:
            self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        else:
            self.norm = transforms.Normalize(std=(0.5,0.5,0.5),
                                            mean=(0.5,0.5,0.5))

    def shape(self,img):
        if isinstance(img,Image.Image):
            w,h = img.size
        elif isinstance(img,np.ndarray):
            h,w,c = img.shape
        return h,w
    
    def resize(self,img):
        if isinstance(img,Image.Image):
            return img.resize((self.size[-1],self.size[-2]))
        elif isinstance(img,np.ndarray):
            return cv2.resize(img,(self.size[-1],self.size[-2]))
    
    def keep_ratio_crop_center(self,img):
        if isinstance(img,Image.Image):
            w,h = img.size
            ratio = self.size[-1]/self.size[-2]
            new_w= int(ratio*h) if ratio*h<=w else w
            new_h= int(new_w/ratio)
            return img.crop((int(0.5*(w-new_w)),int(0.5*(h-new_h)),int(0.5*(w-new_w))+new_w,int(0.5*(h-new_h))+new_h))
    
    def crop(self,img,left,up):
        if isinstance(img,Image.Image):
            w,h = img.size
            assert left+self.size[-1]<=w and up+self.size[-2]<=h,'crop size is larger than image'
            return img.crop((left,up,left+self.size[-1],up+self.size[-2]))
        elif isinstance(img,np.ndarray):
            h,w,c = img.shape
            assert left+self.size[-1]<=w and up+self.size[-2]<=h,'crop size is larger than image'
            return img[up:up+self.size[-2],left:left+self.size[-1],:]
    
    def cv2PIL(self,img_cv):
        return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))

    def PIL2cv(self,img_pil):
        return cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)
    
    def img2tensor(self,img):
        if isinstance(img,np.ndarray):
            img = self.cv2PIL(img)
        return self.totensor(img)
    
    def normalize(self,img):
        return self.norm(img)
    
    def write(self,img,path):
        if isinstance(img,Image.Image):
            img = self.PIL2cv(img)
        cv2.imwrite(path,img)
            
    def list2tensor(l:list):
        return torch.Tensor(l)

    def tensor2list(t:torch.Tensor):
        return t.tolist()
        
class YoloDataset(Dataset,BaseDataset):
    def __init__(self,root,image_path,label_path,size:Iterable,imagenet:bool,nsample:int=None) -> None:
        super().__init__(size=size,imagenet=imagenet)
        
        self.root = root
        self.image_path = image_path
        self.label_path = label_path
        self.images_path = glob(os.path.join(root,image_path,'*.jpg'),recursive=True)
        self.labels_path = glob(os.path.join(root,label_path,'*.txt'),recursive=True)
        for label in self.labels_path:
            filename,ext = os.path.splitext(os.path.basename(label))
            filename = os.path.join(self.root,self.image_path,filename+'.jpg')
            if filename not in self.images_path:
                self.images_path.remove(filename)
        self.nsample = nsample
        
    def __len__(self):
        if self.nsample is None:
            return len(self.images_path)
        else:
            return self.nsample
    
    def __getitem__(self, index:int):
        image = Image.open(self.images_path[index])
        filename = os.path.basename(self.images_path[index])
        filename,ext = os.path.splitext(filename)
        labels=[]
        with open(os.path.join(self.root,self.label_path,filename+'.txt')) as txtfile:
            for line in txtfile.readlines():
                line = [float(i) for i in line.split(' ')]
                labels.append(line)
        
        return image,labels

class ImageDataset(Dataset,BaseDataset):
    def __init__(self, path, size: Iterable, imagenet: bool = False,ext='.jpg', *args, **kw) -> None:
        super().__init__(size, imagenet, *args, **kw)
        
        self.images_path = glob(os.path.join(path,'*'+ext),recursive=True)
    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, index:int):
        image = Image.open(self.images_path[index]) 
        return image
