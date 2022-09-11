'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2022-09-11 11:12:52
'''

from requests import patch
import torch
import numpy as np
import cv2 as cv
import torchvision

class PatchShuffle():
    def __init__(self,img_size=(3,32,128),patch_size=(16,16),batchsize=1) -> None:
        self.channel = img_size[0]
        self.img_h = img_size[1]
        self.img_w = img_size[2]
        self.patch_size = patch_size
        assert self.img_h%patch_size[0]==0 and self.img_w%patch_size[1]==0,'imagesize must be devided by patchsize'
        self.patch_num_h = self.img_h//patch_size[0]
        self.patch_num_w = self.img_w//patch_size[1]
        self.patch_num = self.patch_num_h * self.patch_num_w
        self.batchsize = batchsize

    def shuffle(self,images:torch.Tensor):
        images_shuffle = torch.zeros([self.batchsize,self.channel,self.img_h,self.img_w])
        idxes = torch.zeros([self.batchsize,self.patch_num])
        images_trans = images.view(self.batchsize,self.channel,self.patch_num_h,self.patch_size[0],self.patch_num_w,self.patch_size[1])
        images_trans =images_trans.permute(0,1,2,4,3,5)
        images_trans = images_trans.reshape(self.batchsize,self.channel,self.patch_num,self.patch_size[0],self.patch_size[1])

        for batch in range(self.batchsize):
            img = images_trans[batch,:,:,:,:]

            idx = np.random.permutation(img.shape[1])
            idx = torch.from_numpy(idx)
            # idx = torch.randperm(y.shape[2])
            img = img[:,idx,:,:].view(img.size())
            img = img.view(self.channel,self.patch_num_h,self.patch_num_w,self.patch_size[0],self.patch_size[1]).permute(0,1,3,2,4)
            img = img.reshape(self.channel,self.img_h,self.img_w)
            images_shuffle[batch,:,:,:]=img
            idxes[batch,:] = idx

        idxes = idxes.long()
        return images_shuffle,idxes

    def decoder(self,images_shuffle,idx):
        images = torch.zeros([self.batchsize,self.channel,self.img_h,self.img_w])
        images_trans = images_shuffle.view(self.batchsize,self.channel,self.patch_num_h,self.patch_size[0],self.patch_num_w,self.patch_size[1])
        images_trans =images_trans.permute(0,1,2,4,3,5)
        images_trans = images_trans.reshape(self.batchsize,self.channel,self.patch_num,self.patch_size[0],self.patch_size[1])

        for batch in range(self.batchsize):
            img = images_trans[batch,:,:,:,:]
            img_ = torch.zeros(img.size())
            for i,pos in enumerate(idx[batch]):
                img_[:,pos,:,:]=img[:,i,:,:]

            img_ = img_.view(self.channel,self.patch_num_h,self.patch_num_w,self.patch_size[0],self.patch_size[1]).permute(0,1,3,2,4)
            img_ = img_.reshape(self.channel,self.img_h,self.img_w)

            images[batch,:,:,:] = img_

        return images

    # 【todo】批量处理时，无法单独对样本随机
    def lookonepic(self,path,images_shuffle:torch.Tensor,idx,batchsize_n:int=0,isRGB=True,isSign=False):
        assert len(images_shuffle.size()) == 4,'images_shuffle:(batchsize,channel,imagesize_h,imagesize_w)'
        assert batchsize_n<images_shuffle.size()[0],"batchsize_n must be less than data's batchsize"
        # img = images_shuffle.view(self.batchsize,self.channel,self.patch_num_h,self.patch_num_w,self.patch_size[0],self.patch_size[1]).permute(0,1,2,4,3,5)
        # img = img.reshape(self.batchsize,self.channel,self.img_h,self.img_w)[batchsize_n,:,:,:].numpy()
        img = images_shuffle[batchsize_n].numpy()
        maxValue = img.max()
        img = img*255/maxValue
        img = np.uint8(img).transpose(1,2,0)
        if isRGB:
            img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
        if isSign:
            img = self._sign(img,idx[batchsize_n])
        cv.imwrite(path,img)

    def _sign(self,image,idx:torch.Tensor):
        n=0
        for i in range(image.shape[0]//self.patch_size[0]):
            for j in range (image.shape[1]//self.patch_size[1]):
                cv.putText(image,str(idx[n].item()),(int((0.5+j)*self.patch_size[1]),int((0.5+i)*self.patch_size[0])),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=2,color=(0,255,0),thickness=5)
                n+=1
        return image

if __name__ == '__main__':
    ps = PatchShuffle(img_size=(3,1000,1500),patch_size=(250,250))

    img = cv.imread('/nvme0n1/lmj/disorder_selfsup/exp.jpg')
    img = cv.resize(img,(1500,1000))
    print(img.shape)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    x = torchvision.transforms.ToTensor()(img)
    print(x.size())

    images_shuffle,idx = ps.shuffle(x)
    print(images_shuffle.size(),idx.size())
    ps.lookonepic('/nvme0n1/lmj/disorder_selfsup/lookonepicture.jpg',images_shuffle,idx,isSign=True)