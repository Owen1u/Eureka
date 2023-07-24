'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-24 14:42:13
'''
import os
import six
import lmdb
from collections.abc import Iterable
from PIL import Image
from torch.utils.data import Dataset
from base import BaseDataset
class LMDB(Dataset,BaseDataset):
    def __init__(self,db_path, size: Iterable, imagenet: bool = False, nsample=None, *args, **kw) -> None:
        super().__init__(size, imagenet, *args, **kw)
        
        self.env = lmdb.open(db_path,max_readers=32,readonly=True,lock=False,readahead=False,meminit=False,map_size=1048576)
        assert self.env,'cannot create lmdb from %s' %(db_path)
        
        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()))

        self.nsample = nsample
        
    def __len__(self):
        if self.nsample is None:
            return self.nSamples
        else:
            return self.nsample
        
    def __getitem__(self, index:int):
        index +=1       # index is from 1 to len()
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            
            image = Image.open(buf)
            label_key = 'label-%09d'.encode() % index
            label = str(txn.get(label_key).decode())
        
        return image,label

if __name__ == '__main__':
    d = LMDB('/server19/lmj/dataset/textimage/test/COCOv1.4',size=[3,32,128])
    print(len(d),d[len(d)-1])