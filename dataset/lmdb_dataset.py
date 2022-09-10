from torch.utils.data import Dataset
from Eureka.dataset.sample import resizeNormalize
# from paddle.io import Dataset
from PIL import Image
import lmdb
import six
import os
class lmdbDataset(Dataset):
    def __init__(self,db_path = None,size=None,isnormal=True,transform=None,target_transform=None):
        super(lmdbDataset,self).__init__()
        self.env = lmdb.open(db_path,max_readers=32,readonly=True,lock=False,readahead=False,meminit=False,map_size=1048576)
        assert self.env,'cannot create lmdb from %s' %(db_path)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()))
        
        self.size = size
        self.resize_normal = resizeNormalize(size,isnormal=isnormal)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index: int):
        assert index < len(self),'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            if self.size[0]==1:
                img = Image.open(buf).convert('L')
            else:
                img = Image.open(buf).convert('RGB')

            img = self.resize_normal(img)

            label_key = 'label-%09d'.encode() % index
            label = str(txn.get(label_key).decode())

            return (img,label)

if __name__ == '__main__':
    data = lmdbDataset(db_path='E:\DeepLearning\my_pro\data\chinese_document\\test',size=[1,32,104])


