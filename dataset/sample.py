from PIL import Image
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import sampler

class resizeNormalize():
    '''
    size: [c,h,w]
    interpolation:插值方式，默认双线性插值
    '''
    def __init__(self,size,isnormal=True,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        if isnormal:
            self.toTensor_Normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(std=(0.5)*size[0],mean=(0.5)*size[0])
            ])
        else:
            self.toTensor_Normalize = transforms.Compose([
                transforms.ToTensor()
            ])

    def __call__(self,img):
        img = img.resize((self.size[2],self.size[1]),self.interpolation)
        img = self.toTensor_Normalize(img)
        # toTensor后数值[0,1]，
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples

        