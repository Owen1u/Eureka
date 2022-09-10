import torch
class Counter():
    def __init__(self):
        self.reset()
    
    def add(self,v:torch.Tensor):
        self.num_count+=v.numel()
        self.sum += v.sum()

    def reset(self):
        self.num_count = 0
        self.sum = 0

    def mean(self):
        return self.sum/float(self.num_count) if self.num_count!=0 else 0