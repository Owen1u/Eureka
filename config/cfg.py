import os
import sys
import yaml
import torch
import random
import numpy as np

class Config():
    def __init__(self,path='config/config.yml') -> None:
        self.path = path
        file = open(path,'r',encoding="utf-8")
        self.file_data = file.read()
        file.close()
        self.config = yaml.load(self.file_data,Loader=yaml.FullLoader)

    def __getitem__(self,key):
        return self.config[key]

    def update(self,**kw):
        '''
        input: key-value
        output: None
        '''
        for key,value in kw.items():
            self.config[key]=value

    def seed(self,*args):
        '''
        iuput: 'np','random','torch'中任意几个，或'all'。
        output: None
        '''
        try:
            seed = self.config['seed']
        except KeyError:
            print("文件{0}中，无seed参数".format(self.path))
            sys.exit()
        
        if "all" in args:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            if ("np" in args) or ("numpy" in args):
                np.random.seed(seed)
            if "random" in args:
                random.seed(seed)
            if "torch" in args:
                torch.manual_seed(seed)
        
if __name__ == '__main__':
    CUR_PATH= os.path.abspath(__file__)
    CUR_DIR = os.path.dirname(CUR_PATH)
    cfg = Config(os.path.join(CUR_DIR,'config.yml'))
    cfg.update(batchsize=7,epoch=30)
    print(cfg['batchsize'],cfg['epoch'])
    cfg.seed('random','np')