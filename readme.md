Eureka: A Toolkit for Piling up Building Blocks at Sweet Will
======
<div align=center><img src="eureka.jpg" width="80%" height=200px></div>
<div align="center">
    <a href="https://pytorch.org/docs/stable/index.html">
        <img alt="Pytorch" src="https://img.shields.io/badge/doc-Pytorch-blueviolet">
    </a>
    <a href="https://www.anaconda.com">
        <img alt="Anaconda3" src="https://img.shields.io/badge/website-Anaconda3-blueviolet">
    </a>
    <a href="https://arxiv.org">
        <img alt="arxiv" src="https://img.shields.io/badge/paper-arxiv-blue">
    </a>
    <a href="https://arxiv.org">
        <img alt="dataset" src="https://img.shields.io/badge/dateset-kaggle-ff69b4">
    </a>
    <a href=""><img src="https://img.shields.io/badge/version-v0.0.1-red" alt="Version"></a>
    <a href=""><img src="https://img.shields.io/github/license/Owen1u/Eureka?style=plastic" alt="license"></a>
    <a href=""><img src="https://img.shields.io/github/stars/Owen1u/Eureka?style=social" alt="social"></a>
</div>
<div align="center"><a href="https://github.com/Owen1u/Eureka/tree/main/paper"><img height='' width='300' src="https://img.shields.io/badge/前沿模型精选%26主流数据集汇总-EF9421?style=for-the-badge&logo=codereview&logoColor=white&label=Review"></a></div>

<!-- TOC -->

- [Eureka: A Toolkit for Piling up Building Blocks at Sweet Will](#eureka-a-toolkit-for-piling-up-building-blocks-at-sweet-will)
  - [Usage](#usage)
    - [Getting Started](#getting-started)
    - [Do It Yourself](#do-it-yourself)
  - [Catalogue](#catalogue)
    - [Config](#config)
      - [Config](#config-1)
      - [config.yaml](#configyaml)
    - [Module](#module)
      - [VisionTransformer](#visiontransformer)
      - [FAN](#fan)
      - [Resnet](#resnet)
      - [SwinTransformer](#swintransformer)
      - [Transformer](#transformer)
    - [Dataset](#dataset)
      - [lmdbDataset](#lmdbdataset)
    - [Utils](#utils)
      - [Counter](#counter)
      - [PatchShuffle](#patchshuffle)
  - [### Output](#-output)
  - [Extension](#extension)
  - [6. 更新readme.md说明文档。](#6-更新readmemd说明文档)
  - [Version Description](#version-description)
    - [V0.0.1](#v001)

<!-- /TOC -->


-------
## Usage
### Getting Started
```Python
Eureka_path='[Eureka_Path]/Eureka' # 输入包路径
import os
import sys
sys.path.append(os.path.dirname(Eureka_path))  # 添加至环境变量
from Eureka.config import Config
from Eureka.dataset import lmdbDataset
……
```
### Do It Yourself
<font size=4 color="red">***不建议改动 Eureka 文件夹下代码，除非你是老司机！！！***</font>
```Python
# 以Swin-Transformer为例
Eureka_path='[Eureka_Path]/Eureka'
import os
import sys
from turtle import forward

sys.path.append(os.path.dirname(Eureka_path))

import torch
import torch.nn as nn
# 此处SwinTransformer_x为官方实例，因导入SwinTransformer
from Eureka.module import SwinTransformer

# 继承
class MySwin(SwinTransformer):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=0, head_dim=32, window_size=7,downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__(hidden_dim=hidden_dim, layers=layers, heads=head，downscaling_factors=downscaling_factors,num_classes=num_classes)
        ……
    # 覆盖父类方法
     def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        ……
        return x
```
-------
## Catalogue
### Config
#### [Config](config/cfg.py)
用于设置工程的超参数。
  * init ([yaml_Path])
    > 可使用字典形式读取参数，e.g. Config()['batchsize']。<br/>
  
  * update (**kw):增改参数。
    > Input: [dict], e.g. *config.update(batchsize=7,epoch=30)*<br/>
    > Output: *None*<br/>
  
  * seed (*args):开启随机数种子。
    > Input: 可多选'np','random','torch'，或'all'。<br/>
    > Output: *None*<br/>
  
#### [config.yaml](config/config.yml)
保存参数的文件。下列内容仅供参考，可自行修改。
  - 工程参数
    + model_name：工程名称
    + train/val/test_data：训练/验证/测试数据集
    + model_save_dir：模型结果保存路径
    + eval_interval：每训练多少步长(step)，跑一次验证集
    + save_interval：每训练多少步长(step)，保存一次结果
    + ……
  - 训练参数
    + img_size：[ch,h,w]
    + batchsize：一般为2的n次方。设定上限由显卡性能决定，大小变化会略微影响训练结果。
    + num_worker：获取数据的子线程个数（并不是越大越好）。
    + epoch：训练周期。……
    + seed：随机数生成种子。
    + device：训练设备。
    + grad_clip：梯度更新的最大值，防止梯度爆炸。
    + optimizer：指定优化器。
    + lr：learning rate，学习率。
    + ……
  - 模型参数
    + patch_size：ViT的patch大小。
    + n_classes：分类器的类别。
    + ……

### Module
#### [VisionTransformer](module/vit.py)
> **VisionTransformer** (in_channels = 3,
                    patch_size = [16,16],
                    img_size = [224,224],
                    num_heads: = 8,
                    num_layers: = 12,
                    dropout: = 0.1,
                    expand_forward = 4,
                    n_classes: = 100,
                    isseq2seq = True)<br/>

#### [FAN](module/fan.py)
> **FAN** (input_channel, output_channel=512, block=BasicBlock, layers=[1, 2, 5, 3])<br/>
常用于文本识别任务，输入图像一般为[b,ch_in,h=32,w=128],输出为[b,ch_out,1,w/4+1]

#### [Resnet](module/resnet.py)
> **Resnet** x ( ) <br/>
x可为18，34，50，101，152。 <br/>
各尺寸resnet有默认对应结构和层数的设定，详见[resnet.py](module/resnet.py)

#### [SwinTransformer](module/swintransformer.py)
> **SwinTransformer**_x (hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs)<br/>
x可为t，s，b，l。<br/>
输入图像尺寸为[b,3,224,224],输出为[b,hidden,7]。 <br/>
也可使用**SwinTransformer**自定义层数，详见[swintransformer.py](module/swintransformer.py)
#### [Transformer](module/transformer.py)
> **Encoder** (embed_size=768, num_heads=8, dropout=0.1, expend=4, num_layers=6)<br/>
> **Decoder** (embed_size=768, num_heads=8, dropout=0.1, expend=4, num_layers=6)<br/>

### Dataset
#### [lmdbDataset](dataset/lmdb_dataset.py)
读取.mdb格式数据集
  * init([Path], [image_size],isnormal=True,transform=None,target_transform=None)
    > Input: *image_size*:[ch,h,w]。后面这堆就默认吧。
    
### Utils
#### [Counter](utils/counter.py)
计数器，用于求每个周期的平均loss（主要给开发者看的）
  * init( )
  * add (value:Tenor)
    > Input: [loss] <br/>
    > Output: *None* <br/>
  * reset ( )
    > Input: *None* <br/>
    > Output: *None* <br/>
  * mean ( )
    > Input: *None* <br/>
    > Output: *None* <br/>

#### [PatchShuffle](utils/patchshuffle.py)
用于图像patch乱序。
  * init(img_size=(3,32,128),patch_size=(16,16),batchsize=1)
  * shuffle (images:Tensor): patch乱序。<br/>
    > Input: *images*可为 [ch,h,w] 或 [b,ch,h,w]<br/>
    > Output: *images_shuffle*: [b,ch,h,w] , *idx*: [b,patch_num]<br/>
  
  * decoder (images_shuffle,idx): 根据idx恢复乱序。<br/>
    > Input: *images_shuffle*:[b,ch,h,w], *idx*: [b,patch_num]<br/>
    > Output: *images*: [b,ch,h,w]
  
  * lookonepic ([path],images_shuffle:Tensor,idx,batchsize_n:int=0,isRGB=True,isSign=False): 将batchsize中第n张图保存在path路径下,用于调试。<br/>
    > Input: *images_shuffle*: [b,ch,h,w]；*isSign*为真时会将patch编号写在图片上；OpenCV读入的图片需将*isRGB*设为False。<br/>
    > Output: *None*<br/>

    
### Output
------
## Extension
1. 文件开头注明贡献者名称和日期。
2. 将模块文件放入module/dataset/utils文件夹下，模块本身必须以类的形式呈现。
3. 对于有多种大小、形式可选的模块，请使用函数的方式实例化对象，并将必要参数以“关键字参数”的形式封装在函数内，参考[resnet.py](module/resnet.py)。
4. 对于模块的子模块，应将子模块根据职能放置在相应（module/dataset/utils）文件夹下，如[ViT的PatchEmbedding](utils/embed.py)。
5. 更新相应文件夹下的__init__.py,格式必须为`from Eureka.文件夹[.子文件夹].模块文件 import 模块名`，参考[module/__init__.py](module/__init__.py)。
6. 更新[readme.md](readme.md)说明文档。
------
## Version Description
### V0.0.1
1. 修正Config类的用法。可通过键值获取参数文件的数据；将seed功能归并到Config类下。
2. Utils增加 PatchShuffle。
3. Module增加 VisionTransformer, FAN, Resnet, SwinTransformer, TransformerEncoder, TransformerDecoder。 
4. 新增 PatchShuffle.decoder,用于恢复patch乱序。
5. Utils新增initialize，尚在构思调试，未投入使用。