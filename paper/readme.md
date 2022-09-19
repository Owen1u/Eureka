Review of Novel Models & Benchmark Datasets
======
![](title.png)
<!-- TOC -->

- [Review of Novel Models & Benchmark Datasets](#review-of-novel-models--benchmark-datasets)
  - [经典咏流传](#经典咏流传)
    - [Resnet](#resnet)
    - [Transformer](#transformer)
  - [计算机视觉 (Computer Vision)](#计算机视觉-computer-vision)
    - [目标识别 (Object Detection)](#目标识别-object-detection)
    - [图像分类 (Image Classification)](#图像分类-image-classification)
    - [语义分割 (Semantic Segmentation)](#语义分割-semantic-segmentation)
    - [文本识别 (Text Spotting)](#文本识别-text-spotting)
    - [交通标志识别 (Traffic Sign Recognition)](#交通标志识别-traffic-sign-recognition)
  - [自然语言处理 (Natural Language Processing)](#自然语言处理-natural-language-processing)
    - [语言模型 (Language Modelling)](#语言模型-language-modelling)
    - [命名实体识别 (Named( Entity Recognition)](#命名实体识别-named-entity-recognition)
    - [情感分析 (Sentiment Analysis)](#情感分析-sentiment-analysis)

<!-- /TOC -->

## 经典咏流传
### Resnet
### Transformer

---
## 计算机视觉 (Computer Vision)
### 目标识别 (Object Detection)
### 图像分类 (Image Classification)
### 语义分割 (Semantic Segmentation)
### 文本识别 (Text Spotting)
|数据集|年份|机构|真实性|语种|数据量|论文链接|
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|BCTR|2021|复旦大学|synthetic only for *DOCUMENT*|中英文|[详见readme](https://github.com/FudanVI/benchmarking-chinese-text-recognition)|[论文](https://arxiv.org/pdf/2112.15093.pdf)|

|模型|年份|机构|Pretraining|En/Ch|是否开源|论文链接|
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|MaskOCR|2022|百度|Yes|Both|Not-Yet|[论文](https://arxiv.org/pdf/2206.00311.pdf)|
|ConCLR|2022|香港中文大学|Yes|English|Not-Yet|[下载PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20245/20004)|
|ABINet|2021|中国科学技术大学|Yes|English|Yes|[论文](https://arxiv.org/pdf/2103.06495.pdf)|
|SEED|2020|中国科学院<br/>中国科学院大学|No|Both|Yes|[论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_SEED_Semantics_Enhanced_Encoder-Decoder_Framework_for_Scene_Text_Recognition_CVPR_2020_paper.pdf)|
|SRN|2020|中国科学院大学<br/>百度<br/>中国科学院|No|Both|Yes|[论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.pdf)|
|MORAN|2019|华南理工大学|No|Both|Yes|[论文](https://arxiv.org/pdf/1901.03003.pdf)|
|ASTER|2018|华中科技大学|No|Both|Yes|[论文](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8395027&tag=1)|
|CRNN|2017|华中科技大学|No|Both|Yes|[论文](https://arxiv.org/pdf/1507.05717.pdf)|



### 交通标志识别 (Traffic Sign Recognition)
**<center>交通标志</center>**

|数据集|年份|机构|真实/合成|国家/地区|图片数量|像素分辨率|类别|论文链接|下载链接|
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|TT100K|2016/2021|清华大学<br/>腾讯|real-world|中国|100K|2048*2048|182/232|[论文](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf)|[点击下载](http://cg.cs.tsinghua.edu.cn/traffic-sign/)|
|Mapillary Traffic Sign Dataset|2020|Mapillary|real-world|全球|52K(fully-annotated)<br/>48K(partially-annotated)|-|313|[论文](https://arxiv.org/pdf/1909.04422.pdf)|[点击下载](www.mapillary.com/dataset/trafficsign)|
|CCTSDB|2017|长沙理工大学|real-world|中国|15.7K|1000*350|3|[论文](https://www.mdpi.com/1999-4893/10/4/127/htm)|[点击下载](https://github.com/csust7zhangjm/CCTSDB)|
|KUL Belgium Traffic Sign dataset|2013|苏黎世联邦理工大学|real-world|比利时|145K|1628*1236||[论文](https://btsd.ethz.ch/shareddata/publications/Timofte-MVA-2011-preprint.pdf)|[点击下载](http://people.ee.ethz.ch/~timofter/traffic_signs/)|
|GTSRB|2012|波鸿鲁尔大学<br/>哥本哈根大学|real-world|德国|50K|1360*1024|43|[论文](https://doi.org/10.1016/j.neunet.2012.02.016)|[点击下载](https://benchmark.ini.rub.de/gtsrb_news.html)
|BelgiumTS|2013|-|real-world|比利时|4.5K+2.5K|200*200以下(only for recognition)|62|[参考论文](https://people.ee.ethz.ch/~timofter/publications/Mathias-IJCNN-2013.pdf)|[点击下载](https://btsd.ethz.ch/shareddata/)|

**<br/><center>信号灯</center>**
|数据集|年份|机构|真实/合成|国家/地区|图片数量|像素分辨率|类别|论文链接|下载链接|
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|DriveU Traffic Light Datase|2018|乌尔姆大学|real-world|德国|232K|2048*1024|344|[论文](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460737)|[点击下载](https://www.uni-ulm.de/in/iui-drive-u/projekte/driveu-traffic-light-dataset/registrierungsformular-dtld/)|
|Bosch Small Traffic Lights|2017|Bosch|real-world|旧金山湾区|13.4K|1280*720|15(按颜色)|[论文](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989163)|[点击下载](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)|

---
## 自然语言处理 (Natural Language Processing)
### 语言模型 (Language Modelling)
|  模型   |  年份  |  作者  |  预训练  |  开源下载  |  论文  |
| :----: | :----: | :----: | :---: | :---: | :---: |
| GPT-3  | 2020 | OpenAI  | 是 | - | [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165v4.pdf)  |
|RoBERTa|2019|Facebook AI<br/>华盛顿大学|是|[Hugging Face平台](https://huggingface.co/models)|[A Robustly Optimized BERT](https://arxiv.org/pdf/1907.11692.pdf)|
<!-- | 单元格  | 单元格 | 单元格 | 单元格 | 单元格 | -->
### 命名实体识别 (Named( Entity Recognition)
### 情感分析 (Sentiment Analysis)

---
