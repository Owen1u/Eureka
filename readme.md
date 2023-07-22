<!--
 * @Descripttion: 
 * @version: 
 * @Contributor: Minjun Lu
 * @Source: Original
 * @LastEditTime: 2023-07-22 15:00:53
-->
# <center>Eureka</center>
<center>网罗天下放失旧闻,略考其行事,综其终始,稽其成败兴坏之纪</center>

## Datasets
- base
- imagenet
- transforms
## Modules
- [ViT](https://pytorch.org/vision/stable/models/vision_transformer.html)
    + 适配任意大小的输入图像和patch size（整除）
    + 可设定 position embedding
- [Focalnet](https://github.com/microsoft/FocalNet)
    + 任意2d卷积的PatchEmbed，use_conv_embed=[kernel_size,padding,stride]
    + num_classes<=0时，不加分类头。
## Models
## Utils
- logger
- meter