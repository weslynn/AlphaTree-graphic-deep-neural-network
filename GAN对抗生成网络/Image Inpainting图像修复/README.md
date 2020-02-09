# GANs Applications in CV

## 3.4  Image Inpainting(图像修复)/Image Outpainting(图像拓展)/图像融合

图像修复传统算法：PatchMatch  PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing 、
 Space-Time completion of Image，据说Adobe ps cs5 中使用作为图像填充。不断迭代的用相似的非孔区域图片块来替换孔区域图片块。 优点是一张图片就行。缺点：慢，非语义


 Fast Marching (OpenCV)

深度学习有了不一样的发展:


| Paper | Source | Code/Project Link  |
| --- | --- | --- |
|Context-Encoders:Feature Learning by Inpainting|CVPR 2016|[code](https://github.com/pathak22/context-encoder​)|
|High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis|CVPR 2017|[code](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting​) [paper](https://arxiv.org/pdf/1611.09969.pdf)|
|Semantic Image Inpainting with Perceptual and Contextual Losses| CVPR2017|[code](https://github.com/bamos/dcgan-completion.tensorflow)|
|On-Demand Learning for Deep Image Restoration| ICCV 2017|[code](https://github.com/rhgao/on-demand-learning​)|
|Globally and Locally Consistent Image Completion | SIGGRAPH 2017 |[code](https://github.com/satoshiiizuka/siggraph2017_inpainting) [code](https://github.com/shinseung428/GlobalLocalImageCompletion_TF)|
|Deep image prior |2017.12|[code](https://dmitryulyanov.github.io/deep_image_prior)|
|Image Inpainting for Irregular Holes Using Partial Convolutions​| ICLR 2018| [code](https://github.com/deeppomf/DeepCreamPy) [paper](https://arxiv.org/pdf/1804.07723.pdf)|
|Deepfill v1:Generative Image Inpainting with Contextual Attention|CVPR 2018| [code](https://github.com/JiahuiYu/generative_inpainting​)|
|Deepfill v2:Free-Form Image Inpainting with Gated Convolution||[paper](https://arxiv.org/abs/1806.03589)|
|Shift-Net: Image Inpainting via Deep Feature Rearrangement |||
|Contextual-based Image Inpainting|ECCV 2018|[paper](https://arxiv.org/abs/1711.08590v3)|
|Image Inpainting via Generative Multi-column Convolutional Neural Networks| NIPS 2018| [code](https://github.com/shepnerd/inpainting_gmcnn​)|
|PGN：Semantic Image Inpainting with Progressive Generative Networks| ACM MM 2018|[code](https://github.com/crashmoon/Progressive-Generative-Networks​)|
|EdgeConnect|2019|[code](https://github.com/knazeri/edge-connect)|
|MUSICAL: Multi-Scale Image Contextual Attention Learning for Inpainting|IJCAI 2019|[link](sigma.whu.edu.cn)|
|Coherent Semantic Attention for Image Inpainting|2019|[code](https://github.com/KumapowerLIU)|
|Foreground-aware Image Inpainting|CVPR 2019|[pdf](https://arxiv.org/abs/1901.05945v1)|
|Pluralistic Image Completion|CVPR2019|[code](https://github.com/lyndonzheng/Pluralistic-Inpainting) [paper](https://arxiv.org/abs/1903.04227​)|


## Deep image prior

项目主页：https://dmitryulyanov.github.io/deep_image_prior

github链接：https://github.com/DmitryUlyanov/deep-image-prior 

## Partial Conv：IMAGE INPAINTING Nvidia

Partial Convolution based Padding 
Guilin Liu, Kevin J. Shih, Ting-Chun Wang, Fitsum A. Reda, Karan Sapra, Zhiding Yu, Andrew Tao, Bryan Catanzaro 
NVIDIA Corporation 
Technical Report (Technical Report) 2018

Image Inpainting for Irregular Holes Using Partial Convolutions 
Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro 
NVIDIA Corporation 
In The European Conference on Computer Vision (ECCV) 2018 

号称秒杀PS的AI图像修复神器，来自于Nvidia 研究团队。引入了局部卷积，只对部分区域做卷积，而破损区域置0,`能够修复任意非中心、不规则区域

在此之前基于深度学习的方法缺点： 
- 有孔区域需要设置初始值，深度学习网络会混淆那些初始值，以为是非孔区域数据。
- 需要后期要处理
- 只能处理规则孔

https://arxiv.org/pdf/1804.07723.pdf

https://www.nvidia.com/en-us/research/ai-playground/?ncid=so-twi-nz-92489DeepCreamPy

Partial Convolution based Padding.
https://github.com/NVIDIA/partialconv

deeppomf 开源了 Image Inpainting for Irregular Holes Using Partial Convolutions 的修复实现DeepCreamPy


预构建模型下载地址：https://github.com/deeppomf/DeepCreamPy/releases

预训练模型地址：https://drive.google.com/open?id=1byrmn6wp0r27lSXcT9MC4j-RQ2R04P1Z

https://github.com/deeppomf/DeepCreamPy



拓展：走红网络的一键生成裸照软件DeepNude，原站已关，延伸： https://github.com/yuanxiaosc/DeepNude-an-Image-to-Image-technology


## DeepFill
 
论文链接：https://arxiv.org/abs/1801.07892

github链接：https://github.com/JiahuiYu/generative_inpainting    

V2:《Free-Form Image Inpainting with Gated Convolution》

论文链接：https://arxiv.org/abs/1806.03589


## EdgeConnect：使用对抗边缘学习进行生成图像修复


https://github.com/knazeri/edge-connect

https://github.com/youyuge34/Anime-InPainting

## Foreground-aware Image Inpainting Adobe 

https://arxiv.org/abs/1901.05945v1

[Adobe放出P图新研究：就算丢了半个头，也能逼真复原](https://tech.sina.com.cn/csj/2019-01-22/doc-ihrfqziz9984559.shtml)


## Noise2Noise ：医学

2018年ICML 
将此项技术应用于含有大量噪声的图像，比如天体摄影、核磁共振成像（MRI）以及大脑扫描图像等。

使用来自IXI数据集近5000张图像来训练Noise2Noise的MRI图像去噪能力。在没有人工噪声的情况下，结果可能比原始图像稍微模糊一些，但仍然很好地还原了清晰度。


论文链接：https://arxiv.org/pdf/1803.04189.pdf

----------------------------------------------------------------------------------------

## Deep Flow-Guided 视频修复/视频内容消除


原文链接

https://nbei.github.io/video-inpainting.html

论文地址

https://arxiv.org/abs/1905.02884?context=cs

Github 开源地址

https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting

[zhihu -量子位](https://zhuanlan.zhihu.com/p/73645545)


--------------------------------------------------------------------------------------------

## Painting Outside the Box: Image Outpainting 

https://cs230.stanford.edu/projects_spring_2018/posters/8265861.pdf

https://github.com/bendangnuksung/Image-OutPainting



---------------------------------------------------
## GP-GAN

GP-GAN，目标是将直接复制粘贴过来的图片，更好地融合进原始图片中，做一个 blending 的事情。

这个过程非常像 iGAN，也用到了类似 iGAN 中的一些约束，比如 color constraint。另一方面，这个工作也有点像 pix2pix，因为它是一种有监督训练模型，在 blending 的学习过程中，会有一个有监督目标和有监督的损失函数。

2017 https://arxiv.org/pdf/1703.07195.pdf
 
## Deep Painterly Harmonization
https://github.com/luanfujun/deep-painterly-harmonization
这个算法将你选择的外部物体添加到了任意一张图像上，并成功让它看上去好像本来就应该在那里一样。你不妨查看这个代码，然后尝试亲自到一系列不同的图像上去操作这项技术。

