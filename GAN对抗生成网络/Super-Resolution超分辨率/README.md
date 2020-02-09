# GANs Applications in CV

## 3.2 超分辨率 (Super-Resolution)

超分辨率的问题研究由来已久，其目标是将低分辨率图像恢复或重建为高分辨率图像，随着GAN的发展，使得这个问题有了惊人的进展。这项技术得以广泛应用于卫星和航天图像分析、医疗图像处理、压缩图像/视频增强及手机摄像领域，有着明确商业用途。SR技术存在一个有趣的“悖论”，即还原或重建后的高分辨率图像与原图相似度越高，则肉眼观察清晰度越差；反之，若肉眼观察清晰度越好，则图像的失真度越高。导致这一现象的原因在于畸变（Distortion）参数和感知（Perception）参数之间侧重点选择的不同。

传统方法有Google发布的 RAISR: Rapid and Accurate Image Super Resolution(2016 [paper](https://arxiv.org/pdf/1606.01299.pdf) )
国内也同期都发布了自己的算法，如腾讯发布的TSR(Tencent Super Resolution），华为的HiSR等。

超分辨率的比赛 为 NTIRE

客观指标：Peak signal-to-noise ratio (PSNR)

主观指标：在纯的超分辨领域，评价性能的指标是 PSNR（和 MSE 直接挂钩），所以如果单纯看 PSNR 值可能还是 L2 要好。如果考虑主观感受的话估计 L1 要好。

Others’ Collection：其他人收集的相关工作：

https://github.com/icpm/super-resolution

https://github.com/YapengTian/Single-Image-Super-Resolution

https://github.com/huangzehao/Super-Resolution.Benckmark

https://github.com/ChaofWang/Awesome-Super-Resolution

SR可分为两类:从多张低分辨率图像重建出高分辨率图像和从单张低分辨率图像重建出高分辨率图像。基于深度学习的SR，主要是基于单张低分辨率的重建方法，即Single Image Super-Resolution (SISR)。

SISR是一个逆问题，对于一个低分辨率图像，可能存在许多不同的高分辨率图像与之对应，因此通常在求解高分辨率图像时会加一个先验信息进行规范化约束。在传统的方法中，这个先验信息可以通过若干成对出现的低-高分辨率图像的实例中学到。而基于深度学习的SR通过神经网络直接学习分辨率图像到高分辨率图像的端到端的映射函数。

较新的基于深度学习的SR方法，包括SRCNN，DRCN, ESPCN，VESPCN和SRGAN等。(SRCNN[1]、FSRCNN[2]、ESPCN[3]、VDSR[4]、EDSR[5]、SRGAN[6])


1. (SRCNN) Image super-resolution using deep convolutional networks

2. (DRCN) Deeply-recursive convolutional network for image super-resolution

3. (ESPCN) Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network

4. (VESPCN) Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation 

5. Spatial transformer networks 

6. Photo-realistic single image super-resolution using a generative adversarial network (SRGAN)


3.2.1 单张图像超分辨率（Single Image Super-Resolution)

Title	Co-authors	Publication	Links
|:---:|:---:|:---:|:---:|

## 

StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks	Zhang & et al.	ICCV 2017

GAN 对于高分辨率图像生成一直存在许多问题，层级结构的 GAN 通过逐层次，分阶段生成，一步步提生图像的分辨率。典型的使用多对 GAN 的模型有 StackGAN，GoGAN。使用单一 GAN，分阶段生成的有 ProgressiveGAN。



## SRGAN 
SRGAN，2017 年 CVPR 中备受瞩目的超分辨率论文，把超分辨率的效果带到了一个新的高度，而 2017 年超分大赛 NTIRE 的冠军 EDSR 也是基于 SRGAN 的变体。

SRGAN 是基于 GAN 方法进行训练的，有一个生成器和一个判别器，判别器的主体使用 VGG19，生成器是一连串的 Residual block 连接，同时在模型后部也加入了 subpixel 模块，借鉴了 Shi et al 的 Subpixel Network 思想，重点关注中间特征层的误差，而不是输出结果的逐像素误差。避免了生成的高分辨图像缺乏纹理细节信息问题。让图片在最后面的网络层才增加分辨率，提升分辨率的同时减少计算资源消耗。

胡志豪提出一个来自工业界的问题
在实际生产使用中，遇到的低分辨率图片并不一定都是 PNG 格式的（无损压缩的图片复原效果最好），而且会带有不同程度的失真（有损压缩导致的 artifacts）。很多算法包括SRGAN、EDSR、RAISR、Fast Neural Style 等等都没法在提高分辨率的同时消除失真。
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (https://arxiv.org/abs/1609.04802, 21 Nov, 2016)这篇文章将对抗学习用于基于单幅图像的高分辨重建。基于深度学习的高分辨率图像重建已经取得了很好的效果，其方法是通过一系列低分辨率图像和与之对应的高分辨率图像作为训练数据，学习一个从低分辨率图像到高分辨率图像的映射函数，这个函数通过卷积神经网络来表示。

得益于 GAN 在超分辨中的应用，针对小目标检测问题，可以通过 GAN 生成小目标的高分辨率图像从而提高目标检测精度

TensorFlow 版本：https://github.com/buriburisuri/SRGAN

Torch 版本：https://github.com/leehomyc/Photo-Realistic-Super-Resoluton

Keras 版本：https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks


## ESRGAN 

ECCV 2018收录，赢得了PIRM2018-SR挑战赛的第一名。


其他应用 ：
Google 马赛克去除 ( Pixel Recursive Super Resolution https://arxiv.org/abs/1702.00783)
