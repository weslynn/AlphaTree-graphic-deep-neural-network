
# The Single Shot Detector（SSD）
SSD是一种直接预测bounding box的坐标和类别的object detection算法，没有生成proposal的过程。它使用object classification的模型作为base network，如VGG16网络，

在特征图的每个位置预测K个box。对于每个box，预测C个类别得分，以及相对于default bounding box的4个偏移值，在m×n的特征图上将产生(C+4)×k×m×n 个预测值。在caffe代码中，C=21，因此输出的类别维度是8732×21 = 183372 ，boundingbox偏移值维度是8732×4 = 34928


paper： Wei Liu, et al. “SSD: Single Shot MultiBox Detector.” . [pdf](https://arxiv.org/pdf/1512.02325.pdf)  

SSD 结构如图：

![ssd](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/ssd.jpg)




用不同节点表示如图，


![ssd](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/objdetection/ssd.png)



<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/objdetection/ssd.png)</p>



预测的类别 和偏移值 计算如图

![ssdcal](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/objdetection/ssd_cal.png)


整个SSD完整对应如图


![ssd_total](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/objdetection/ssd_total.png)



<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/objdetection/ssd_total.png)</p>




源码：

tensorflow 源码 https://github.com/balancap/SSD-Tensorflow/blob/master/nets/ssd_vgg_300.py

caffe ：https://github.com/weiliu89/caffe/tree/ssd

可参考的ppt : https://docs.google.com/presentation/d/1rtfeV_VmdGdZD5ObVVpPDPIODSDxKnFSU0bsN_rgZXc/edit#slide=id.g179f601b72_0_51

intro: ECCV 2016 Oral
arxiv: http://arxiv.org/abs/1512.02325
paper: http://www.cs.unc.edu/~wliu/papers/ssd.pdf
slides: http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf
github(Official): https://github.com/weiliu89/caffe/tree/ssd
video: http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973
github: https://github.com/zhreshold/mxnet-ssd
github: https://github.com/zhreshold/mxnet-ssd.cpp
github: https://github.com/rykov8/ssd_keras
github: https://github.com/balancap/SSD-Tensorflow
github: https://github.com/amdegroot/ssd.pytorch
github(Caffe): https://github.com/chuanqi305/MobileNet-SSD
What’s the diffience in performance between this new code you pushed and the previous code? #327

https://github.com/weiliu89/caffe/issues/327

DSSD : Deconvolutional Single Shot Detector

intro: UNC Chapel Hill & Amazon Inc
arxiv: https://arxiv.org/abs/1701.06659
github: https://github.com/chengyangfu/caffe/tree/dssd
github: https://github.com/MTCloudVision/mxnet-dssd
demo: http://120.52.72.53/www.cs.unc.edu/c3pr90ntc0td/~cyfu/dssd_lalaland.mp4
Enhancement of SSD by concatenating feature maps for object detection

intro: rainbow SSD (R-SSD)
arxiv: https://arxiv.org/abs/1705.09587
Context-aware Single-Shot Detector

keywords: CSSD, DiCSSD, DeCSSD, effective receptive fields (ERFs), theoretical receptive fields (TRFs)
arxiv: https://arxiv.org/abs/1707.08682
Feature-Fused SSD: Fast Detection for Small Objects

https://arxiv.org/abs/1709.05054

FSSD: Feature Fusion Single Shot Multibox Detector

https://arxiv.org/abs/1712.00960

Weaving Multi-scale Context for Single Shot Detector

intro: WeaveNet
keywords: fuse multi-scale information
arxiv: https://arxiv.org/abs/1712.03149
Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network

keywords: ESSD
arxiv: https://arxiv.org/abs/1801.05918
Tiny SSD: A Tiny Single-shot Detection Deep Convolutional Neural Network for Real-time Embedded Object Detection

https://arxiv.org/abs/1802.06488

MDSSD: Multi-scale Deconvolutional Single Shot Detector for small objects

intro: Zhengzhou University
arxiv: https://arxiv.org/abs/1805.07009
Accurate Single Stage Detector Using Recurrent Rolling Convolution

intro: CVPR 2017. SenseTime
keywords: Recurrent Rolling Convolution (RRC)
arxiv: https://arxiv.org/abs/1704.05776
github: https://github.com/xiaohaoChen/rrc_detection
Residual Features and Unified Prediction Network for Single Stage Detection

https://arxiv.org/abs/1707.05031

# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)