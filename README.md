# Graphic Deep Neural Network
在AI学习的漫漫长路上，理解不同文章中的模型与方法是每个人的必经之路，偶尔见到Fjodor van Veen所作的[A mostly complete chart of Neural Networks](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png) 和 FeiFei Li AI课程中对模型的[画法](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/feifei.png)，大为触动。决定将深度神经网络中的一些模型 进行统一的图示，便于大家对模型的理解。


模型中用到的图标规定如下：
![图标](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/cellsreadme.png)

当大家描述网络结构时，常常会将卷积层和maxpooling层画在一起，我们也提供了这样的简化方法

  <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/equal.png" width="205">

一个具体的问题是否能用人工智能，或者更进一步说用深度学习某个算法解决，首先需要人对问题进行分解，提炼成可以用机器解决的问题，譬如说分类问题，回归问题，聚类问题等。

PS： caffe 模型可视化网址 http://ethereon.github.io/netscope/#/editor

## object classification 物体分类

深度学习在解决分类问题上非常厉害。让它声名大噪的也是对于图像分类问题的解决。也产生了很多很经典的模型。因此我们首先了解一下它们。

从模型的发展过程中，我们可以看到网络结构不断的进行改进，包括不断增加的网络深度，不断变换的卷积核上，多种多样的网络模块等。

### LeNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)  Yann LeCun
* LeNet  最经典的CNN网络

    <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/lenet.png" width="405">


    [1] LeCun, Yann; Léon Bottou; Yoshua Bengio; Patrick Haffner (1998). "Gradient-based learning applied to document recognition" [pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

	tf code  https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/lenet.py

	pytorch code  https://github.com/pytorch/examples/blob/master/mnist/main.py

	caffe code  https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    
 
     PyTorch定义了常用模型，并且提供了预训练版本：
		AlexNet: AlexNet variant from the “One weird trick” paper.
		VGG: VGG-11, VGG-13, VGG-16, VGG-19 (with and without batch normalization)
		ResNet: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
		SqueezeNet: SqueezeNet 1.0, and SqueezeNet 1.1
     其中ImageNet比赛中相关的网络，可参见 https://github.com/pytorch/examples/tree/master/imagenet 
     另外也可以参考https://github.com/aaron-xichen/pytorch-playground.git 里面各种网络结构写法 （非官方）
### AlexNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)  Alex Krizhevsky,  Geoffrey Hinton
* AlexNet  2012年，Alex Krizhevsky用AlexNet 在当年的ImageNet图像分类竞赛中(ILSVRC 2012)，以top-5错误率15.3%拿下第一。 他的top-5错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。

   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet.png" width="505">

   [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

   tensorflow 源码 https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/alexnet.py

   caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt


### GoogLeNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md) Christian Szegedy / Google
* GoogLeNet  采用InceptionModule和全局平均池化层，构建了一个22层的深度网络,使得很好地控制计算量和参数量的同时（ AlexNet 参数量的1/12），获得了非常好的分类性能.
它获得2014年ILSVRC挑战赛冠军，将Top5 的错误率降低到6.67%.
GoogLeNet名字将L大写，是为了向开山鼻祖的LeNet网络致敬.

   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/googlenet.png" width="805">

   [3] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)


   tensorflow 源码 https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/inception_v1.py

   caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt


* Inception V3


### VGG [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md) Karen Simonyan , Andrew Zisserman  /  [Visual Geometry Group（VGG）Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
* VGG   
VGG-Net是2014年ILSVRC classification第二名(第一名是GoogLeNet)，ILSVRC localization 第一名。VGG-Net的所有 convolutional layer 使用同样大小的 convolutional filter，大小为 3 x 3

   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/vgg.png" width="805">

单独看VGG19的模型：

   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/vgg19.png" width="805">


   [4] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014). [pdf](https://arxiv.org/pdf/1409.1556.pdf)

   tensorflow 源码: https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/vgg.py


   caffe ：

   vgg16 https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt

   vgg19 https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt

### ResNet 何凯明 [He Kaiming](http://kaiminghe.com/) 
* ResNet 

[7] He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015). [pdf] (ResNet,Very very deep networks, CVPR best paper) 

## Face Detection and Recognition 人脸检测与识别
人脸检测与识别是一个研究很久的课题。传统方法之前也有了很多稳定可行的方法。而深度学习的出现，无论对检测还是识别又有了很大的提升。随着算法和代码的开源，现在很多公司都可以自己搭建一套自己的人脸检测识别系统。那么下面几篇经典论文，都是会需要接触到的。

### MTCNN  乔宇 SIAT

### facenet 



### arcface



## OCR：Optical Character Recognition 字符识别

传统的文本文字检测，主要实现了在文档上的文字检测，并且有了很好的商用。但是场景文字检测一直没有很好的被解决。随着深度学习的发展，近年来相应工作了有了较好的进展。

场景文字检测与传统的文本文字检测的重要区别是需要将照片或视频中的文字识别出来。

其主要分为两个步骤：

对照片中存在文字的区域进行定位（Text Detection)，即找到单词或文本行（word/linelevel）的边界框（bounding box)；

然后对定位后的文字进行识别（Text Recognition)

将这两个步骤合在一起就能得到文字的端到端检测（End-to-end Recognition)

### Text Detection (文字定位)

### CTPN (Connectionist Text Proposal Network)  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CTPN.md)  [Zhi Tian](http://www.ece.mtu.edu/faculty/ztian/),  乔宇 [Qiao Yu](http://mmlab.siat.ac.cn/yuqiao/) / CUHK & SIAT

* CTPN 使用CNN + RNN 进行文本检测与定位。采用Top-down的方式，先找到文本区域，然后找出文本线。

   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ctpn.png" width="905">

作者caffe中模型结构如图：

   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ctpn_caffe.png" width="905">

   [1] [ECCV2016] Detecting Text in Natural Image with Connectionist Text Proposal Network [pdf](https://arxiv.org/pdf/1609.03605.pdf) 


   Caffe 源码：https://github.com/tianzhi0549/CTPN 官方
   


### Text Recognition (文字识别)

### CRNN [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CRNN.md) 白翔 Xiang Bai/Media and Communication Lab, HUST
* CRNN 将特征提取CNN，序列建模 RNN 和转录 CTC 整合到统一框架，完成端对端的识别任务.

   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/crnn.png" width="805">

   [1] [2015-CoRR] An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition  [pdf](http://arxiv.org/pdf/1507.05717v1.pdf) 



   code： http://mclab.eic.hust.edu.cn/~xbai/CRNN/crnn_code.zip 

   Torch 源码：https://github.com/bgshih/crnn Torch7 官方
   pytorch https://github.com/meijieru/crnn.pytorch

## Object Detection 物体检测

物体分类（物体识别）解决的是这个东西是什么的问题（What）。而物体检测则是要解决这个东西是什么，具体位置在哪里（What and Where）。

Christian Szegedy / Google 用AlexNet也做过物体检测的尝试。

   [1] Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "Deep neural networks for object detection." Advances in Neural Information Processing Systems. 2013.  [pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

不过真正取得巨大突破，引发基于深度学习目标检测的热潮的还是RCNN

### RCNN  Ross B. Girshick(RBG) [link](https://people.eecs.berkeley.edu/~rbg/index.html) / UC-Berkeley

* RCNN R-CNN框架，取代传统目标检测使用的滑动窗口+手工设计特征，而使用CNN来进行特征提取。

Traditional region proposal methods + CNN classifier

创新点：将CNN用在物体检测上，提高了检测率。
缺点：每个proposal都卷积一次，重复计算，速度慢。

R-CNN在PASCAL VOC2007上的检测结果提升到66%(mAP)

   [2] SGirshick, Ross, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)

### SPPNet 何凯明 [He Kaiming](http://kaiminghe.com/) /MSRA
* SPPNet
   [3] He, Kaiming, et al. "Spatial pyramid pooling in deep convolutional networks for visual recognition." European Conference on Computer Vision. Springer International Publishing, 2014. [pdf](http://arxiv.org/pdf/1406.4729)

### Fast RCNN Ross B. Girshick
* Fast RCNN
   [4] Girshick, Ross. "Fast r-cnn." Proceedings of the IEEE International Conference on Computer Vision. 2015.

### Faster RCNN 何凯明 [He Kaiming](http://kaiminghe.com/)
* Faster RCNN
将Region Proposal Network和特征提取、目标分类和边框回归统一到了一个框架中。

Faster R-CNN = Region Proposal Network +Fast R-CNN


   [5] Ren, Shaoqing, et al. "Faster R-CNN: Towards real-time object detection with region proposal networks." Advances in neural information processing systems. 2015.

### Yolo
* Yolo
   [6] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." arXiv preprint arXiv:1506.02640 (2015). [pdf] (YOLO,Oustanding Work, really practical)

### SSD
* SSD
   [7] Liu, Wei, et al. "SSD: Single Shot MultiBox Detector." arXiv preprint arXiv:1512.02325 (2015). [pdf] ⭐️⭐️⭐️

### R-FCN
* R-FCN
   [8] Dai, Jifeng, et al. "R-FCN: Object Detection via Region-based Fully Convolutional Networks." arXiv preprint arXiv:1605.06409 (2016). [pdf] 

### Mask R-CNN
* Mask R-CNN
   [9] He, Gkioxari, et al. "Mask R-CNN" arXiv preprint arXiv:1703.06870 (2017). [pdf] 

## Object Segmentation 物体分割

### FCN
[1] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation.” in CVPR, 2015. [pdf]

### U-NET


### SegNet

### DeconvNet

### Deeplab 

### RefineNet

### BlitzNet



# 贡献力量

如果想做出贡献的话，你可以：

帮忙对没有收录的paper进行模型绘制

帮忙进行模型校对等

提出修改建议


