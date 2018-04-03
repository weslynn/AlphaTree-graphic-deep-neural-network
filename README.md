# Graphic Deep Neural Network
在AI学习的漫漫长路上，理解不同文章中的模型与方法是每个人的必经之路，偶尔见到Fjodor van Veen所作的[A mostly complete chart of Neural Networks](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png) 和 FeiFei Li AI课程中对模型的[画法](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/feifei.png)，大为触动。决定将深度神经网络中的一些模型 进行统一的图示，便于大家对模型的理解。


模型中用到的图标规定如下：
![图标](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/cellsreadme.png)

当大家描述网络结构时，常常会将卷积层和maxpooling层画在一起，我们也提供了这样的简化方法

  <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/equal.png" width="205">

一个具体的问题是否能用人工智能，或者更进一步说用深度学习某个算法解决，首先需要人对问题进行分解，提炼成可以用机器解决的问题，譬如说分类问题，回归问题，聚类问题等。

PS： caffe 模型可视化网址 http://ethereon.github.io/netscope/#/editor


---------------------------------------------------------------------------------------------------
## object classification 物体分类

深度学习在解决分类问题上非常厉害。让它声名大噪的也是对于图像分类问题的解决。也产生了很多很经典的模型。因此我们首先了解一下它们。

从模型的发展过程中，我们可以看到网络结构不断的进行改进，包括不断增加的网络深度，不断变换的卷积核上，多种多样的网络模块等。

### LeNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)  Yann LeCun

* LeNet  最经典的CNN网络

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/lenet.png" width="405"> </a>

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

  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet.png" width="505"></a>

   [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

   tensorflow 源码 https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/alexnet.py

   caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt


### GoogLeNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md) Christian Szegedy / Google
* GoogLeNet  采用InceptionModule和全局平均池化层，构建了一个22层的深度网络,使得很好地控制计算量和参数量的同时（ AlexNet 参数量的1/12），获得了非常好的分类性能.
它获得2014年ILSVRC挑战赛冠军，将Top5 的错误率降低到6.67%.
GoogLeNet名字将L大写，是为了向开山鼻祖的LeNet网络致敬.

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/googlenet.png" width="805"></a>

   [3] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)


   tensorflow 源码 https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/inception_v1.py

   caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt



### Inception V3  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md) Christian Szegedy / Google
* Inception V3，GoogLeNet的改进版本,采用InceptionModule和全局平均池化层，v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），ILSVRC 2012 Top-5错误率降到3.58% test error 

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/v3-tf.png" width="805"></a>

   [4] Szegedy, Christian, et al. “Rethinking the inception architecture for computer vision.” arXiv preprint arXiv:1512.00567 (2015). [pdf](http://arxiv.org/abs/1512.00567)


   tensorflow 源码 https://github.com/tensorflow/tensorflow/blob/fc1567c78b3746b44aa50373489a767afbb95d2b/tensorflow/contrib/slim/python/slim/nets/inception_v3.py




### VGG [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md) Karen Simonyan , Andrew Zisserman  /  [Visual Geometry Group（VGG）Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
* VGG   
VGG-Net是2014年ILSVRC classification第二名(第一名是GoogLeNet)，ILSVRC localization 第一名。VGG-Net的所有 convolutional layer 使用同样大小的 convolutional filter，大小为 3 x 3

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/vgg.png" width="805"></a>

单独看VGG19的模型：

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/vgg19.png" width="805"></a>


   [5] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014). [pdf](https://arxiv.org/pdf/1409.1556.pdf)

   tensorflow 源码: https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/vgg.py


   caffe ：

   vgg16 https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt

   vgg19 https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt

### ResNet 何凯明 [He Kaiming](http://kaiminghe.com/) 
* ResNet 

[7] He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015). [pdf] (ResNet,Very very deep networks, CVPR best paper) 


-----------------------------------------------------------------------------------------------------------

## Face Detection and Recognition 人脸检测与识别
人脸检测与识别是一个研究很久的课题。传统方法之前也有了很多稳定可行的方法。而深度学习的出现，无论对检测还是识别又有了很大的提升。随着算法和代码的开源，现在很多公司都可以自己搭建一套自己的人脸检测识别系统。那么下面几篇经典论文，都是会需要接触到的。

### MTCNN [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/MTCNN.md) [zhang kaipeng](https://kpzhang93.github.io/) 乔宇 [Qiao Yu](http://mmlab.siat.ac.cn/yuqiao/) / CUHK & SIAT

* MTCNN 
MTCNN 将人脸检测与关键点检测放到了一起来完成。整个任务分解后让三个子网络来完成。每个网络都很浅，使用多个小网络级联，较好的完成任务。

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/MTCNN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mtcnn.png" width="400"></a>


   [1] [ECCV2016] Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks [pdf](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)


   Caffe 源码：https://github.com/kpzhang93/MTCNN_face_detection_alignment 官方

   tensorflow 源码 : https://github.com/davidsandberg/facenet/tree/master/src/align 



### facenet 



### arcface


-------------------------------------------------------------------------------

## OCR：Optical Character Recognition 字符识别 / STR, Scene Text Recognition 场景文字识别

传统的文本文字检测识别，有了很好的商用。但是场景文字检测识别一直没有很好的被解决。随着深度学习的发展，近年来相应工作了有了较好的进展，其主要分为两个步骤：

1.文字定位（Text Detection)，即找到单词或文本行（word/linelevel）的边界框（bounding box)，近些年的难点主要针对场景内的倾斜文字检测。

2.文字识别（Text Recognition)

将这两个步骤合在一起就能得到文字的端到端检测（End-to-end Recognition)


传统常用的方法有：

	MSER(Maximally Stable Extremal Regions)最大稳定极值区域

	Chen H, Tsai S S, Schroth G, et al. Robust text detection in natural images with edge-enhanced maximally stable extremal regions[C]//Image Processing (ICIP), 2011 18th IEEE International Conference on. IEEE, 2011: 2609-2612.

	通过MSER得到文本候选区域，再通过几何和笔划宽度信息滤掉非文本区域， 剩余的文本信息形成文本直线，最终可被切分为单个文字。

	Matlab code：http://cn.mathworks.com/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html

此外Google DeepMind提出了一种新的网络结构，叫做STN(Spatial Transformer Networks) ，可以用在文字校正方面：

   STN可以被安装在任意CNN的任意一层中，相当于在传统Convolution中间，装了一个“插件”，可以使得传统的卷积带有了[裁剪]、[平移]、[缩放]、[旋转]等特性。

   [1] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks[C]//Advances in Neural Information Processing Systems. 2015: 2017-2025.

   https://github.com/skaae/recurrent-spatial-transformer-code


### Text Detection (文字定位)

文字定位分为 如下几类：Proposal-based method ,Segmantation-based method ， Part-based method 和 Hybrid method


### Proposal-based method

### DeepText 金连文

DeepText（此方法不是Google的DeepText哦），对fasterRCNN进行改进用在文字检测上，先用Inception-RPN提取候选的单词区域，再利用一个text-detection网络过滤候选区域中的噪声区域，最后对重叠的box进行投票和非极大值抑制



Deep Matching Prior Network
金连文教授发表在 CVPR2017 上的工作提出了一个重要观点：在生成 proposal 时回归矩形框不如回归一个任意多边形。

理由：这是因为文本在图像中更多的是具有不规则多边形的轮廓。他们在SSD（Single ShotMultiBox Detector）的检测框架基础上，将回归边界框的过程和匹配的过程都加入到网络结构中，取得了较好的识别效果并且兼顾了速度。




现在的方法越来越倾向于从整体上自动处理文本行或者边界框，如 arXiv上的一篇文章就将 Faster R-CNN中的RoI pooling替换为可以快速计算任意方向的操作来对文本进行自动处理。

Arbitrary Oriented Scene Text Detection via Rotation Proposals



TextProposals: a Text-specific Selective Search Algorithm for Word Spotting in the Wild.

这篇文章针对文本的特殊属性，将object proposal 的方法用在了文本检测中，形成了text-proposal。

text-proposal也是基于联通区域的组合，但又与之前的方法有所不同：初始化的区域并不对应单个字符，也不需要知道里面的字符数。

代码见：https://github.com/lluisgomez/TextProposals




Gupta A, et al. Synthetic data for text localisation in natural images. CVPR, 2016.


### CTPN (Connectionist Text Proposal Network)  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CTPN.md)  [Zhi Tian](http://www.ece.mtu.edu/faculty/ztian/),  乔宇 [Qiao Yu](http://mmlab.siat.ac.cn/yuqiao/) / CUHK & SIAT


* CTPN 使用CNN + RNN 进行文本检测与定位。

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CTPN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ctpn.png" width="905"> </a>


作者caffe中模型结构如图：

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CTPN.md"><img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ctpn_caffe.png" width="905"></a>


   [1] [ECCV2016] Detecting Text in Natural Image with Connectionist Text Proposal Network [pdf](https://arxiv.org/pdf/1609.03605.pdf) 


   Caffe 源码：https://github.com/tianzhi0549/CTPN 官方


### TextBoxes  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes.md) 白翔 Xiang Bai/Media and Communication Lab, HUST


* TextBoxes，一个端到端的场景文本检测模型。这个算法是基于SSD来实现的,解决水平文字检测问题，将原来3×3的kernel改成了更适应文字的long conv kernels 3×3 -> 1×5。default boxes 也做了修改。
   
   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/TextBoxes.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/textboxes.png" width="905"> </a>


作者caffe中模型结构如图：

  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/textboxes_caffe.png" width="905"> </a>


   [2]  M. Liao et al. TextBoxes: A Fast Text Detector with a Single Deep Neural Network. AAAI, 2017. [pdf](https://arxiv.org/pdf/1611.06779.pdf) 


   Caffe 源码：https://github.com/MhLiao/TextBoxes 官方




### TextBoxes++ [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes++.md) 白翔 Xiang Bai/Media and Communication Lab, HUST


* TextBoxes++ 这个算法也是基于SSD来实现的，实现了对多方向文字的检测。boundingbox的输出从4维的水平的boundingbox扩展到4+8=12维的输出。long conv kernels 从 1×5 改成了 3×5。default boxes 也做了修改。
   
   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes++.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/textboxes++.png" width="905"> </a>



   [3]  M. Liao et al. TextBoxes++: Multi-oriented text detection [pdf](https://arxiv.org/pdf/1801.02765.pdf)



   Caffe 源码：https://github.com/MhLiao/TextBoxes_plusplus 官方


### Segmantation-based method:

### FCN_Text
基于分割的方法，使用FCN来做，
将文本行视为一个需要分割的目标，通过分割得到文字的显著性图像（salience map），这样就能得到文字的大概位置、整体方向及排列方式，再结合其他的特征进行高效的文字检测。

[CVPR2016]Zhang Z, et al.Multi-Oriented Text Detection with Fully Convolutional Networks,CVPR, 2016. [pdf](http://mc.eistar.net/UpLoadFiles/Papers/TextDectionFCN_CVPR16.pdf)

caffe torch code :https://github.com/stupidZZ/FCN_Text 


这篇文章将局部和全局信息结合，使用了一种coarse-to-fine的方法来定位自然场景中的文本。首先，使用了全卷积的神经网络来训练和预测文字区域的显著图；然后，结合显著图和文字元素来估计文字所在的直线；最后，另一个全卷积模型的分类器用来估计每个字符的中心，从而去掉误检区域。这个系统能够处理不同方向、语言、字体的文本检测，在MSRA-TD500, ICDAR2015和ICDAR2013的评测集上都取得了state-of-the-art的结果。



### Scene Text Detection Via Holistic multi-channel Prediction
发现在卷积神经网络中可以同时预测字符的位置及字符之间的连接关系，这些特征对定位文字具有很好的帮助。其过程如下：

得到文字文本行的分割结果；

得到字符中心的预测结果；

得到文字的连接方向。

通过得到的这三种特征构造连通图(graph)，然后对图进行逐边裁剪来得到文字位置。



### Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation




### Part-based method:
对于多方向文字检测的问题，回归或直接逼近bounding box的方法难度都比较大，所以考虑使用 part-based model 对多方向文字进行处理。



### SegLink

将文字视为小块单元。对文字小块同时进行旋转和回归。并且通过对文字小块之间的方向性进行计算来学习文字之间的联系，最后通过简单的后处理就能得到任意形状甚至具有形变的文字检测结果。

例如，对于那些很长的文本行，其卷积核的尺寸难以控制，但是如果将其分解为局部的文字单元之后就能较好地解决。

SegLink+ CRNN 在ICDAR 2015上得到了当时最好的端到端识别效果。

B. Shi et al. Detecting Oriented Text in Natural Images by Linking Segments. IEEE CVPR, 2017.

Code: https://github.com/bgshih/seglink


### Hybrid method
最近有些方法同时使用分割（segmentation）和边界框回归（bounding box regression）的方式对场景文字进行检测。

如 CVPR2017 上的一篇文章使用PVANet对网络进行优化、加速，并输出三种不同的结果：

边缘部分分割的得分（score）结果；

可旋转的边界框（rotated bounding boxes）的回归结果；

多边形bounding boxes（quadrangle bounding boxes）的结果。

同时对非极大值抑制（NMS）进行改进，得到了很好的效果。


He W, et al. Deep Direct Regression for Multi-Oriented Scene Text Detection. ICCV, 2017

arXiv上的一篇文章使用了相似的思想：一个分支对图像分割进行预测，另一个分支对边界框（bounding box）进行预测，最后利用经过改进的非极大抑制（Refined NMS）进行融合。







### Text Recognition (文字识别)


###Word/Char Lever
通过多类分类器，如word 分类器，char 分类器 来判断。每一类都是一个word 或者char。


M. Jaderberg et al. Reading text in the wild with convolutional neural networks. IJCV, 2016.


### Sequence Level
从图片获取Sequence feature，然后通过RNN + CTC 
B. Su et al. Accurate scene text recognition based on recurrent neural network. ACCV, 2014.

He et al. Reading Scene Text in Deep Convolutional Sequences. AAAI, 2016.

### CRNN [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CRNN.md) 白翔 Xiang Bai/Media and Communication Lab, HUST
* CRNN 将特征提取CNN，序列建模 RNN 和转录 CTC 整合到统一框架，完成端对端的识别任务.
   
   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CRNN.md">
   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/crnn.png" width="805"></a>

   [1] [2015-CoRR] An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition  [pdf](http://arxiv.org/pdf/1507.05717v1.pdf) 



   code： http://mclab.eic.hust.edu.cn/~xbai/CRNN/crnn_code.zip 

   Torch 源码：https://github.com/bgshih/crnn Torch7 官方
   pytorch https://github.com/meijieru/crnn.pytorch

### RARE 白翔 Xiang Bai/Media and Communication Lab, HUST

用STN 加上 SRN 解决弯曲形变的文字识别问题

SRN: an attention-based encoder-decoder framework

Encoder: ConvNet + Bi-LSTM

Decoder: Attention-based character generator

   [2]Shi B, Wang X, Lv P, et al. Robust Scene Text Recognition with Automatic Rectification[J]. arXiv preprint arXiv:1603.03915, 2016. [paper](https://arxiv.org/pdf/1603.03915v2.pdf)



### End to End 端到端文字检测与识别

[2016-IJCV]Reading Text in the Wild with Convolutional Neural Networks [pdf](https://arxiv.org/pdf/1412.1842.pdf)
较早的端到端识别研究是VGG 组发表在 IJCV2016中的一篇文章，其识别效果很好，并且在两年内一直保持领先地位。

这篇文章针对文字检测问题对R-CNN进行了改造：

通过edge box或其他的handcraft feature来计算proposal；

然后使用分类器对文本框进行分类，去掉非文本区域；

再使用 CNN对文本框进行回归来得到更为精确的边界框（bounding box regression）；

最后使用一个文字识别算法进一步滤除非文本区域。



VGG组在CVPR2016上又提出了一个很有趣的工作。文章提出文本数据非常难以标注，所以他们通过合成的方法生成了很多含有文本信息的样本。虽然图像中存在合成的文字，但是依然能得到很好的效果。


### Deep TextSpotter
[ICCV2017] Lukas Neumann ,Deep TextSpotter：An End-to-End Trainable Scene Text Localization and Recognition Framework [pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busta_Deep_TextSpotter_An_ICCV_2017_paper.pdf)


https://github.com/MichalBusta/DeepTextSpotter

该方法将文字检测和识别整合到一个端到端的网络中。检测使用YOLOv2+RPN，并利用双线性采样将文字区域统一为高度一致的变长特征序列，再使用RNN+CTC进行识别。

-----------------------------------------------------------------------------

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

### SSD(The Single Shot Detector) [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20detection%20%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B/SSD.md)

* SSD SSD是一种直接预测bounding box的坐标和类别的object detection算法，没有生成proposal的过程。它使用object classification的模型作为base network，如VGG16网络，

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20detection%20%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B/SSD.md"><img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ssd.png" width="805"></a>

   [7] Liu, Wei, et al. "SSD: Single Shot MultiBox Detector." arXiv preprint arXiv:1512.02325 (2015). [pdf](https://arxiv.org/pdf/1512.02325.pdf)  


   tensorflow 源码 https://github.com/balancap/SSD-Tensorflow/blob/master/nets/ssd_vgg_300.py

   caffe ：https://github.com/weiliu89/caffe/tree/ssd



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


----------------------------------------------------------------------------------

## Datasets 数据库

 Incidental Scene Text dataset

 Incidental Scene Text dataset 是 ICDAR2015竞赛中使用的数据集，是很常用的英文文字检测数据集。

它涵盖1000张训练图片（约包含4500个单词）和500张测试图片；

它重点采集了一些随机场景，在这些场景中文字具有方向任意、字体小、低像素的特性。


MSRA-TD500

MSRA-TD500数据集含有英文和中文两种语言。包含了500张自然图片（涵盖室内、室外采集）；

包含中文、英文及中英混合形式，具有不同的字体、大小、颜色、方向；
文本边框标注；


Ref. Detecting texts of arbitrary orientations in natural images,CVPR2012

RCTW-17

ICDAR 2017中文场景文字检测比赛使用了中文数据集RCTW-17，包含中文文本的图片共12034张（其中8034张训练图片，4000张测试图片）；

图片涵盖汉字、数字、英文单词，其中汉字占最大比例；

链接：http://mclab.eic.hust.edu.cn/icdar2017chinese/


ICDAR2017 Competition on Reading Chinese Text in the Wild ( RCTW-17). ICDAR’17
Dataset : http://mclab.eic.hust.edu.cn/icdar2017chinese

场景文字识别
SynthText in the Wild Dataset(41G) http://www.robots.ox.ac.uk/~vgg/data/scenetext/

# 贡献力量

如果想做出贡献的话，你可以：

帮忙对没有收录的paper进行模型绘制

帮忙进行模型校对等

提出修改建议


