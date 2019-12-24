## Object Classification 物体分类

深度学习在解决分类问题上非常厉害。让它声名大噪的也是对于图像分类问题的解决。也产生了很多很经典的模型。其他方向的模型发展很多都是源于这各部分，它是很多模型的基础工作。因此我们首先了解一下它们。


![object classification](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/ObjectClassification.png)


从模型的发展过程中，随着准确率的提高，网络结构也在不断的进行改进，现在主要是两个方向，一是深度，二是复杂度。此外还有卷积核的变换等等。

深度神经网络的发展要从经典的LeNet模型说起，那是1998年提出的一个模型，在手写数字识别上达到商用标准。之后神经网络的发展就由于硬件和数据的限制，调参的难度等各种因素进入沉寂期。

到了2012年，Alex Krizhevsky 设计了一个使用ReLu做激活函数的AlexNet 在当年的ImageNet图像分类竞赛中(ILSVRC 2012)，以top-5错误率15.3%拿下第一。 他的top-5错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。而且网络针对多GPU训练进行了优化设计。从此开始了深度学习的黄金时代。

大家发表的paper一般可以分为两大类，一类是网络结构的改进，一类是训练过程的改进，如droppath，loss改进等。

之后网络结构设计发展主要有两条主线，一条是Inception系列（即上面说的复杂度），从GoogLeNet 到Inception V2 V3 V4，Inception ResNet。 Inception module模块在不断变化，一条是VGG系列（即深度），用简单的结构，尽可能的使得网络变得更深。从VGG 发展到ResNet ，再到DenseNet ，DPN等。 

最终Google Brain用500块GPU训练出了比人类设计的网络结构更优的网络NASNet,最近训出了mNasNet。

此外，应用方面更注重的是，如何将模型设计得更小，这中间就涉及到很多卷积核的变换。这条路线则包括 SqueezeNet，MobileNet V1 V2 Xception shuffleNet等。

ResNet的变种ResNeXt 和SENet 都是从小模型的设计思路发展而来。


![allmodel](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/allmodel.png)




|模型名 |AlexNet |ZFNet|VGG |GoogLeNet |ResNet|
|:---:|:---:|:---:|:---:|:---:|:---:|
|初入江湖  |2012 |2013 |2014  |2014  |2015|
|层数  |8 |8 |19 |22  |152|
|Top-5错误 |16.4% |11.2%|7.3%  |6.7%  |3.57%|
|Data Augmentation |+ |+ |+ |+ |+|
|Inception(NIN)  |– |– |– |+ |–|
|卷积层数  |5 |5 |16  |21  |151|
|卷积核大小 |11,5,3 | 7,5,3| 3 |7,1,3,5 |7,1,3|
|全连接层数 |3 |3 |3 |1 |1|
|全连接层大小  |4096,4096,1000|4096,4096,1000|4096,4096,1000  |1000  |1000|
|Dropout |+|+|+ |+ |+|
|Local Response Normalization  |+|+|– |+ |–|
|Batch Normalization |–|–|– |– |+|

ILSVRC2016
2016 年的 ILSVRC，来自中国的团队大放异彩：

CUImage（商汤和港中文），Trimps-Soushen（公安部三所），CUvideo（商汤和港中文），HikVision（海康威视），SenseCUSceneParsing（商汤和香港城市大学），NUIST（南京信息工程大学）包揽了各个项目的冠军。

CUImage（商汤科技和港中文）：目标检测第一；
Trimps-Soushen（公安部三所）：目标定位第一；
CUvideo（商汤和港中文）：视频中物体检测子项目第一；
NUIST（南京信息工程大学）：视频中的物体探测两个子项目第一；
HikVision（海康威视）：场景分类第一；
SenseCUSceneParsing（商汤和港中文）：场景分析第一。

其中，Trimps-Soushen 以 2.99% 的 Top-5 分类误差率和 7.71% 的定位误差率赢得了 ImageNet 分类任务的胜利。该团队使用了分类模型的集成（即 Inception、Inception-ResNet、ResNet 和宽度残差网络模块的平均结果）和基于标注的定位模型 Faster R-CNN 来完成任务。训练数据集有 1000 个类别共计 120 万的图像数据，分割的测试集还包括训练未见过的 10 万张测试图像。

ILSVRC 2017
Momenta 提出的SENet 获得了最后一届 ImageNet 2017 竞赛 Image Classification 任务的冠军， 2.251% Top-5 错误率


### LeNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)  Yann LeCun

* LeNet  最经典的CNN网络

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/lenet-org.jpg" width="705"> </a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/lenet.png" width="405"> </a>

   [1] LeCun, Yann; Léon Bottou; Yoshua Bengio; Patrick Haffner (1998). "Gradient-based learning applied to document recognition" [pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

   tf code  https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py

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

  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/alexnet-org.jpg" width="805"></a>

  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet.png" width="505"></a>

   [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

   tensorflow 源码 https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py

   caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt


### GoogLeNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md) Christian Szegedy / Google
* GoogLeNet  采用InceptionModule和全局平均池化层，构建了一个22层的深度网络,使得很好地控制计算量和参数量的同时（ AlexNet 参数量的1/12），获得了非常好的分类性能.
它获得2014年ILSVRC挑战赛冠军，将Top5 的错误率降低到6.67%.
GoogLeNet名字将L大写，是为了向开山鼻祖的LeNet网络致敬.


  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/googlenet_th.jpeg" width="805"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/googlenet.png" width="805"></a>

   [3] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)


   tensorflow 源码 https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py

   caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt



### Inception V3  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md) Christian Szegedy / Google
* Inception V3，GoogLeNet的改进版本,采用InceptionModule和全局平均池化层，v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），ILSVRC 2012 Top-5错误率降到3.58% test error 

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/inceptionv3.png" width="805"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/v3-tf.png" width="805"></a>

   [4] Szegedy, Christian, et al. “Rethinking the inception architecture for computer vision.” arXiv preprint arXiv:1512.00567 (2015). [pdf](http://arxiv.org/abs/1512.00567)


   tensorflow 源码 https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py

   https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/inception_model.py




### VGG [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md) Karen Simonyan , Andrew Zisserman  /  [Visual Geometry Group（VGG）Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
* VGG   
VGG-Net是2014年ILSVRC classification第二名(第一名是GoogLeNet)，ILSVRC localization 第一名。VGG-Net的所有 convolutional layer 使用同样大小的 convolutional filter，大小为 3 x 3


   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/vgg.png" width="505"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/vgg.png" width="805"></a>

单独看VGG19的模型：

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/vgg19.png" width="805"></a>


   [5] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014). [pdf](https://arxiv.org/pdf/1409.1556.pdf)

   tensorflow 源码: https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py


   caffe ：

   vgg16 https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt

   vgg19 https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt




### ResNet and ResNeXt[详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md) 何凯明 [He Kaiming](http://kaiminghe.com/) 
* ResNet 
ResNet,深度残差网络，通过shortcut( skip connection )的设计，打破了深度神经网络深度的限制，使得网络深度可以多达到1001层。
它构建的152层深的神经网络，在ILSVRC2015获得在ImageNet的classification、detection、localization以及COCO的detection和segmentation上均斩获了第一名的成绩，其中classificaiton 取得3.57%的top-5错误率，
 
  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/resnet.png" width="805"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/resnet3.png" width="1005"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/resnet.png" width="805"></a>
	[6] He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015). [pdf](https://arxiv.org/pdf/1512.03385.pdf) (ResNet,Very very deep networks, CVPR best paper) 


	tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/resnet_v1.py

	https://github.com/tensorflow/models/tree/master/research/slim/nets/resnet_v2.py

	caffe https://github.com/KaimingHe/deep-residual-networks

	torch https://github.com/facebook/fb.resnet.torch

* ResNeXt 

结构采用grouped convolutions，减少了超参数的数量（子模块的拓扑结构一样），不增加参数复杂度，提高准确率。

  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/resnext.png" width="605"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/resnext.png" width="805"></a>

	[7] He, Kaiming, et al. "Aggregated Residual Transformations for Deep Neural Networks." arXiv preprint arXiv:1611.05431 . [pdf](https://arxiv.org/pdf/1611.05431.pdf) (ResNet,Very very deep networks, CVPR best paper) 


	torch https://github.com/facebookresearch/ResNeXt


### Inception-Resnet-V2[详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionResnetV2.md) Christian Szegedy / Google

Inception Resnet V2是基于Inception V3 和 ResNet结构发展而来的一个网络。在这篇paper中，还同期给出了Inception V4. 

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionResnetV2.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/inception_resnet_v2.png" width="805"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionResnetV2.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/inceptionresnet_v2_tf.png" width="805"></a>

  [8] Christian Szegedy, et al. “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning” arXiv preprint arXiv:1602.07261 (2015). [pdf](http://arxiv.org/abs/1602.07261)


github链接：
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py


### DenseNet[详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DenseNet.md) 黄高Gao Huang, 刘壮Zhuang Liu

作者发现（Deep networks with stochastic depth）通过类似Dropout的方法随机扔掉一些层，能够提高ResNet的泛化能力。于是设计了DenseNet。
DenseNet 将ResNet的residual connection 发挥到了极致，它做了两个重要的设计，一是网络的每一层都直接与其前面层相连，实现特征的重复利用，第二是网络的每一层都很窄，达到降低冗余性的目的。

DenseNet很容易训练,但是它有很多数据需要重复使用，因此显存占用很大。不过现在的更新版本，已经通过用时间换空间的方法，将DenseLayer(Contact-BN-Relu_Conv)中部分数据使用完就释放，而在需要的时候重新计算。这样增加少部分计算量，节约大量内存空间。

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DenseNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/densenet.png" width="605"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DenseNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/densenet_structure.png" width="805"></a>

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DenseNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/densenet.png" width="805"></a>
  [9] Gao Huang,Zhuang Liu, et al. DenseNet：2016，Densely Connected Convolutional Networks arXiv preprint arXiv:1608.06993 . [pdf](https://arxiv.org/pdf/1608.06993.pdf)  CVPR 2017 Best Paper
  [10]Geoff Pleiss, Danlu Chen, Gao Huang, et al.Memory-Efficient Implementation of DenseNets. [pdf](https://arxiv.org/pdf/1707.06990.pdf)

github链接：
  torch https://github.com/liuzhuang13/DenseNet

  pytorch https://github.com/gpleiss/efficient_densenet_pytorch

  caffe https://github.com/liuzhuang13/DenseNetCaffe



### DPN[详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DPN.md)  颜水成
之前我们已经了解了ResNet 和 DenseNet，ResNet使用的是相加(element-wise adding),DenseNet则使用的是拼接(concatenate)。

DPN把DenseNet和ResNet联系到了一起，该神经网络结合ResNet和DenseNet的长处，共享公共特征，并且通过双路径架构保留灵活性以探索新的特征。在设计上，采用了和ResNeXt一样的group操作。

 它在在图像分类、目标检测还是语义分割领域都有极大的优势，可以去看2017 ImageNet NUS-Qihoo_DPNs 的表现。

 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DPN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/dpn_org1.jpg" width="605"></a>


 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DPN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/dpn_struct.png" width="805"></a>

 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DPN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/dpn.png" width="805"></a>
  [11]Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng.Dual Path Networks  [pdf](https://arxiv.org/pdf/1707.01629.pdf)

github链接：

MxNet https://github.com/cypw/DPNs  (官方)

caffe:https://github.com/soeaver/caffe-model

 


### PolyNet [Xingcheng Zhang] 林达华[Dahua Lin]  / CUHK-MMLAB & 商汤科技 [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/PolyNet.md) 

这个模型在Inception_ResNet_v2 的基础上，替换了之前的Inception module，改用 PolyInception module 作为基础模块，然后通过数学多项式来组合设计每一层网络结构。因此结构非常复杂。

 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/PolyNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/polynet_th.jpg" width="605"></a>



PolyNet在ImageNet大规模图像分类测试集上获得了single-crop错误率4.25%和multi-crop错误率3.45%。在ImageNet2016的比赛中商汤科技与香港中大-商汤科技联合实验室在多项比赛中选用了这种网络结构并取得了三个单项第一的优异成绩。

提供了caffe的proto 和模型。
caffe：https://github.com/CUHK-MMLAB/polynet

模型结构图  （官方）
http://ethereon.github.io/netscope/#/gist/b22923712859813a051c796b19ce5944
https://raw.githubusercontent.com/CUHK-MMLAB/polynet/master/polynet.png

  [12] Xingcheng Zhang, Zhizhong Li, ChenChange Loy, Dahua Lin，PolyNet: A Pursuit of Structural Diversity in Very Deep Networks.2017 [pdf](https://arxiv.org/pdf/1611.05725v2.pdf)


### SENet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/SENet.md)

Momenta 提出的SENet 获得了最后一届 ImageNet 2017 竞赛 Image Classification 任务的冠军。
它在结构中增加了一个se模块，通过Squeeze 和 Excitation 的操作，学习自动获取每个特征通道的重要程度，然后依照这个重要程度去提升有用的特征并抑制对当前任务用处不大的特征。


 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/SENet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/semodule.jpg" width="605"></a>

 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/SENet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/senet.jpg" width="805"></a>


 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/SENet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/senet.png" width="805"></a>
 

  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/SENet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/senet.png" width="805"></a>
 
  [13] Squeeze-and-Excitation Networks  [pdf](https://arxiv.org/pdf/1709.01507.pdf)

  caffe:https://github.com/hujie-frank/SENet


### NASNet Google

这是谷歌用AutoML(Auto Machine Learning)在500块GPU上自行堆砌convolution cell（有两种cell
）设计的网络。性能各种战胜人类设计。


<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/NasNet_cell.jpeg">


  [14]Learning Transferable Architectures for Scalable Image Recognition[pdf](https://arxiv.org/pdf/1707.07012.pdf)


github链接：
  https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py



-----------------------------------------------------------------------------------------------------------
## 轻量级模型 & 剪枝
随着模型结构的发展，在很多机器智能领域，深度神经网络都展现出了超人的能力。但是，随着准确率的提升，这些网络也需要更多的计算资源和运行时的内存，这些需求使得高精度的大型网络无法在移动设备或者嵌入式系统上运行。

于是从应用角度发展了另外一条支线，着重在于轻量化模型的设计与发展。它的主要思想在于从卷积层的设计来构建更高效的网络计算方式，从而使网络参数减少的同时，不损失网络性能。

除了模型的设计，还有Deep Compression ，剪枝等多种方法将模型小型化。


SqueezeNet使用bottleneck方法设计一个非常小的网络，使用不到1/50的参数（125w --- 1.25million）在ImageNet上实现AlexNet级别的准确度。 MobileNetV1使用深度可分离卷积来构建轻量级深度神经网络，其中MobileNet-160（0.5x），和SqueezeNet大小差不多，但是在ImageNet上的精度提高4％。 ShuffleNet利用pointwise group卷积和channel shuffle来减少计算成本并实现比MobileNetV1更高的准确率。 MobileNetV2基于inverted residual structure with linear bottleneck，改善了移动模型在多个任务和基准测试中的最新性能。mNASNet是和NASNet一样强化学习的构造结果，准确性略优于MobileNetV2,在移动设备上具有比MobileNetV1，ShuffleNet和MobileNetV2更复杂的结构和更多的实际推理时间。(总结出自MobileFaceNets) 


“
With the same accuracy, our MnasNet model runs 1.5x faster than the hand-crafted state-of-the-art MobileNetV2, and 2.4x faster than NASNet, which also used architecture search. After applying the squeeze-and-excitation optimization, our MnasNet+SE models achieve ResNet-50 level top-1 accuracy at 76.1%, with 19x fewer parameters and 10x fewer multiply-adds operations. On COCO object detection, our model family achieve both higher accuracy and higher speed over MobileNet, and achieves comparable accuracy to the SSD300 model with 35x less computation cost.

在相同的准确率下，MnasNet 模型的速度比手工调参得到的当前最佳模型 MobileNet V2 快 1.5 倍，并且比 NASNet 快 2.4 倍，它也是使用架构搜索的算法。在应用压缩和激活（squeeze-and-excitation）优化方法后，MnasNet+SE 模型获得了 76.1% 的 ResNet 级别的 top-1 准确率，其中参数数量是 ResNet 的 1/19，且乘法-加法运算量是它的 1/10。在 COCO 目标检测任务上，我们的MnasNet模型系列获得了比 MobileNet 更快的速度和更高的准确率，并在 1/35 的计算成本下获得了和 SSD300 相当的准确率。”
https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html



<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/mallmodel.jpeg" width="700">

|网络名称|最早公开日期|发表情况|作者团队|
|:---:|:---:|:---:|:---:|
|SqueezeNet|2016.02|ICLR2017|Berkeley&Stanford|
|MobileNet|2016.04|CVPR2017|Google|
|ShuffleNet|2016.06|CVPR2017|Face++|
|Xception|2016.10|----|Google|
|MobileNetV2|2018.01|----|Google|
|ShuffleNet V2|2018.07|ECCV2018|Face++|
|MnasNet|2018.07|----|Google|

ShuffleNet 论文中引用了 SqueezeNet；Xception 论文中引用了 MobileNet

CNN网络时间消耗分析

对于一个卷积层，假设其大小为 h \times w \times c \times n （其中c为#input channel, n为#output channel），输出的feature map尺寸为 H' \times W' ，则该卷积层的

paras = n \times (h \times w \times c + 1)
FLOPS= H' \times W' \times n \times(h \times w \times c + 1)
即#FLOPS= H' \times W' \times #paras

但是虽然Conv等计算密集型操作占了其时间的绝大多数，但其它像Elemwise/Data IO等内存读写密集型操作也占了相当比例的时间





ShuffleNet_V1与MobileNet_V2上的时间消耗分析

从上图中可看出，因此像以往那样一味以FLOPs来作为指导准则来设计CNN网络是不完备的，虽然它可以反映出占大比例时间的Conv操作。
                                                                                 --------此处来自shufflenet v2



### SqueezeNet

SqueezeNet 的核心在于 Fire module，Fire module 由两层构成，分别是 squeeze 层+expand 层，squeeze 层是一个 1×1 卷积核的卷积层，对上一层 feature map 进行卷积，主要目的是减少 feature map 的维数，expand 层是 1×1 和 3×3 卷积核的卷积层，expand 层中，把 1×1 和 3×3 得到的 feature map 进行 concat。

 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/SqueezeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/squeeze.png" width="405"></a>

 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/SqueezeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/squeezenet.png" ></a>


  [1]AlexNet-level accuracy with 50x fewer parameters and <0.5MB[pdf](https://arxiv.org/pdf/1602.07360.pdf)


github链接：
 caffe: https://github.com/DeepScale/SqueezeNet


### MobileNet [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/MobileNet.md) Google

MobileNet 顾名思义，可以用在移动设备上的网络，性能和效率取得了很好平衡。它发展了两个版本，第一个版本基本结构和VGG类似，主要通过 depthwise separable convolution 来减少参数和提升计算速度。 第二代结合了ResNet的特性，提出了一种新的 Inverted Residuals and Linear Bottleneck。性能优于对应的NasNet。

![mobilenetv2_compare](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/mobilenetv2_compare.jpg)


MobileNet v1：2017，MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
  
![mobilenet_struct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/mobilenetv1.jpg)


   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/MobileNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mobilenet.png" width="805"></a>

MobileNet v2：2018，Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation

![mobilenetv2_struct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/mobilenetv2.jpg)


   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/MobileNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mobilenetv2.png" width="805"></a>



TensorFlow实现：

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py

caffe实现：https://github.com/pby5/MobileNet_Caffe


### Xception

### ShuffleNet

手机端caffe实现：https://github.com/HolmesShuan/ShuffleNet-An-Extremely-Efficient-CNN-for-Mobile-Devices-Caffe-Reimplementation 
caffe实现：https://github.com/camel007/Caffe-ShuffleNet


shufflenet v2

https://arxiv.org/abs/1807.11164

### MnasNet

  [5]MnasNet: Platform-Aware Neural Architecture Search for Mobile[pdf](https://arxiv.org/pdf/1807.11626.pdf)

### other

pytorch pretrained-model https://github.com/Cadene/pretrained-models.pytorch

