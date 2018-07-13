
# AlphaTree : Graphic Deep Neural Network 神经网络模型图示

在AI学习的漫漫长路上，理解不同文章中的模型与方法是每个人的必经之路，偶尔见到Fjodor van Veen所作的[A mostly complete chart of Neural Networks](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png) 和 FeiFei Li AI课程中对模型的[画法](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/feifei.png)，大为触动。决定将深度神经网络中的一些模型 进行统一的图示，便于大家对模型的理解。

从AI研究的角度来说，AI的学习和跟进是有偏向性的，更多的精英是擅长相关的一到两个领域，在这个领域做到更好。而从AI应用工程师的角度来说，每一个工程都可能涉及很多个AI的方向，而他们需要了解掌握不同的方向才能更好的开发和设计。但是AI中每一个领域都在日新月异的成长。而每一位研究人员写paper的风格都不一样，相似的模型，为了突出不同的改进点，他们对模型的描述和图示都可能大不相同。为了帮助更多的人在不同领域能够快速跟进前沿技术，我们构建了“AlphaTree计划”，每一篇文章都会对应文章，代码，然后进行图示输出。

进度：

2018/05/30 目前object classification 主干部分基本完成 包括 LeNet, AlexNet, GoogLeNet, Inception V3,Inception-Resnet-V2, VGG, ResNet ,ResNext, DenseNet ,DPN等。

2018/06/15 完成 MobileNet 与 MobileNet V2.

2018/07/04 完成FaceNet系列

其他：
Face ： mtcnn 

OCR ： CRNN CTPN Textboxes Textboxes++

Object Detection：ssd




在了解这个计划之前，我们先了解一下各个方向的发展情况，用地铁图的形式绘制了第一个版本。第一个版本主要包括了图像方向的十二个子方向，里面列的模型，除了商业化的部分外，其他则选择了一些应用方向使用较多的开源模型。也许还有遗漏与不完善的地方，大家一起来更新吧。



<a href="https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/map/mapclean_1.0.png"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/mapclean_1.0.png" width="1200"> </a>


[大图链接](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/map/mapclean_1.0.png)


现在主要有如下方向

![方向](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/directions.png)


每个方向上都记录了一些比较重要的模型结构。为了方便理解模型的变化，我们对模型进行统一的图示

一个常用的神经网络结构可以进行如下的等价变换。每个圆圈上的数字代表这一层有多少个神经元。


<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/simplelize.png" width="600">

图像的深度卷积网络 每一层的卷积核大小也是大家感兴趣的，因此会在简化结构中，加上卷积核大小。这个可以写在圆圈中，也可以写在圆圈下方。

如 [LeNet网络](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md) 

![Lenet_short](https://github.com/weslynn/graphic-deep-neural-network/raw/master/modelpic/lenet-short.png)



模型中用到的图标规定如下：
![图标](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/cellsreadme.png)

当大家描述网络结构时，常常会将卷积层和maxpooling层画在一起，我们也提供了这样的简化方法

  <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/equal.png" width="205">

一个具体的问题是否能用人工智能，或者更进一步说用深度学习某个算法解决，首先需要人对问题进行分解，提炼成可以用机器解决的问题，譬如说分类问题，回归问题，聚类问题等。

PS： caffe 模型可视化网址 http://ethereon.github.io/netscope/#/editor




版权申明：CC-BY-NC-SA 知识共享-署名-非商业性-相同方式共享
---------------------------------------------------------------------------------------------------
## Object Classification 物体分类

深度学习在解决分类问题上非常厉害。让它声名大噪的也是对于图像分类问题的解决。也产生了很多很经典的模型。其他方向的模型发展很多都是源于这各部分，它是很多模型的基础工作。因此我们首先了解一下它们。


![object classification](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/ObjectClassification.png)


从模型的发展过程中，随着准确率的提高，网络结构也在不断的进行改进，现在主要是两个方向，一是深度，二是复杂度。此外还有卷积核的变换等等。

深度神经网络的发展要从经典的LeNet模型说起，那是1998年提出的一个模型，在手写数字识别上达到商用标准。之后神经网络的发展就由于硬件和数据的限制，调参的难度等各种因素进入沉寂期。

到了2012年，Alex Krizhevsky 设计了一个使用ReLu做激活函数的AlexNet 在当年的ImageNet图像分类竞赛中(ILSVRC 2012)，以top-5错误率15.3%拿下第一。 他的top-5错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。而且网络针对多GPU训练进行了优化设计。从此开始了深度学习的黄金时代。

大家发表的paper一般可以分为两大类，一类是网络结构的改进，一类是训练过程的改进，如droppath，loss改进等。

之后网络结构设计发展主要有两条主线，一条是Inception系列（即上面说的复杂度），从GoogLeNet 到Inception V2 V3 V4，Inception ResNet。 Inception module模块在不断变化，一条是VGG系列（即深度），用简单的结构，尽可能的使得网络变得更深。从VGG 发展到ResNet ，再到DenseNet ，DPN等。 

最终Google Brain用500块GPU训练出了比人类设计的网络结构更优的网络NASNet。

此外，应用方面更注重的是，如何将模型设计得更小，这中间就涉及到很多卷积核的变换。这条路线则包括 SqueezeNet，MobileNet V1 V2 Xception shuffleNet等。ResNet的变种ResNeXt 和SENet 都是从小模型的设计思路发展而来。


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


### LeNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)  Yann LeCun

* LeNet  最经典的CNN网络

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



  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet.png" width="505"></a>

   [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

   tensorflow 源码 https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py

   caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt


### GoogLeNet  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md) Christian Szegedy / Google
* GoogLeNet  采用InceptionModule和全局平均池化层，构建了一个22层的深度网络,使得很好地控制计算量和参数量的同时（ AlexNet 参数量的1/12），获得了非常好的分类性能.
它获得2014年ILSVRC挑战赛冠军，将Top5 的错误率降低到6.67%.
GoogLeNet名字将L大写，是为了向开山鼻祖的LeNet网络致敬.

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/googlenet.png" width="805"></a>

   [3] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)


   tensorflow 源码 https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py

   caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt



### Inception V3  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md) Christian Szegedy / Google
* Inception V3，GoogLeNet的改进版本,采用InceptionModule和全局平均池化层，v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），ILSVRC 2012 Top-5错误率降到3.58% test error 

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/v3-tf.png" width="805"></a>

   [4] Szegedy, Christian, et al. “Rethinking the inception architecture for computer vision.” arXiv preprint arXiv:1512.00567 (2015). [pdf](http://arxiv.org/abs/1512.00567)


   tensorflow 源码 https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py

   https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/inception_model.py




### VGG [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md) Karen Simonyan , Andrew Zisserman  /  [Visual Geometry Group（VGG）Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
* VGG   
VGG-Net是2014年ILSVRC classification第二名(第一名是GoogLeNet)，ILSVRC localization 第一名。VGG-Net的所有 convolutional layer 使用同样大小的 convolutional filter，大小为 3 x 3

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


   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/resnet.png" width="805"></a>
	[6] He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015). [pdf](https://arxiv.org/pdf/1512.03385.pdf) (ResNet,Very very deep networks, CVPR best paper) 


	tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/resnet_v1.py

	https://github.com/tensorflow/models/tree/master/research/slim/nets/resnet_v2.py

	caffe https://github.com/KaimingHe/deep-residual-networks

	torch https://github.com/facebook/fb.resnet.torch

* ResNeXt 

结构采用grouped convolutions，减少了超参数的数量（子模块的拓扑结构一样），不增加参数复杂度，提高准确率。
   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/resnext.png" width="805"></a>
	[7] He, Kaiming, et al. "Aggregated Residual Transformations for Deep Neural Networks." arXiv preprint arXiv:1611.05431 . [pdf](https://arxiv.org/pdf/1611.05431.pdf) (ResNet,Very very deep networks, CVPR best paper) 


	torch https://github.com/facebookresearch/ResNeXt


### Inception-Resnet-V2[详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionResnetV2.md) Christian Szegedy / Google

Inception Resnet V2是基于Inception V3 和 ResNet结构发展而来的一个网络。在这篇paper中，还同期给出了Inception V4. 


   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionResnetV2.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/inceptionresnet_v2_tf.png" width="805"></a>

  [8] Christian Szegedy, et al. “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning” arXiv preprint arXiv:1602.07261 (2015). [pdf](http://arxiv.org/abs/1602.07261)


github链接：
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py


### DenseNet[详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DenseNet.md) 黄高Gao Huang, 刘壮Zhuang Liu

作者发现（Deep networks with stochastic depth）通过类似Dropout的方法随机扔掉一些层，能够提高ResNet的泛化能力。于是设计了DenseNet。
DenseNet 将ResNet的residual connection 发挥到了极致，它做了两个重要的设计，一是网络的每一层都直接与其前面层相连，实现特征的重复利用，第二是网络的每一层都很窄，达到降低冗余性的目的。

DenseNet很容易训练,但是它有很多数据需要重复使用，因此显存占用很大。不过现在的更新版本，已经通过用时间换空间的方法，将DenseLayer(Contact-BN-Relu_Conv)中部分数据使用完就释放，而在需要的时候重新计算。这样增加少部分计算量，节约大量内存空间。

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

 <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DPN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/dpn.png" width="805"></a>
  [11]Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng.Dual Path Networks  [pdf](https://arxiv.org/pdf/1707.01629.pdf)

github链接：

MxNet https://github.com/cypw/DPNs  (官方)

caffe:https://github.com/soeaver/caffe-model

 


### PolyNet [Xingcheng Zhang] 林达华[Dahua Lin]  / CUHK-MMLAB & 商汤科技

这个模型在Inception_ResNet_v2 的基础上，替换了之前的Inception module，改用 PolyInception module 作为基础模块，然后通过数学多项式来组合设计每一层网络结构。因此结构非常复杂。


PolyNet在ImageNet大规模图像分类测试集上获得了single-crop错误率4.25%和multi-crop错误率3.45%。在ImageNet2016的比赛中商汤科技与香港中大-商汤科技联合实验室在多项比赛中选用了这种网络结构并取得了三个单项第一的优异成绩。

提供了caffe的proto 和模型。
caffe：https://github.com/CUHK-MMLAB/polynet

模型结构图  （官方）
http://ethereon.github.io/netscope/#/gist/b22923712859813a051c796b19ce5944
https://raw.githubusercontent.com/CUHK-MMLAB/polynet/master/polynet.png

  [12] Xingcheng Zhang, Zhizhong Li, ChenChange Loy, Dahua Lin，PolyNet: A Pursuit of Structural Diversity in Very Deep Networks.2017 [pdf](https://arxiv.org/pdf/1611.05725v2.pdf)

### NASNet Google

这是谷歌用AutoML(Auto Machine Learning)在500块GPU上自行堆砌convolution cell（有两种cell
）设计的网络。性能各种战胜人类设计。


<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/NasNet_cell.jpeg">


  [13]Learning Transferable Architectures for Scalable Image Recognition[pdf](https://arxiv.org/pdf/1707.07012.pdf)


github链接：
  https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py



-----------------------------------------------------------------------------------------------------------
## 轻量级模型 & 剪枝
随着模型结构的发展，在很多机器智能领域，深度神经网络都展现出了超人的能力。但是，随着准确率的提升，这些网络也需要更多的计算资源和运行时的内存，这些需求使得高精度的大型网络无法在移动设备或者嵌入式系统上运行。

于是从应用角度发展了另外一条支线，着重在于轻量化模型的设计与发展。它的主要思想在于从卷积层的设计来构建更高效的网络计算方式，从而使网络参数减少的同时，不损失网络性能。

除了模型的设计，还有Deep Compression ，剪枝等多种方法将模型小型化。

|网络名称|最早公开日期|发表情况|作者团队|
|:---:|:---:|:---:|:---:|
|SqueezeNet|2016.02|ICLR2017|Berkeley&Stanford|
|MobileNet|2016.04|CVPR2017|Google|
|ShuffleNet|2016.06|CVPR2017|Face++|
|Xception|2016.10|----|Google|
|MobileNetV2|2018.01|----|Google|


### SqueezeNet
SqueezeNet：AlexNet-level accuracy with 50x fewer parameters and <0.5MB

SqueezeNet 的核心在于 Fire module，Fire module 由两层构成，分别是 squeeze 层+expand 层，squeeze 层是一个 1×1 卷积核的卷积层，expand 层是 1×1 和 3×3 卷积核的卷积层，expand 层中，把 1×1 和 3×3 得到的 feature map 进行 concat。


### MobileNet [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/MobileNet.md) Google

MobileNet 顾名思义，可以用在移动设备上的网络，性能和效率取得了很好平衡。它发展了两个版本，第一个版本基本结构和VGG类似，主要通过 depthwise separable convolution 来减少参数和提升计算速度。 第二代结合了ResNet的特性，提出了一种新的 Inverted Residuals and Linear Bottleneck。性能优于对应的NasNet。

MobileNet v1：2017，MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
  
   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/MobileNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mobilenet.png" width="805"></a>

MobileNet v2：2018，Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/MobileNet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mobilenetv2.png" width="805"></a>



TensorFlow实现：

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py

caffe实现：https://github.com/pby5/MobileNet_Caffe


### Xception

### ShuffleNet

手机端caffe实现：https://github.com/HolmesShuan/ShuffleNet-An-Extremely-Efficient-CNN-for-Mobile-Devices-Caffe-Reimplementation 
caffe实现：https://github.com/camel007/Caffe-ShuffleNet





### other

pytorch pretrained-model https://github.com/Cadene/pretrained-models.pytorch



![FaceDetection](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/FaceDetection.png)

## Face Detection and Face Alignment 人脸检测与矫正
人脸检测与识别是一个研究很久的课题。传统方法之前也有了很多稳定可行的方法。而深度学习的出现，无论对检测还是识别又有了很大的提升。随着算法和代码的开源，现在很多公司都可以自己搭建一套自己的人脸检测识别系统。那么下面几篇经典论文，都是会需要接触到的。

### MTCNN [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/MTCNN.md) [zhang kaipeng](https://kpzhang93.github.io/) 乔宇 [Qiao Yu](http://mmlab.siat.ac.cn/yuqiao/) / CUHK-MMLAB & SIAT

* MTCNN 
MTCNN 将人脸检测与关键点检测放到了一起来完成。整个任务分解后让三个子网络来完成。每个网络都很浅，使用多个小网络级联，较好的完成任务。

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/MTCNN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/face/mtcnn.png" width="400"></a>


   [1] [ECCV2016] Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks [pdf](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)


   Caffe 源码：https://github.com/kpzhang93/MTCNN_face_detection_alignment 官方

   tensorflow 源码 : https://github.com/davidsandberg/facenet/tree/master/src/align 




### FaceBoxes


文章链接： 《FaceBoxes: A CPU Real-time Face Detector with High Accuracy》

非官方实现 https://github.com/zeusees/FaceBoxes


### PCN 


https://arxiv.org/pdf/1804.06039.pdf.



https://github.com/Jack-CV/PCN


此外我们可以了解一下商用的一些算法：



### DenseBox Baidu

DenseBox: Unifying Landmark Localization with End to End Object Detection

arxiv: http://arxiv.org/abs/1509.04874
demo: http://pan.baidu.com/s/1mgoWWsS
KITTI result: http://www.cvlibs.net/datasets/kitti/eval_object.php

-------------------------------------------------------------
### Landmark Localization 68 points
-------------------------------------------------------------

### dlib and CLM-framework
dlib：【ERT算法】One Millisecond Face Alignment with an Ensemble of Regression Trees.

## face++  DCNN

## TCDCN  VanillaCNN TCNN（Tweaked Convolutional Neural Networks） 
TCDCN Facial Landmark Detection by Deep Multi-task Learning
http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
VanillaCNN


## DAN Deep-Alignment-Network

https://github.com/MarekKowalski/DeepAlignmentNetwork

![FaceRecognition](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/FaceRecognition.png)




## face Recognition
DeepFace是FaceBook提出来的，后续有DeepID和FaceNet出现。DeepFace是第一个真正将大数据和深度神经网络应用于人脸识别和验证的方法，人脸识别精度接近人类水平，可以谓之CNN在人脸识别的奠基之作



### Deep Face

DeepFace: Closing the Gap to Human-Level Performance in Face Verification 



### Facenet [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/Facenet.md) 

和物体分类这种分类问题不同，Facenet是构建了一种框架，通过已有的深度模型，结合不同loss，训练一个很棒的人脸特征。它直接使用端对端的方法去学习一个人脸图像到欧式空间的编码，这样构建的映射空间里的距离就代表了人脸图像的相似性。然后基于这个映射空间，就可以轻松完成人脸识别，人脸验证和人脸聚类。

[CVPR2015] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[J]. arXiv preprint arXiv:1503.03832, 2015.[pdf](https://arxiv.org/pdf/1503.03832.pdf) 

Model name          LFW accuracy  Training dataset  Architecture

[20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE)        0.987      CASIA-WebFace    Inception ResNet v1

[20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk)        0.992       MS-Celeb-1M     Inception ResNet v1

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/Facenet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/facepic/facenet_struct.png"></a>

它使用现有的模型结构，然后将卷积神经网络去掉sofmax后，经过L2的归一化，然后得到特征表示，之后基于这个特征表示计算Loss。文章中使用的结构是[ZFNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)，[GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)，tf代码是改用了Inception_resnet_v1。

Loss的发展：
文中使用的Loss 是 triplet loss。后来相应的改进有ECCV2016的 center loss，SphereFace，2018年的AMSoftmax和ArchFace（InsightFace），现在效果最好的是ArchFace（InsightFace）。


https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py


 tensorflow 源码 :https://github.com/davidsandberg/facenet

 caffe center loss:https://github.com/kpzhang93/caffe-face
 mxnet center loss :https://github.com/pangyupo/mxnet_center_loss
 
 caffe sphereface:  https://github.com/wy1iu/sphereface

 deepinsight： https://github.com/deepinsight/insightface
 AMSoftmax ：https://github.com/happynear/AMSoftmax


### SeetaFace 

Detection: Funnel-Structured Cascade for Multi-View Face Detection with Alignment-Awareness

2016 


### OpenFace

B. Amos, B. Ludwiczuk, M. Satyanarayanan,
"Openface: A general-purpose face recognition library with mobile applications,"
CMU-CS-16-118, CMU School of Computer Science, Tech. Rep., 2016.

https://github.com/cmusatyalab/openface/





## 3d face

Face Alignment Across Large Poses: A 3D Solution
http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm


HPEN High-Fidelity Pose and Expression Normalization for Face Recognition in the Wild

dense face alignment /Pose-Invariant Face Alignment (PIFA)
http://cvlab.cse.msu.edu/project-pifa.html

face-alignment
2D-and-3D-face-alignment
两个github项目，在做同一件事，2d和3d的人脸对齐问题，区别在于前者是Pytorch 的代码，后者是Torch7的。 
论文有个很霸道的名字：《How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks) 》ICCV2017
github：https://github.com/1adrianb/face-alignment 
github: https://github.com/1adrianb/2D-and-3D-face-alignment




2D-FAN：https://www.adrianbulat.com/downloads/FaceAlignment/2D-FAN-300W.t7

3D-FAN：https://www.adrianbulat.com/downloads/FaceAlignment/3D-FAN.t7

2D-to-3D FAN：https://www.adrianbulat.com/downloads/FaceAlignment/2D-to-3D-FAN.tar.gz

3D-FAN-depth：https://www.adrianbulat.com/downloads/FaceAlignment/3D-FAN-depth

Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network
https://github.com/anhttran/3dmm_cnn

vrn
用CNN Regression的方法解决大姿态下的三维人脸重建问题。 
ICCV论文：《Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression》

github：https://github.com/AaronJackson/vrn






Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network

https://github.com/anhttran/3dmm_cnn

Expression-Net
https://github.com/fengju514/Expression-Net
Estimate 3D face pose by a Convolutional Neural Network
FACE POSE NET
https://github.com/fengju514/Face-Pose-Net
-------------------------------------------------------------------------------
![OCR](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/OCR.png)

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


### CTPN (Connectionist Text Proposal Network)  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CTPN.md)  [Zhi Tian](http://www.ece.mtu.edu/faculty/ztian/),  乔宇 [Qiao Yu](http://mmlab.siat.ac.cn/yuqiao/) / CUHK-MMLAB & SIAT


* CTPN 使用CNN + RNN 进行文本检测与定位。

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CTPN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/ctpn.png" width="905"> </a>


作者caffe中模型结构如图：

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/CTPN.md"><img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/ctpn_caffe.png" width="905"></a>


   [1] [ECCV2016] Detecting Text in Natural Image with Connectionist Text Proposal Network [pdf](https://arxiv.org/pdf/1609.03605.pdf) 


   Caffe 源码：https://github.com/tianzhi0549/CTPN 官方


### TextBoxes  [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes.md) 白翔 Xiang Bai/Media and Communication Lab, HUST


* TextBoxes，一个端到端的场景文本检测模型。这个算法是基于SSD来实现的,解决水平文字检测问题，将原来3×3的kernel改成了更适应文字的long conv kernels 3×3 -> 1×5。default boxes 也做了修改。
   
   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/TextBoxes.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/textboxes.png" width="905"> </a>


作者caffe中模型结构如图：

  <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/textboxes_caffe.png" width="905"> </a>


   [2]  M. Liao et al. TextBoxes: A Fast Text Detector with a Single Deep Neural Network. AAAI, 2017. [pdf](https://arxiv.org/pdf/1611.06779.pdf) 


   Caffe 源码：https://github.com/MhLiao/TextBoxes 官方




### TextBoxes++ [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes++.md) 白翔 Xiang Bai/Media and Communication Lab, HUST


* TextBoxes++ 这个算法也是基于SSD来实现的，实现了对多方向文字的检测。boundingbox的输出从4维的水平的boundingbox扩展到4+8=12维的输出。long conv kernels 从 1×5 改成了 3×5。default boxes 也做了修改。
   
   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes++.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/textboxes++.png" width="905"> </a>



   [3]  M. Liao et al. TextBoxes++: Multi-oriented text detection [pdf](https://arxiv.org/pdf/1801.02765.pdf)



   Caffe 源码：https://github.com/MhLiao/TextBoxes_plusplus 官方



### East: An Efficient and Accurate Scene Text Detector

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
   <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/crnn.png" width="805"></a>

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
![ObjectDetection&Seg](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/ObjectDetection&Seg.png)

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

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20detection%20%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B/SSD.md"><img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/objdetection/ssd.png" width="805"></a>

   [7] Liu, Wei, et al. "SSD: Single Shot MultiBox Detector." arXiv preprint arXiv:1512.02325 (2015). [pdf](https://arxiv.org/pdf/1512.02325.pdf)  


   tensorflow 源码 https://github.com/balancap/SSD-Tensorflow/blob/master/nets/ssd_vgg_300.py

   caffe ：https://github.com/weiliu89/caffe/tree/ssd


ParseNet

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



### DeepMask

https://github.com/facebookresearch/deepmask



![Art&Gan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/Art&Ganpic.png)

## Art

### Interactive Deep Colorization
https://github.com/junyanz/interactive-deep-colorization


### pix2pix
Edges2cats：

http://affinelayer.com/pixsrv/index.html

Github：

https://github.com/phillipi/pix2pix


http://paintschainer.preferred.tech/index_zh.html


### Neural Doodle

使用深度神经网络把你的二流涂鸦变成艺术品。

Champandard（2016） “Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks”

基于 Chuan Li 和 Michael Wand（2016）在论文“Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis”中提出的 Neural Patches 算法。这篇文章中深入解释了这个项目的动机和灵感来源：https://nucl.ai/blog/neural-doodles/

doodle.py 脚本通过使用1个，2个，3个或4个图像作为输入来生成新的图像，输入的图像数量取决于你希望生成怎样的图像：原始风格及它的注释（annotation），以及带有注释（即你的涂鸦）的目标内容图像（可选）。该算法从带风格图像中提取 annotated patches，然后根据它们匹配的紧密程度用这些 annotated patches 渐进地改变目标图像的风格。

Github 地址：https://github.com/alexjc/neural-doodle
### Deep Painterly Harmonization
https://github.com/luanfujun/deep-painterly-harmonization


### 

### Colornet



Github 地址：https://github.com/pavelgonchar/colornet
## 强化学习




SAC-X


## GAN


https://github.com/wiseodd/generative-models

### info gan
https://github.com/openai/InfoGAN

GAN - Ian Goodfellow, arXiv:1406.2661v1

DCGAN - Alec Radford & Luke Metz, arxiv:1511.06434

CGAN - Mehdi Mirza, arXiv:1411.1784v1

LAPGAN - Emily Denton & Soumith Chintala, arxiv: 1506.05751

InfoGAN - Xi Chen, arxiv: 1606.03657

PPGAN - Anh Nguyen, arXiv:1612.00005v1

WGAN - Martin Arjovsky, arXiv:1701.07875v1

LS-GAN - Guo-Jun Qi, arxiv: 1701.06264

SeqGAN - Lantao Yu, arxiv: 1609.05473

EBGAN - Junbo Zhao, arXiv:1609.03126v2

VAEGAN - Anders Boesen Lindbo Larsen, arxiv: 1512.09300

......

特定任务中提出来的模型，如GAN-CLS、GAN-INT、SRGAN、iGAN、IAN 等
LS-GAN

Torch 版本：https://github.com/guojunq/lsgan

SRGAN

TensorFlow 版本：https://github.com/buriburisuri/SRGAN

Torch 版本：https://github.com/leehomyc/Photo-Realistic-Super-Resoluton

Keras 版本：https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks

iGAN

Theano 版本：https://github.com/junyanz/iGAN

IAN

Theano 版本：https://github.com/ajbrock/Neural-Photo-Editor

Pix2pix

Torch 版本：https://github.com/phillipi/pix2pix

TensorFlow 版本：https://github.com/yenchenlin/pix2pix-tensorflow

GAN for Neural dialogue generation

Torch 版本：https://github.com/jiweil/Neural-Dialogue-Generation

Text2image

Torch 版本：https://github.com/reedscot/icml2016

TensorFlow+Theano 版本：https://github.com/paarthneekhara/text-to-image

GAN for Imitation Learning

Theano 版本：https://github.com/openai/imitation

SeqGAN

TensorFlow 版本：https://github.com/LantaoYu/SeqGAN


Qi G J. Loss-Sensitive Generative Adversarial Networks onLipschitz Densities[J]. arXiv preprint arXiv:1701.06264, 2017.

Li J, Monroe W, Shi T, et al. Adversarial Learning for NeuralDialogue Generation[J]. arXiv preprint arXiv:1701.06547, 2017.

Sønderby C K, Caballero J, Theis L, et al. Amortised MAPInference for Image Super-resolution[J]. arXiv preprint arXiv:1610.04490, 2016.

Ravanbakhsh S, Lanusse F, Mandelbaum R, et al. Enabling DarkEnergy Science with Deep Generative Models of Galaxy Images[J]. arXiv preprintarXiv:1609.05796, 2016.

Ho J, Ermon S. Generative adversarial imitationlearning[C]//Advances in Neural Information Processing Systems. 2016:4565-4573.

Zhu J Y, Krähenbühl P, Shechtman E, et al. Generative visualmanipulation on the natural image manifold[C]//European Conference on ComputerVision. Springer International Publishing, 2016: 597-613.

Isola P, Zhu J Y, Zhou T, et al. Image-to-image translationwith conditional adversarial networks[J]. arXiv preprint arXiv:1611.07004,2016.

Shrivastava A, Pfister T, Tuzel O, et al. Learning fromSimulated and Unsupervised Images through Adversarial Training[J]. arXivpreprint arXiv:1612.07828, 2016.

Ledig C, Theis L, Huszár F, et al. Photo-realistic singleimage super-resolution using a generative adversarial network[J]. arXivpreprint arXiv:1609.04802, 2016.

Nguyen A, Yosinski J, Bengio Y, et al. Plug & playgenerative networks: Conditional iterative generation of images in latentspace[J]. arXiv preprint arXiv:1612.00005, 2016.

Yu L, Zhang W, Wang J, et al. Seqgan: sequence generativeadversarial nets with policy gradient[J]. arXiv preprint arXiv:1609.05473,2016.

Lotter W, Kreiman G, Cox D. Unsupervised learning of visualstructure using predictive generative networks[J]. arXiv preprintarXiv:1511.06380, 2015.

Reed S, Akata Z, Yan X, et al. Generative adversarial textto image synthesis[C]//Proceedings of The 33rd International Conference onMachine Learning. 2016, 3.

Brock A, Lim T, Ritchie J M, et al. Neural photo editingwith introspective adversarial networks[J]. arXiv preprint arXiv:1609.07093,2016.

Pfau D, Vinyals O. Connecting generative adversarialnetworks and actor-critic methods[J]. arXiv preprint arXiv:1610.01945, 2016.



### iGan
https://github.com/junyanz/iGAN#igan-interactive-image-generation-via-generative-adversarial-networks










Neural Dialogue Generation
https://github.com/jiweil/Neural-Dialogue-Generation










----------------------------------------------------------------------------------

## Datasets 数据库


### 人脸数据库 [详解 detail](https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/blob/master/database/face.md)
目前人脸识别领域常用的人脸数据库主要有:

* WebFaces,Caltech， 10k+人，约500K张图片，非限制场景， http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/#Download

* CelebA，Multimedia Laboratory The Chinese University of Hong Kong 汤晓鸥，10K 名人，202K 脸部图像，每个图像40余标注属性 http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
该数据集为香港中文大学汤晓鸥老师组开源的数据集，主要包含了5个关键点，40个属性值等，包含了202599张图片，图片都是高清的名人图片，可以用于人脸检测，5点训练，人脸头部姿势的训练等。


* MSRA-CFW,MSRA,202792 张, 1583人 Data Set of Celebrity Faces on the Web http://research.microsoft.com/en-us/projects/msra-cfw/CASIA 

* CASIA-WebFace,李子青 Center for Biometrics and Security Research， 500k图片，10k个人 http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
该数据集为中科院自动化所，李子青老师组开源的数据集，包含了10575个人，一共494414张图片，其中有3个人和lfw中的一样。该数据集主要用于人脸识别。图像都是著名电影中crop而出的，每个图片的大小都是250×250，每个类下面都有3张以上的图片，非常适合做人脸识别的训练。需要邮箱申请 


* MegaFace，华盛顿大学百万人脸MegaFace数据集 
MegaFace资料集包含一百万张图片，代表690000个独特的人。所有数据都是华盛顿大学从Flickr（雅虎旗下图片分享网站）组织收集的。这是第一个在一百万规模级别的面部识别算法测试基准。 现有脸部识别系统仍难以准确识别超过百万的数据量。为了比较现有公开脸部识别算法的准确度，华盛顿大学在去年年底开展了一个名为“MegaFace Challenge”的公开竞赛。这个项目旨在研究当数据库规模提升数个量级时，现有的脸部识别系统能否维持可靠的准确率。需要邮箱申请才可以下载，下载链接为http://megaface.cs.washington.edu/dataset/download.html

* IMDB-WIKI 20k+个名人的460k+张图片 和维基百科62k+张图片, 总共： 523k+张图片，名人年龄、性别 https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

* CACD2000 2k名人160k张人脸图片 http://bcsiriuschen.github.io/CARC/

* AFLW（Annotated Facial Landmarks in the Wild）

  AFLW人脸数据库是一个包括多姿态、多视角的大规模人脸数据库，而且每个人脸都被标注了21个特征点。此数据库信息量非常大，包括了各种姿态、表情、光照、种族等因素影响的图片。AFLW人脸数据库大约包括25000万已手工标注的人脸图片，其中59%为女性，41%为男性，大部分的图片都是彩色，只有少部分是灰色图片。该数据库非常适合用于人脸识别、人脸检测、人脸对齐等方面的研究，具有很高的研究价值。图像如下图所示，需要申请帐号才可以下载，下载链接为http://lrs.icg.tugraz.at/research/aflw/


* WIDER FACE
   WIDER FACE是香港中文大学的一个提供更广泛人脸数据的人脸检测基准数据集，由YangShuo， Luo Ping ，Loy ，Chen Change ，Tang Xiaoou收集。它包含32203个图像和393703个人脸图像，在尺度，姿势，闭塞，表达，装扮，关照等方面表现出了大的变化。WIDER FACE是基于61个事件类别组织的，对于每一个事件类别，选取其中的40%作为训练集，10%用于交叉验证（cross validation），50%作为测试集。和PASCAL VOC数据集一样，该数据集也采用相同的指标。和MALF和Caltech数据集一样，对于测试图像并没有提供相应的背景边界框。图像如下图所示，下载链接为http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

* 300W

  300W数据集是由AFLW，AFW，Helen，IBUG，LFPW，LFW等数据集组成的数据库。需要邮箱申请才可以下载，下载链接为http://ibug.doc.ic.ac.uk/resources/300-W/

* VGG Face dataset
该数据集包含了2622个不同的人，每个人包含1000张图片，是一个训练人脸识别的大的数据集，官网提供了每个图片的URL，需要自己解析下载，当然有些链接是需要翻墙的。下载链接：http://www.robots.ox.ac.uk/~vgg/data/vgg_face/



# 贡献力量

如果想做出贡献的话，你可以：

帮忙对没有收录的paper进行模型绘制

帮忙进行模型校对等

提出修改建议



一树一获者，谷也；一树十获者，木也；一树百获者；人也。 希望我们每一个人的努力，能够建立一片森林，为后人投下一片树荫。

每一位加入的作者，都可以选取植物的名称来表示自己，然后logo和名字将会作为自己的署名。

我希望，这终将成为一片森林。



此外，关于深度学习系统中模型结构要怎样设计，特定的任务要不要加入特定的结构和方法，Yann LeCun 和 Christopher Manning 有一个讨论 ，大家可以看一看 https://youtu.be/fKk9KhGRBdI 
雷锋网有介绍 https://www.leiphone.com/news/201804/6l2mAsZQCQG2qYbi.html