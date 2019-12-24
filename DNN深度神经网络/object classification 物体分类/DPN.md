
# DPN(Dual Path Networks) 

之前我们已经了解了ResNet 和 DenseNet，ResNet使用的是相加(element-wise adding),DenseNet则使用的是拼接(concatenate)。

作者用High Order RNN结构（HORNN）把DenseNet和ResNet联系到了一起，证明了DenseNet能从靠前的层级中提取到新的特征，即能探索新的特征，而ResNet本质上是对之前层级中已提取特征的复用，即会重复利用特征，这两种方法都对如何学会一种优秀的特征表征方式很重要。为了获得两种路径拓扑的长处，作者提出了双路径网络（DPN），该神经网络能共享公共特征，并且通过双路径架构保留灵活性以探索新的特征。

在设计上，采用了和ResNeXt一样的group操作。由于ResNet和ResNext思想本质一致，这里要介绍的更多是DPN设计结构，而不讨论group。因此统一称为ResNet的设计

 
1、模型复杂度：The DPN-92 costs about 15% fewer parameters than ResNeXt-101 (32 4d), while the DPN-98 costs about 26% fewer parameters than ResNeXt-101 (64 4d). 
2、计算复杂度，：DPN-92 consumes about 19% less FLOPs than ResNeXt-101(32 4d), and the DPN-98 consumes about 25% less FLOPs than ResNeXt-101(64 4d).

DPN 网络如图：

![dpn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/dpn_struct.png)
其实DPN和ResNeXt（ResNet），DenseNet的结构都很相似。

然后来看 DPN是怎么结合这两种网络的

![dpn_org](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/dpn_org.jpg)


![dpn_org1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/dpn_org1.jpg)

dpn将数据分成两个部分，一部分做ResNet结构，一部分做DenseNet结构



用不同节点表示如图：


![dpnpic](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/dpn.png)



<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/dpn.png)</p>



Caffe里面实现还有一点不同。


![dpncaffepic](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/dpn_caffe.png)



<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/dpn_caffe.png)</p>



第一次看到这个网络的时候，感受就是肯定很复杂吧，但实际上 ResNet 系列发展而来的网络，都简单而易于理解，而且它在在图像分类、目标检测还是语义分割领域都有极大的优势。适合大家好好掌握


源码：

MxNet https://github.com/cypw/DPNs  (官方)

caffe:https://github.com/soeaver/caffe-model


参考：
https://zhuanlan.zhihu.com/p/32702293

# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
