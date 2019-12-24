
# ResNet

ResNet,深度残差网络，通过shortcut( skip connection )的设计，打破了深度神经网络深度的限制，使得网络深度可以多达到1001层。
它构建的152层深的神经网络，在ILSVRC2015获得在ImageNet的classification、detection、localization以及COCO的detection和segmentation上均斩获了第一名的成绩，其中classificaiton 取得3.57%的top-5错误率，


paper：He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015). [pdf](https://arxiv.org/pdf/1512.03385.pdf) 

ResNet 原始结构如图

![resnet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnet.png)


ResNet最根本的动机就是所谓的“退化”问题，即当模型的层次加深时，错误率却提高了。 但是模型的深度加深，学习能力增强，因此更深的模型不应当产生比它更浅的模型更高的错误率。

ResNet公式：
这里的l表示层，xl表示l层的输出，Hl表示一个非线性变换。所以对于ResNet而言，l层的输出是l-1层的输出加上对l-1层输出的非线性变换。
![resnet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnet.jpg)


ResNet的基本模块如图，通过增加shortcut，增加一个identity mapping（恒等映射），将原始所需要学的函数H(x)转换成F(x)+x，而这两种表达的效果相同，但是优化的难度却并不相同，这一想法也是源于图像处理中的残差向量编码，通过一个reformulation，将一个问题分解成多个尺度直接的残差问题，能够很好的起到优化训练的效果。此外当模型的层数加深时，这个简单的结构能够很好的解决退化问题。

![resnetmodule](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnet2.png)

![resnetmoduledetail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnetmodule.png)

在不同的网络结构中，这个模块设计稍微有不同。18 和34层的网络，设计如左图，而50层以上的网络，设计如右图。

![differentresnetmodule](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnet3.jpg)

将左图可表达如下：
![resnetmodule-2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnet2.jpg)

resnet网络结构对比图如图：

![resnetstruct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnet3.png)

可以将34层和vgg的结构进行对比，如下：

![resnet34](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnet34.jpg)


用不同节点表示如图：


![resnet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/resnet.png)

<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/resnet.png)</p>


从图像尺寸，我们可以看到整个过程中有多次降采样。在conv3_1 conv4_1 conv5_1，都会有stride = 2 的降采样. 降采样前后的尺寸不同，因此不能直接相加，可以采用zero_padding:补0，或者projection：用1×1卷积变换尺寸，来处理。

源码：

tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/resnet_v1.py

https://github.com/tensorflow/models/tree/master/research/slim/nets/resnet_v2.py

caffe https://github.com/KaimingHe/deep-residual-networks

torch https://github.com/facebook/fb.resnet.torch






改进：

## Identity Mappings in Deep Residual Networks

2016发表了“Identity Mappings in Deep Residual Networks”[pdf](https://arxiv.org/pdf/1603.05027.pdf)中表明，通过使用identity mapping来更新残差模块，可以获得更高的准确性。

![resnetmodule_new](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnet_new.png)


torch :  https://github.com/KaimingHe/resnet-1k-layers

## WRN（wide residual network）：

验证了宽度给模型性能带来的提升

![wrn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/wrn.png)

github地址：https://github.com/szagoruyko/wide-residual-networks



## ResNeXt

paper：Aggregated Residual Transformations for Deep Neural Networks
Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
 [pdf](https://arxiv.org/pdf/1611.05431.pdf) 

2016的ImageNet第二
提出 ResNeXt是因为：传统方法要提高模型的准确率，都是加深或加宽网络，但是随着超参数数量的增加（比如channels数，filter size等等），网络设计的难度和计算开销也会增加。

ResNeXt 结构采用grouped  convolutions，减少了超参数的数量（子模块的拓扑结构一样），不增加参数复杂度，提高准确率。


![resnextmodule](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnextmodule.png)

ResNeXt和ResNet-50/101的区别仅仅在于其中的block，其他都不变

![resnextstruct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/resnext.png)



![resnext](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/resnext.png)

<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/resnext.png)</p>



torch: https://github.com/facebookresearch/ResNeXt



# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)
# [ResNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md)