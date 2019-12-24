
# DenseNet

作者发现（Deep networks with stochastic depth）通过类似Dropout的方法随机扔掉一些层，能够提高ResNet的泛化能力。于是设计了DenseNet。

DenseNet 将ResNet的residual connection 发挥到了极致，它做了两个重要的设计，一是网络的每一层都直接与其前面层相连，实现特征的重复利用，第二是网络的每一层都很窄，达到降低冗余性的目的。

DenseNet很容易训练,但是它有很多数据需要重复使用，因此显存占用很大。不过现在的更新版本，已经通过用时间换空间的方法，将DenseLayer(Contact-BN-Relu_Conv)中部分数据使用完就释放，而在需要的时候重新计算。这样增加少部分计算量，节约大量内存空间。，


paper： Gao Huang,Zhuang Liu, et al. DenseNet：2016，Densely Connected Convolutional Networks arXiv preprint arXiv:1608.06993 . [pdf](https://arxiv.org/pdf/1608.06993.pdf)  CVPR 2017 Best Paper

DenseNet 网络如图：

![densenet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/densenet.png)
上图是dense block ，下图是DenseNet结构图，其中包含了3个dense block。每个dense block内的feature map的size是一致的。
![densenet1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/densenet1.png)

DenseNet的公式。[x0,x1,…,xl-1]表示将0到l-1层的输出feature map做concatenation。Hl包括BN-ReLU-Conv。

![densenet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/densenet.jpg)




densenet网络结构如下表：

![densenetstruct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/densenet_structure.png)




用不同节点表示如图：


![densenetpic](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/densenet.png)

<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/densenet.png)</p>


很多人看了上面论文中的图，自然会觉得网络结构一定实现很复杂，再看节点表示图，会不会觉得有种画错了的感觉。 看上去怎么就像ResNet一样简单。

是的，它的代码很简单，但是每一层和之前层的连接，ResNet是sum 而 DenseNet 是contact。

好好思考一下吧。

源码：

  torch https://github.com/liuzhuang13/DenseNet

  pytorch https://github.com/gpleiss/efficient_densenet_pytorch

  caffe https://github.com/liuzhuang13/DenseNetCaffe

# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)
# [ResNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md)