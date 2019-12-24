
# Inception V3

Inception V3，GoogLeNet的改进版本,采用InceptionModule和全局平均池化层，v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，还有值得注意的地方是网络输入从224x224变为了299x299，更加精细设计了35x35/17x17/8x8的模块；ILSVRC 2012 Top-5错误率降到3.58% test error 

paper：Szegedy, Christian, et al. “Rethinking the inception architecture for computer vision.” arXiv preprint arXiv:1512.00567 (2015). [pdf](http://arxiv.org/abs/1512.00567)

Inception V3 抽象结构如图（来自[googleblog](https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html)）

![inceptionv3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/inceptionv3.png)




Factorization 如图：

![factorization](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/factor.jpg)


Inception V3 原始结构如图:

![inceptionv3 model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/inception_architecture.jpg)



用不同节点表示如图，在这里，我们是按照tensorflow中代码进行重画，由于bn的配置关系，因此没有特别将bn层进行绘制，而是着重帮助理解所有的inception module。


![inceptionv3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/v3-tf.png)


<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/v3-tf.png)</p>



源码：

tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/inception_v3.py



# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)