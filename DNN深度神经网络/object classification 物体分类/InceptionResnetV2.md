
# Inception Resnet V2

Inception Resnet V2是基于Inception V3 和 ResNet结构发展而来的一个网络。在这篇paper中，还同期给出了Inception V4.

准确率和模型对比如下图：
![inceptionresult](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/inceptionresnet_result.png)

paper：“Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning” arXiv preprint arXiv:1602.07261 (2015). [pdf](http://arxiv.org/abs/1602.07261)

Inception ResNet V2 抽象结构如图（来自googleblog）

![inceptionresnetv2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/inception_resnet_v2.png)


Inception ResNet V2 原始结构如图 [大图来源](http://yeephycho.github.io/2016/08/31/A-reminder-of-algorithms-in-Convolutional-Neural-Networks-and-their-influences-III/):

![Inception ResNet V2 model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/Inception_ResNet_v2_raw.jpg)

对比Inception V4 ，V4结构如下：

![InceptionV4 model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/Inception_v4.jpg)


用不同节点表示如图(按照tensorflow代码重画)

![inceptionresnetv2tf](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/inceptionresnet_v2_tf.png)


<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/inceptionresnet_v2_tf.png)</p>



源码：

tensorflow 源码 https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py


缩放Residuals
当卷积核个数超过1000时，训练将会变得不稳定，这时候缩小Residuals有助于稳定训练，缩小因子介于0.1到0.3。

作者提出了“two phase”训练。首先“warm up”，使用较小的学习率。接着再使用较大的学习率。

训练方法
使用Momentum + SGM，momentum=0.9。使用RMSProp，decay为0.9，ϵ=1.0。

学习率为0.045，每2个epoch缩小为原理的0.94。





# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
