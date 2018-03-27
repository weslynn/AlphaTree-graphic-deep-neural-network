# AlexNet
2012年，Alex Krizhevsky用AlexNet 在当年的ImageNet图像分类竞赛中(ILSVRC 2012)，以top-5错误率15.3%拿下第一。 他的top-5错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。它使用ReLU代替了传统的激活函数，而且网络针对多GPU训练进行了优化设计，用于提升速度。不过随着硬件发展，现在我们训练AlexNet都可以直接用简化后的代码来实现了。


paper ：Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

slides: http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf

AlexNet原始结构如图

![alexnet org model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/alexnet-org.jpg)

可以将模型结构表示如图：

![alexnet other model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/alexnet2.png)

模型结构如下

一层卷积层： 11×11的卷积核，96个，步长位4 （stride = 4）

一层maxpooling

一层卷积层：5×5的卷积核，256个，pad=2

一层maxpooling

一层卷积层：3×3的卷积核，384个，pad=1

一层卷积层：3×3的卷积核，384个，pad=1

一层卷积层：3×3的卷积核，256个

一层maxpooling

一层全连接层：4096个隐含节点，激活函数为ReLU

一层全连接层：4096个隐含节点，激活函数为ReLU

最后通过softmax分类输出1000类


模型下方的柱状图，表示一张图片的尺寸，因为图片输入目前绝大部分都是正方形，因此简化成柱状图，表示图片的边长，可以看到对应网络结构中，图片大小的变化。

![lenet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet.png)



将卷积层和maxpooling层画在一起

  <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/equal.png" width="305">


简化成

![lenet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet-short.png)

源码：
tensorflow 源码 https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/alexnet.py

caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt


作者源码

code: https://code.google.com/p/cuda-convnet/

github: https://github.com/dnouri/cuda-convnet

code: https://code.google.com/p/cuda-convnet2/

github: https://github.com/akrizhevsky/cuda-convnet2






# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)
