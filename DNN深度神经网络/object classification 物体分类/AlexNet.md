# AlexNet
2012年，Alex Krizhevsky用AlexNet 在当年的ImageNet图像分类竞赛中(ILSVRC 2012)，以top-5错误率15.3%拿下第一。 他的top-5错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。它使用ReLU代替了传统的激活函数，而且网络针对多GPU训练进行了优化设计，用于提升速度。不过随着硬件发展，现在我们训练AlexNet都可以直接用简化后的代码来实现了。

1 ReLU函数作为激活函数
2 dropout
3 max-pooling
4 利用双GPU NVIDIA GTX 580训练

paper ：Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

slides: http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf

AlexNet原始结构如图

![alexnet org model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/alexnet-org.jpg)


ZF Net(2013)

Visualizing and Understanding Convolutional Neural Networks http://arxiv.org/pdf/1311.2901v3.pdf

AlexNet在2012年大出风头之后，2013年随即出现了大量的CNN模型。当年的的ILSVRC比赛胜者是来自纽约大学NYU的Matthew Zeiler以及Rob Fergus设计的模型，叫做ZF Net。它达到了11.2%的错误率。本文的主要贡献是一个改进型AlexNet的细节及其可视化特征图层feature map的表现方式。这种卷积网络可视化技术命名为“解卷积网络”deconvnet是因为它把特征投影为可见的像素点，有助于检查不同激活特征以及它们与输入空间的关系。这跟卷积层把像素投影为特征的过程是刚好相反的。

![zfnet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/zfnet.png)



可以将模型结构表示如图：

![alexnet other model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/alexnet2.png)


模型结构如下

一层卷积层： 11×11的卷积核，96个，步长位4 （stride = 4）

一层maxpooling

一层LRN

一层卷积层：5×5的卷积核，256个，pad=2

一层maxpooling

一层LRN

一层卷积层：3×3的卷积核，384个，pad=1

一层卷积层：3×3的卷积核，384个，pad=1

一层卷积层：3×3的卷积核，256个

一层maxpooling

一层全连接层：4096个隐含节点，激活函数为ReLU

一层全连接层：4096个隐含节点，激活函数为ReLU

最后通过softmax分类输出1000类



模型下方的柱状图，表示一张图片的尺寸，因为图片输入目前绝大部分都是正方形，因此简化成柱状图，表示图片的边长，可以看到对应网络结构中，图片大小的变化。

![alexnet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet.png)


但是tf官方后来给出的代码，进行了修改，将初始化选择用xavier_initializer的方法，将LRN层移除了。


![alexnet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnettf.png)

将卷积层和maxpooling层画在一起

  <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/equal.png" width="305">


简化成

![alexnet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet-short.png)


数据变化为

![alexnet data](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/alexnet_data.png)

源码：
tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/alexnet.py

caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt


作者源码

code: https://code.google.com/p/cuda-convnet/

github: https://github.com/dnouri/cuda-convnet

code: https://code.google.com/p/cuda-convnet2/

github: https://github.com/akrizhevsky/cuda-convnet2






# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 

<table>
    <tr>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md">LeNet</a></td>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md">AlexNet</a></td>
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md">GoogLeNet</a></td>
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md">Inception V3</a></td>
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md">VGG</a></td>
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md">ResNet and ResNeXt</a></td>
    </tr>    
    <tr>      
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionResnetV2.md">Inception-Resnet-V2</a></td>
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DenseNet.md">DenseNet</a></td>
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/DPN.md">DPN</a></td>
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/PolyNet.md">PolyNet</a></td>
      <td align="center"><a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/DNN%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/SENet.md">SENet</a></td>
      <td align="center"><a href="">NasNet</a></td>      
    </tr>    
</table>
