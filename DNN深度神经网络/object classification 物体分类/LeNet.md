
# LeNet
![lenet](http://yann.lecun.com/exdb/lenet/gifs/a35.gif)

LeNet-5(http://yann.lecun.com/exdb/lenet/a35.tml), 一个手写体数字识别模型，是一个广为人知的商用的卷积神经网络，
当年美国大多数银行用它来识别支票上面的手写数字。

paper ：LeCun, Yann; Léon Bottou; Yoshua Bengio; Patrick Haffner (1998). "Gradient-based learning applied to document recognition"
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

Lenet5 原始结构如，包括卷积层，降采样，卷积层，降采样，卷积层（实现全连接），全连接层，高斯连接层（进行分类）

![lenet org model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/lenet-org.jpg)

其中降采样层在后期发展中被maxpooling所取代，分类也被softmax所替代，因此在看tensorflow 或者其他框架下网络实现的时候，可以将模型结构表示如图：

一层卷积层： 5×5的卷积核，6个

一层maxpooling

一层卷积层：5×5的卷积核，16个

一层maxpooling

一层卷积层：5×5的卷积核，120个

一层全连接层：84个隐含节点，激活函数为ReLU（paper中激活函数为sigmoid）

最后通过softmax分类输出（paper之前为一个高斯连接层，由Euclidean Radial Basis Function单元组成）

模型下方的柱状图，表示一张图片的尺寸，因为图片输入目前绝大部分都是正方形，因此简化成柱状图，表示图片的边长，可以看到对应网络结构中，图片大小的变化。

![lenet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/lenet.png)


将卷积层和maxpooling层画在一起

  <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/equal.png" width="305">

简化成

![lenet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/lenet-short.png)

数据变化为

![lenet data](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/lenet_data2.png)

源码：

tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/lenet.py

tensorflow的输入 改成了28×28，因此少了一层卷积层，最后使用softmax输出

pytorch 源码 https://github.com/pytorch/examples/blob/master/mnist/main.py

caffe https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt



# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)