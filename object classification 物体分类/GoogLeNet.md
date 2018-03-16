
# GoogLeNet

GoogLeNet,采用InceptionModule和全局平均池化层，构建了一个22层的深度网络,使得很好地控制计算量和参数量的同时（ AlexNet 参数量的1/12），获得了非常好的分类性能.
它获得2014年ILSVRC挑战赛冠军，将Top5 的错误率降低到6.67%.

GoogLeNet这个名字也是挺有意思的，将L大写，为了向开山鼻祖的LeNet网络致敬


GoogLeNet 原始结构如图

![googlenet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/googlenet_org.png)

大图参见[link](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/googlenet-nologo.png)

GoogLeNet的网络结构设计很大程度上借鉴了2014年 ICLR 的paper：Network In Network([NIN](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/NIN.md))。

它打破了常规的卷积层串联的模式，精心设计了InceptionModule，提高了参数的利用效率。此外，它去除了最后的全连接层，用全局平均池化层来取代。

![inceptionmodule](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/inceptionmodule.png)


![googlenet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/googlenet_th.jpeg)

用不同节点表示如图：


![googlenet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/googlenet.png)



源码：

tensorflow 源码 https://github.com/tensorflow/models/blob/57014e4c7a8a5cd8bdcb836587a094c082c991fc/research/slim/nets/inception_v1.py

caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt