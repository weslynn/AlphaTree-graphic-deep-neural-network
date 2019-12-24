
# GoogLeNet

GoogLeNet,采用InceptionModule和全局平均池化层，构建了一个22层的深度网络,使得很好地控制计算量和参数量的同时（ AlexNet 参数量的1/12），获得了非常好的分类性能.
它获得2014年ILSVRC挑战赛冠军，将Top5 的错误率降低到6.67%.

GoogLeNet这个名字也是挺有意思的，将L大写，为了向开山鼻祖的LeNet网络致敬

paper：Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)

GoogLeNet 原始结构如图

![googlenet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/googlenet_org.png)

大图参见[link](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/googlenet-nologo.png)

GoogLeNet的网络结构设计很大程度上借鉴了2014年 ICLR 的paper：Network In Network([NIN](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/NIN.md))。

它打破了常规的卷积层串联的模式，精心设计了InceptionModule，提高了参数的利用效率。此外，它去除了最后的全连接层，用全局平均池化层来取代。

![inceptionmodule](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/inceptionmodule.png)


![googlenet model](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/googlenet_th.jpeg)

用不同节点表示如图，在这里，我们将重复的结构进行了合并，而将不同的参数列在每个inception module下面。

另外在GoogLeNet inception v1中的结构，除了最终输出分类结果之外，层中间的地方还有两个地方输出了分类结果，而在后来发展的inception v3中发现这个在新结构中对结果提升作用不大。由于模型示意图只为了帮助大家快速理解，为此就在这里省去中间结果的节点。


![googlenet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/googlenet.png)

<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/googlenet.png)</p>




源码：

tensorflow 源码 https://github.com/tensorflow/models/tree/master/research/slim/nets/inception_v1.py

caffe https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt



# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)