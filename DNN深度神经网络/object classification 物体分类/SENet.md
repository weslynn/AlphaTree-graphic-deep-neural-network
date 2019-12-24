
# SENet( Squeeze-and-Excitation Networks) 

paper：Squeeze-and-Excitation Networks  [pdf](https://arxiv.org/pdf/1709.01507.pdf)
Momenta 提出的SENet 获得了最后一届 ImageNet 2017 竞赛 Image Classification 任务的冠军， 2.251% Top-5 错误率

它在结构中增加了一个se模块，通过Squeeze 和 Excitation 的操作，学习自动获取每个特征通道的重要程度，然后依照这个重要程度去提升有用的特征并抑制对当前任务用处不大的特征。

这个模块可以灵活的加在任何结构中，而且模型和计算复杂度上具有良好的特性，在现有网络架构中嵌入 SE 模块而导致额外的参数和计算量的增长微乎其微。



SE 模块详细如图：

![senetmodule](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/semodule.jpg)


对比不同网络结构增加了SE模块之后：

![senet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/senet.jpg)

SENet 网络结构如下：
![senet1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/senet.png)



用不同节点表示如图：


![senetpic](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/senet.png)



<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/senet.png)</p>





源码：



caffe:  caffe:https://github.com/hujie-frank/SENet


参考：
http://www.sohu.com/a/161633191_465975


# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)
# [ResNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md)