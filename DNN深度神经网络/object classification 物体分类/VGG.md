
# VGG


VGG-Net是2014年ILSVRC classification第二名(第一名是GoogLeNet)，ILSVRC localization 第一名。VGG-Net的所有 convolutional layer 使用同样大小的 convolutional filter，大小为 3 x 3

paper：Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014). [pdf](https://arxiv.org/pdf/1409.1556.pdf)

VGG-Net 原始结构如图

![vggnet-org](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/vgg_org.png)

结构如图

![link](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/vgg.png)

将卷积层和maxpooling层画在一起

  <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/equal.png" width="305">

VGG-Net 有五个stage，VGG-11 VGG-13 VGG-16 VGG-19 主要就是每个stage中的卷积层数目不同。现在将网络结构用不同节点表示如下：


![vggnet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/vgg.png)


单独看VGG-19：

![vggnet19-org](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/vgg19.png)

重新表述如下：

![vggnet19](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/vgg19.png)




源码：

   tensorflow 源码: https://github.com/tensorflow/models/tree/master/research/slim/nets/vgg.py


   caffe ：

      vgg16 https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt

      vgg19 https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt





# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)