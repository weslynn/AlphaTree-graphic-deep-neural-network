
# MobileNet

MobileNet 顾名思义，可以用在移动设备上的网络，性能和效率取得了很好平衡。它发展了两个版本，第一个版本基本结构和VGG类似，主要通过 depthwise separable convolution 来减少参数和提升计算速度。第二个版本则是基于ResNet的结构进行改进。

MobileNet v2使用了 ReLU6（即对 ReLU 输出的结果进行 Clip，使得输出的最大值为 6）适配移动设备更好量化，然后提出了一种新的 Inverted Residuals and Linear Bottleneck，即 ResNet 基本结构中间使用了 depthwise 卷积，一个通道一个卷积核，减少计算量，中间的通道数比两头还多，并且全去掉了最后输出的 ReLU。

paper：

MobileNet v1：2017，MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[pdf](https://arxiv.org/pdf/1704.04861.pdf) 


MobileNet v2：2018，Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation[pdf](
https://arxiv.org/pdf/1801.04381.pdf)



MobileNet 微结构对比：

![mobilenetv2_compare](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenetv2_compare.jpg)


MobileNet 的根本思想是使用deep-wise方式的卷积在不减少精度的情况下来减少计算量。

![mobilenet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenet.jpg)

其中M为输入的通道数，Dk为卷积核的宽和高，其中DF为输入的宽和高，在某一层如果使用N个卷积核，这一个卷积层的计算量为：

![mobilenetcal](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenet_cal.jpg)

如果使用deep-wise方式，将会把卷积过程拆成两个步骤，第一步使用一组M个3×3的depth卷积，每次只处理一个输入通道的，之后第二步使用1×1×M×N的卷积核进行计算。

![mobilenetcal2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenet_cal2.jpg)

从数学上看 矩阵乘法拆解后计算量大大减小。

![mobilenetcal3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenet_cal3.jpg)



而MobileNet v2结合ResNet的基础结构进行改进，设计了Inverted Residuals and Linear Bottleneck，即 ResNet 基本结构中间使用了 depthwise 卷积，一个通道一个卷积核，减少计算量，中间的通道数比两头还多，并且全去掉了最后输出的 ReLU。


在Michael Yuan [zhihu](https://zhuanlan.zhihu.com/p/33075914) 绘制的图中加上节点表示如下：

![mobilenet_compare](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenet_compare.png)


两两对比微结构：
MobileNet V1 是基于VGG的结构， 而Mobilenet V2 是基于ResNet的结构
Q：为啥Mobilenet V2 一个有add 一个没有add 

A : 看官方代码 Mobilenet V2选取的是红色框住那一部分，第一步是stride =2 ，没有add操作， 第二次循环是stride =1 有add操作。  如果前后通道数不一致，无论stride 为多少是没有add操作的。（当然 ，如果你就是想要add，那么可以 自己加个1×1卷积，把通道数变成一致后进行add）
    if callable(residual):  # custom residual
      net = residual(input_tensor=input_tensor, output_tensor=net)
    elif (residual and
          # stride check enforces that we don't add residuals when spatial
          # dimensions are None
          stride == 1 and
          # Depth matches
          net.get_shape().as_list()[3] ==
          input_tensor.get_shape().as_list()[3]):
      net += input_tensor


![mobilenet_struct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenetv2_tip.jpg)

![MobileNetcomparepic](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mobilentv1_v2.png)

ResNet 是没有用depthwise separable convolution 的结构， 而Mobilenet V2 使用depthwise separable convolution ，还加入Inverted Residuals and Linear Bottleneck的设计

![MobileNetcomparepic1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mobilentresent.png)




MobileNet 结构如图：

![mobilenet_struct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenetv1.jpg)


MobileNet用不同节点表示如图：


![MobileNetpic](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mobilenet.png)

<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/mobilenet.png)</p>



MobileNet V2 结构如图：

![mobilenetv2_struct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/mobilenetv2.jpg)



MobileNet V2用不同节点表示如图：


![MobileNetv2pic](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/mobilenetv2.png)

<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/mobilenetv2.png)</p>


源码：


TensorFlow：

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py

caffe：https://github.com/pby5/MobileNet_Caffe


https://github.com/Zehaos/MobileNet 




# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)
# [ResNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md)