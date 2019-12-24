
# PolyNet

PolyNet: A pursuitof structural diversity in very deep networks
 paper ：Xingcheng Zhang, Zhizhong Li, ChenChange Loy, Dahua Lin，PolyNet: A Pursuit of Structural Diversity in Very Deep Networks.2017 [pdf](https://arxiv.org/pdf/1611.05725v2.pdf)

PolyNet在ImageNet大规模图像分类测试集上获得了single-crop错误率4.25%和multi-crop错误率3.45%。在ImageNet2016的比赛中商汤科技与香港中大-商汤科技联合实验室在多项比赛中选用了这种网络结构并取得了三个单项第一的优异成绩。

PolyNet的基本概念

当模型的深度达到一定程度的时候，深度的增加对模型的提高起到的作用大大降低。因此PolyNet从结构多样性structural diversity的角度去探索模型的结构，它设计了比Inception modules更加复杂的PolyInception modules

![polynet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/polynet.png)


residual unit: 

一阶： (I+F)∙x=x+F∙x:=x+F(x)(I+F)∙x=x+F∙x:=x+F(x)

二阶：(I+F+F2)∙x:=x+F(x)+F(F(x))

|名称  |公式 |说明 |
|:---:|:---:|:---:|
|poly-2模块	| I+F+F2I+F+F2	|共享权值|
|mpoly-2模块	|I+F+GFI+F+GF	|共享一阶权值，二阶不共享|
|2-way模块	|I+F+GI+F+G	|一阶多项式网络结构|
|poly-3模块	|I+F+F2+F3I+F+F2+F3	|
|mpoly-3模块	|I+F+GF+HGFI+F+GF+HGF|	
|3-way模块	|I+F+G+HI+F+G+H	|

PolyNet 模块详细如图：

![polynetmodule](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/polynet_th.jpg)



整个网络建立在Inception-ResNet-v2的整体结构基础上，对其中的Inception模块进行了改进：

![polynet1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/basicpic/polynetstruct.png)



由于结构过于复杂，就不进行绘制，大家可以参考


模型结构图  （官方）
http://ethereon.github.io/netscope/#/gist/b22923712859813a051c796b19ce5944
https://raw.githubusercontent.com/CUHK-MMLAB/polynet/master/polynet.png


源码：
caffe：https://github.com/CUHK-MMLAB/polynet






# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/) 
# [LeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/LeNet.md)   
# [AlexNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)                  
# [GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)
# [Inception V3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/InceptionV3.md)
# [VGG](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/VGG.md)
# [ResNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/ResNet.md)