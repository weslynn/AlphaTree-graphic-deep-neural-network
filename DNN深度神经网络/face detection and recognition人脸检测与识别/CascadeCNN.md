
# CascadeCNN

这个结构没有去对代码。作为开创意义的代码，准确率不会特别高。主要是思想可以参考

H. Li, Z. Lin, X. Shen, J. Brandt, and G. Hua, “A convolutional neuralnetwork cascade for face detection,” in IEEE Conference on ComputerVision and Pattern Recognition, 2015, pp. 5325-5334.
这篇文章保留了传统人脸检测方法中Cascade的概念，级联了6个CNN，使用3种输入大小分别为12、24、48的浅层网络，一类为分类网络(12-net,24...)：2分类，判断是不是人脸，同时产生候选框，一类是矫正网络(12-Calibration-net,24...)它们是45分类（当时训练的时候将每一个正样本进行scale、x轴、y轴变换（共45种变换），生成45张图片）对候选框进行位置矫正。在每个分类网络之后接一个矫正网络用于回归人脸框的位置。


![cascade](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/cascade.jpeg)

对比传统人脸检测方法，CascadeCNN将Cascade级联结构中每个stage中CNN的分类器代替了传统的分类器；2. 每个分类stage之后应用了一个矫正网络使得人脸框的位置更为精确。该论文是当时基于CNN的人脸检测方法中速度最快的

1.第一级12-net要尽量快，保证召回率的同时，过滤大量非人脸，在每一个尺度，使用NMS，recall达到99% 
2.第二级24-net的输入为第一级检测的样本，首先与gt匹配，标定出正负样本，然后分类，另外，第二级使用了多尺度，即将两个stage的fc进行concate。同样，在每一个尺度，使用NMS，recall达到97% 
3.第三级48-net网格稍微加深，过程同stage2，本次所有的尺度一块使用NMS 
4.回归网络使用bounding box的方式



训练数据AFLW
作者先从AFLW数据集中的图片中进行裁剪获得人脸图片作为正样本，再从背景中裁剪获得负样本；


![cascadestruct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/cascadestruct.png)

<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/face/cascadeCNN.png" width="500">


CascadeCNN 训练网络结构如图：

![cascadestruct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/cascadetrain.jpeg)




<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/face/cascadeCNNtrain.png" width="500">


https://github.com/anson0910/CNN_face_detection



# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)