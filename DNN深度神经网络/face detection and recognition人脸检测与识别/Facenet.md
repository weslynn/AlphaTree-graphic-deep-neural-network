
# FaceNet

和物体分类这种分类问题不同，Facenet是构建了一种框架，通过已有的深度模型，结合不同loss，训练一个很棒的人脸特征。它直接使用端对端的方法去学习一个人脸图像到欧式空间的编码，这样构建的映射空间里的距离就代表了人脸图像的相似性。然后基于这个映射空间，就可以轻松完成人脸识别，人脸验证和人脸聚类。



在LFW数据集上，准确率为99.63%，在YouTube Faces DB数据集上，准确率为95.12%，比以往准确度提升了将近 30%。 

paper ：[CVPR2015] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[J]. arXiv preprint arXiv:1503.03832, 2015.[pdf](https://arxiv.org/pdf/1503.03832.pdf) 


先验知识：相同个体的人脸的距离，总是小于不同个体的人脸


![facenet_struct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/facenet_struct.png)

它使用现有的模型结构，然后将卷积神经网络去掉sofmax后，经过L2的归一化，然后得到特征表示，之后基于这个特征表示计算Loss。文章中使用的结构是[ZFNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)，[GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)，tf代码是改用了Inception_resnet_v1。

文中使用的Loss 是 triplet loss。后来相应的改进有ECCV2016的 center loss，SphereFace，2018年的AMSoftmax和ArchFace（InsightFace），现在效果最好的是ArchFace（InsightFace）。
之前的工作有人使用的是二元损失函数，二元损失函数的目标是把相同个体的人脸特征映射到空间中的相同区域，而三元损失函数目标是相同个体的人脸特征映射到相同的区域，而且每个人的特征和其他人的特征能够分开，类内距离小于类间距离。 

![triplet_loss](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/tripleloss.png)

我们可以看到loss公式如下：

![triplet_loss1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/tripleloss1.png)

## center Loss 

A Discriminative Feature Learning Approach for Deep Face Recognition  ECCV:2016 [pdf](http://ydwen.github.io/papers/WenECCV16.pdf)

通过softmax+center loss 让简单的softmax 能够训练出更有内聚性的特征。

损失采用softmax loss，那么最后各个类别学出来的特征（MNIST）分布大概如下图

![softmax](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/softmax.png)

损失采用softmax loss+center loss的损失，那么最后各个类别的特征分布大概如下图，类间距离变大了，类内距离减少了

![centerloss](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/centerloss.png)

公式如下：

![centerloss1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/centerloss1.png)


## SphereFace:A-softmax
SphereFace: Deep Hypersphere Embedding for Face Recognition 2017cvpr
[pdf](https://arxiv.org/pdf/1704.08063.pdf)

提出了归一化权值（normalize weights and zero biases）和角度间距（angular margin），基于这2个点，对softmax，L-softmax （Large-Margin Softmax Loss for Convolutional Neural Networks ）进行了改进，提出了一种新的损失函数，叫做angular softmax(A-Softmax),从而实现了在超球面上做到特征分布高内聚、低耦合,最大类内距离小于最小类间距离的识别标准。

一般都是想在欧式空间中学习到有判别力的特征，作者提出了一个问题：欧式空间的间隔总是适合于表示学习到了有判别力的特征吗？ 



![sphereface](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/sphereface.png)

在上图中，对于一个二分类的softmax，决策边界是(W1−W2)x+b1−b2=0，假如定义||W1||=||W2||=1,b1=b2=0，那么决策边界的形式变换为||x||(cos(θ1)−cos(θ2))=0，这样设计的损失函数直接关注的是特征的角度可分性，使得训练出的CNN学习到具有角度判别力的特征。

作者在修改的softmax的基础上进一步加强了限制条件设计了A-Softmax，引入了一个整数m加大角度间隔。决策边界变形为||x||(cos(mθ1)−cos(θ2))=0和||x||(cos(θ1)−cos(mθ2))=0。通过最优化A-Softmax，决策区域更加可分，在增大类间间隔的同时压缩了类内的角度分布。


公式如下：

![sphereface1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/sphereface1.png)

最优化A-Softmax损失本质上是使得学习到的特征在超球面上更加具有可区分性。


![sphereface2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/sphereface2.png)


## CosFace :AM-softmax

关于 margin 刚刚只讲了一个，就是 Asoftmax ，其实还有其他的几种形式，第一个是AMSoftmax，还有一个叫 CosFace，这两个是同样的想法，内容也基本是一模一样的。Margin 是指的类似于 SVM 里面的那种分类面之间要有一个间隔，通过最大化这个间隔把两个类分得更开。在 AMSoftmax 里面本来 xi 和类中心的距离是 cosθ，不管是 ASoftmax 还是 AMSoftmax，都是想把这个数字变小一点，它本来相似度已经 0.99 了，我希望让它更小一点，使得它不要那么快达到我想要的值，对它做更强的限制，对它有更高的要求，所以它减 m 就相当于这个东西变得更小了，xi 属于应该在那个类的概率就更难达到 99% 或者 1，这个 θ 角必须要更小，这个概率才能更接近 1，才能达到我们想要的标准。右边也是一样的，这两个公式几乎完全一模一样，同时大家都在要求 W 的 norm 要是固定的，x 的 Norm 要是固定的，我们只关心 cos 距离


CosFace: Large Margin Cosine Loss for Deep Face Recognition

[pdf](https://arxiv.org/pdf/1801.09414.pdf)

AM-Softmax：Additive Margin Softmax for Face Verification

[pdf](https://arxiv.org/pdf/1801.05599.pdf)

![cosface](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/cosface.png)

## ArcFace deepinsight


ArcFace: Additive Angular Margin Loss for Deep Face Recognition
InsightFace: 2D and 3D Face Analysis Project[pdf](https://arxiv.org/pdf/1801.07698.pdf) 

AMSoftmax对sphereface的A-Softmax进行修改，将Cos(mθ)修改为一个新函数，Asoftmax是用m乘以θ，而AMSoftmax是用cosθ减去m，这样做的好处在于ASoftmax的倍角计算是要通过倍角公式，反向传播时不方便求导，而只减m反向传播时导数不用变化
。

ArcFace在AMSoftmax的基础上进行了改进：

![insightface1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/insightface1.png)

![insightface0](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/arcface.png)


不同loss对比：

![insightface](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/insightface.png)

![insightface2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/insightface2.png)

## 其他loss

coco loss
Rethinking Feature Discrimination and Polymerization for Large-scale Recognition

## code
 tensorflow 源码 :https://github.com/davidsandberg/facenet

 caffe center loss:https://github.com/kpzhang93/caffe-face
 mxnet center loss :https://github.com/pangyupo/mxnet_center_loss
 
 caffe sphereface:  https://github.com/wy1iu/sphereface

 deepinsight： https://github.com/deepinsight/insightface
 AMSoftmax ：https://github.com/happynear/AMSoftmax
             https://github.com/Joker316701882/Additive-Margin-Softmax

 coco loss ：https://github.com/sciencefans/coco_loss

参考文献 https://blog.csdn.net/cdknight_happy/article/details/79268613


# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)