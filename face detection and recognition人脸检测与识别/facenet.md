
# FaceNet

和最基础的Object classification 物体分类这种分类问题不同，facenet解决的是一个聚类问题。

聚类问题有一种解决方法，提取分类的深度模型某一层的特征数据进行聚类。但是FaceNet不同，它是直接使用端对端的学习一个人脸图像到欧式空间的编码方法，这样构建的映射空间里的距离就代表了人脸图像的相似型。然后基于这个映射空间，再进行人脸识别，人脸验证和人脸聚类。


在LFW数据集上，准确率为99.63%，在YouTube Faces DB数据集上，准确率为95.12%，比以往准确度提升了将近 30%。 

先验知识：相同个体的人脸的距离，总是小于不同个体的人脸

FaceNet的模型结构是使用了其他人的，文章中使用了GoogLeNet，tf代码是新改用了Inception_resnet_v1
结构的使用就是将卷积神经网络去掉sofmax后，经过L2的归一化，然后得到特征表示，之后基于这个特征表示计算Loss。

文中使用的Loss 是 triplet loss。后来相应的改进有 Central Loss 和ArchFace


训练数据来源于wider和celeba 
	Wider_face包含人脸边框groundtruth标注数据，大概人脸在20万
	CelebA包含边框标注数据和5个点的关键点信息

根据参与任务的不同，将训练数据分为四类：人脸正样本（positives）、非人脸负样本（negatives）、部分脸（partfaces）、关键点（landmark）。
三个网络，提取过程类似，但是图像尺寸不同．

	在每个batchSize中的样本比例如下，positives：negatives：partfaces：landmark = 1 ： 3 ： 1 ： 2。

	negative，IOU<0.3; positive,IOU>0.65; part face,0.4

Online Hard sample mining在线困难样本选择：在一个batch中只选择loss占前70%的样本进行BP;

paper ：[CVPR2015] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[J]. arXiv preprint arXiv:1503.03832, 2015.[pdf](https://arxiv.org/pdf/1503.03832.pdf) 



MTCNN 每个网络功能如图：

![MTCNN](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/facepic/mtcnn.png)

分为P-Net，R-Net，O-Net：

Proposal Network (P-Net)：该网络结构主要获得了人脸区域的候选窗口和边界框的回归向量。并用该边界框做回归，对候选窗口进行校准，然后通过非极大值抑制（NMS）来合并高度重叠的候选框。

Refine Network (R-Net)：该网络结构还是通过边界框回归和NMS来去掉那些false-positive区域。
只是由于该网络结构和P-Net网络结构有差异，多了全连接层，所以会取得更好的抑制false-positive的作用。

Output Network (O-Net)：该层比R-Net层又多了一层卷基层，所以处理的结果会更加精细。作用和R-Net层作用一样。但是该层对人脸区域进行了更多的监督，同时还会输出5个landmark。

返回值：人脸的10个点，以caffe代码为例是[left_eye_x,right_eye_x,nose_x,left_mouth_x,right_mouth_x,left_eye_y,right_eye_y,nose_y,left_mouth_y,right_mouth_y]

MTCNN 详细网络结构如图：

![MTCNNS](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/facepic/mtcnn_struct.png)

用不同节点表示如图：

![mtcnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/face/mtcnn.png)


 Caffe 源码：

 tensorflow 源码 :https://github.com/davidsandberg/facenet




# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)