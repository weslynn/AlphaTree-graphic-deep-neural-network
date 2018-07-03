
# FaceNet

和最基础的Object classification 物体分类这种分类问题不同，Facenet是训练了一个很棒的人脸特征。它直接使用端对端的方法去学习一个人脸图像到欧式空间的编码，这样构建的映射空间里的距离就代表了人脸图像的相似性。然后基于这个映射空间，就可以轻松完成人脸识别，人脸验证和人脸聚类。


在LFW数据集上，准确率为99.63%，在YouTube Faces DB数据集上，准确率为95.12%，比以往准确度提升了将近 30%。 

paper ：[CVPR2015] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[J]. arXiv preprint arXiv:1503.03832, 2015.[pdf](https://arxiv.org/pdf/1503.03832.pdf) 


先验知识：相同个体的人脸的距离，总是小于不同个体的人脸

FaceNet的思想可以认为是一种框架。

![facenet_struct](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/facepic/facenet_struct.png)

它使用现有的模型结构，然后将卷积神经网络去掉sofmax后，经过L2的归一化，然后得到特征表示，之后基于这个特征表示计算Loss。文章中使用的结构是GoogLeNet，tf代码是改用了Inception_resnet_v1。

文中使用的Loss 是 triplet loss。后来相应的改进有 Central Loss 和ArchFace（InsightFace）
之前的工作有人使用的是二元损失函数，二元损失函数的目标是把相同个体的人脸特征映射到空间中的相同区域，而三元损失函数目标是相同个体的人脸特征映射到相同的区域，而且每个人的特征和其他人的特征能够分开，类内距离小于类间距离。 

![triplet_loss](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/facepic/tripleloss.png)

![triplet_loss1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/facepic/tripleloss1.png)

Central Loss ：A Discriminative Feature Learning Approach for Deep Face Recognition  ECCV:2016 

通过添加center loss 让简单的softmax 能够训练出更有内聚性的特征。





 tensorflow 源码 :https://github.com/davidsandberg/facenet




# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)