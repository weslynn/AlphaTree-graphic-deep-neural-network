
# MTCNN

MTCNN 将人脸检测与关键点检测放到了一起来完成。整个任务分解后让三个子网络来完成。每个网络都很浅，使用多个小网络级联，较好的完成任务。


训练数据来源于wider和celeba 
	Wider_face包含人脸边框groundtruth标注数据，大概人脸在20万
	CelebA包含边框标注数据和5个点的关键点信息

根据参与任务的不同，将训练数据分为四类：人脸正样本（positives）、非人脸负样本（negatives）、部分脸（partfaces）、关键点（landmark）。
三个网络，提取过程类似，但是图像尺寸不同．

	在每个batchSize中的样本比例如下，positives：negatives：partfaces：landmark = 1 ： 3 ： 1 ： 2。

	negative，IOU<0.3; positive,IOU>0.65; part face,0.4

Online Hard sample mining在线困难样本选择：在一个batch中只选择loss占前70%的样本进行BP;

paper ：[ECCV2016] Zhi Tian, Weilin Huang, Tong He, Pan He, Yu Qiao，Detecting Text in Natural Image with Connectionist Text Proposal Network [pdf](https://arxiv.org/pdf/1609.03605.pdf) 

MTCNN 每个网络功能如图：

![MTCNN](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/mtcnn.png)

分为P-Net，R-Net，O-Net：

Proposal Network (P-Net)：该网络结构主要获得了人脸区域的候选窗口和边界框的回归向量。并用该边界框做回归，对候选窗口进行校准，然后通过非极大值抑制（NMS）来合并高度重叠的候选框。

Refine Network (R-Net)：该网络结构还是通过边界框回归和NMS来去掉那些false-positive区域。
只是由于该网络结构和P-Net网络结构有差异，多了全连接层，所以会取得更好的抑制false-positive的作用。

Output Network (O-Net)：该层比R-Net层又多了一层卷基层，所以处理的结果会更加精细。作用和R-Net层作用一样。但是该层对人脸区域进行了更多的监督，同时还会输出5个landmark。

返回值：人脸的10个点，以caffe代码为例是[left_eye_x,right_eye_x,nose_x,left_mouth_x,right_mouth_x,left_eye_y,right_eye_y,nose_y,left_mouth_y,right_mouth_y]

MTCNN 详细网络结构如图：

![MTCNNS](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/mtcnn_struct.png)

用不同节点表示如图：

![mtcnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/face/mtcnn.png)


 Caffe 源码：https://github.com/kpzhang93/MTCNN_face_detection_alignment 官方

 tensorflow 源码 : https://github.com/davidsandberg/facenet/tree/master/src/align 



# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)