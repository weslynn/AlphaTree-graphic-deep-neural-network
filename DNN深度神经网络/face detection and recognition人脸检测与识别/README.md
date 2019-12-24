# 人脸检测与识别

概述 ：https://arxiv.org/pdf/1804.06655.pdf

![FaceDetection](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/facerecognition.png)


![FaceDetection](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/FaceDetection.png)




## Face Detection and Face Alignment 人脸检测与矫正
人脸检测与识别是一个研究很久的课题。传统方法之前也有了很多稳定可行的方法。而深度学习的出现，无论对检测还是识别又有了很大的提升。随着算法和代码的开源，现在很多公司都可以自己搭建一套自己的人脸检测识别系统。那么下面几篇经典论文，都是会需要接触到的。


在了解深度学习算法之前，也要了解一下传统的方法：如 harr特征（ 2004 Viola和Jones的《Robust Real-Time Face Detection》），LAP（Locally Assembled Binary）等。LAP是结合haar特征和LBP(local binary pattern)特征，把不同块的haar特征按照lbp的编码方法形成一个编码。

常见的人脸检测开源算法可以使用 opencv dlib seetaface等。seetafce采用了多种特征（LAB、SURF、SIFT）和多种分类器（boosted、MLP）的结合。



深度学习最早的代表作之一是2015年CVPR的 CascadeCNN 。

### CascadeCNN[详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/CascadeCNN.md)
H. Li, Z. Lin, X. Shen, J. Brandt, and G. Hua, “A convolutional neuralnetwork cascade for face detection,” in IEEE Conference on ComputerVision and Pattern Recognition, 2015, pp. 5325-5334.
这篇文章保留了传统人脸检测方法中Cascade的概念，级联了6个CNN，使用3种输入大小分别为12、24、48的浅层网络，一类为分类网络(12-net,24...)：2分类，判断是不是人脸，同时产生候选框，一类是矫正网络(12-Calibration-net,24...)它们是45分类（当时训练的时候将每一个正样本进行scale、x轴、y轴变换（共45种变换），生成45张图片）对候选框进行位置矫正。在每个分类网络之后接一个矫正网络用于回归人脸框的位置。

对比传统人脸检测方法，CascadeCNN将Cascade级联结构中每个stage中CNN的分类器代替了传统的分类器；2. 每个分类stage之后应用了一个矫正网络使得人脸框的位置更为精确。该论文是当时基于CNN的人脸检测方法中速度最快的

![cascade](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/cascadestruct.png)


   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/CascadeCNN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/face/cascadeCNN.png" width="400"></a>



### MTCNN [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/MTCNN.md) [zhang kaipeng](https://kpzhang93.github.io/) 乔宇 [Qiao Yu](http://mmlab.siat.ac.cn/yuqiao/) / CUHK-MMLAB & SIAT

* MTCNN 
MTCNN 将人脸检测与关键点检测放到了一起来完成。整个任务分解后让三个子网络来完成。每个网络都很浅，使用多个小网络级联，较好的完成任务。

![mtcnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/mtcnn_struct.png)
   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/MTCNN.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/face/mtcnn.png" width="400"></a>


   [1] [ECCV2016] Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks [pdf](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)


   Caffe 源码：https://github.com/kpzhang93/MTCNN_face_detection_alignment 官方

   tensorflow 源码 : https://github.com/davidsandberg/facenet/tree/master/src/align 

- project page: https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
- arxiv: https://arxiv.org/abs/1604.02878
- github(official, Matlab): https://github.com/kpzhang93/MTCNN_face_detection_alignment
- github: https://github.com/pangyupo/mxnet_mtcnn_face_detection
- github: https://github.com/DaFuCoding/MTCNN_Caffe
- github(MXNet): https://github.com/Seanlinx/mtcnn
- github: https://github.com/Pi-DeepLearning/RaspberryPi-FaceDetection-MTCNN-Caffe-With-Motion
- github(Caffe): https://github.com/foreverYoungGitHub/MTCNN
- github: https://github.com/CongWeilin/mtcnn-caffe
- github(OpenCV+OpenBlas): https://github.com/AlphaQi/MTCNN-light
- github(Tensorflow+golang): https://github.com/jdeng/goface

Face Detection using Deep Learning: An Improved Faster RCNN Approach

- intro: DeepIR Inc
- arxiv: https://arxiv.org/abs/1701.08289

Faceness-Net: Face Detection through Deep Facial Part Responses

- intro: An extended version of ICCV 2015 paper
- arxiv: https://arxiv.org/abs/1701.08393

Multi-Path Region-Based Convolutional Neural Network for Accurate Detection of Unconstrained “Hard Faces”

- intro: CVPR 2017. MP-RCNN, MP-RPN
- arxiv: https://arxiv.org/abs/1703.09145

End-To-End Face Detection and Recognition

https://arxiv.org/abs/1703.10818

Face R-CNN

https://arxiv.org/abs/1706.01061

Face Detection through Scale-Friendly Deep Convolutional Networks

https://arxiv.org/abs/1706.02863

Scale-Aware Face Detection

- intro: CVPR 2017. SenseTime & Tsinghua University
- arxiv: https://arxiv.org/abs/1706.09876

Detecting Faces Using Inside Cascaded Contextual CNN

- intro: CVPR 2017. Tencent AI Lab & SenseTime
- paper: http://ai.tencent.com/ailab/media/publications/Detecting_Faces_Using_Inside_Cascaded_Contextual_CNN.pdf

Multi-Branch Fully Convolutional Network for Face Detection

https://arxiv.org/abs/1707.06330

SSH: Single Stage Headless Face Detector

- intro: ICCV 2017. University of Maryland
- arxiv: https://arxiv.org/abs/1708.03979
- github(official, Caffe): https://github.com/mahyarnajibi/SSH

Dockerface: an easy to install and use Faster R-CNN face detector in a Docker container

https://arxiv.org/abs/1708.04370

### FaceBoxes: A CPU Real-time Face Detector with High Accuracy

- intro: IJCB 2017
- keywords: Rapidly Digested Convolutional Layers (RDCL), Multiple Scale Convolutional Layers (MSCL)
- intro: the proposed detector runs at 20 FPS on a single CPU core and 125 FPS using a GPU for VGA-resolution images
- arxiv: https://arxiv.org/abs/1708.05234
- github(official): https://github.com/sfzhang15/FaceBoxes
- github(Caffe): https://github.com/zeusees/FaceBoxes




### S3FD: Single Shot Scale-invariant Face Detector

- intro: ICCV 2017. Chinese Academy of Sciences
- intro: can run at 36 FPS on a Nvidia Titan X (Pascal) for VGA-resolution images
- arxiv: https://arxiv.org/abs/1708.05237
- github(Caffe, official): https://github.com/sfzhang15/SFD
- github: https://github.com//clcarwin/SFD_pytorch

Detecting Faces Using Region-based Fully Convolutional Networks

https://arxiv.org/abs/1709.05256

AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection

https://arxiv.org/abs/1709.07326

Face Attention Network: An effective Face Detector for the Occluded Faces

https://arxiv.org/abs/1711.07246

Feature Agglomeration Networks for Single Stage Face Detection

https://arxiv.org/abs/1712.00721

Face Detection Using Improved Faster RCNN

- intro: Huawei Cloud BU
- arxiv: https://arxiv.org/abs/1802.02142

PyramidBox: A Context-assisted Single Shot Face Detector

- intro: Baidu, Inc
- arxiv: https://arxiv.org/abs/1803.07737

PyramidBox++: High Performance Detector for Finding Tiny Face

- intro: Chinese Academy of Sciences & Baidu, Inc.
- arxiv: https://arxiv.org/abs/1904.00386

A Fast Face Detection Method via Convolutional Neural Network

- intro: Neurocomputing
- arxiv: https://arxiv.org/abs/1803.10103

Beyond Trade-off: Accelerate FCN-based Face Detector with Higher Accuracy

- intro: CVPR 2018. Beihang University & CUHK & Sensetime
- arxiv: https://arxiv.org/abs/1804.05197

### PCN:Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks

- intro: CVPR 2018
- arxiv: https://arxiv.org/abs/1804.06039
- github(binary library): https://github.com/Jack-CV/PCN

SFace: An Efficient Network for Face Detection in Large Scale Variations

- intro: Beihang University & Megvii Inc. (Face++)
- arxiv: https://arxiv.org/abs/1804.06559

Survey of Face Detection on Low-quality Images

- https://arxiv.org/abs/1804.07362

Anchor Cascade for Efficient Face Detection

- intro: The University of Sydney
- arxiv: https://arxiv.org/abs/1805.03363

Adversarial Attacks on Face Detectors using Neural Net based Constrained Optimization

- intro: IEEE MMSP
- arxiv: https://arxiv.org/abs/1805.12302

Selective Refinement Network for High Performance Face Detection

https://arxiv.org/abs/1809.02693

### DSFD: Dual Shot Face Detector

https://arxiv.org/abs/1810.10220


[介绍](https://blog.csdn.net/wwwhp/article/details/83757286)

Learning Better Features for Face Detection with Feature Fusion and Segmentation Supervision

https://arxiv.org/abs/1811.08557

FA-RPN: Floating Region Proposals for Face Detection

https://arxiv.org/abs/1812.05586

Robust and High Performance Face Detector

https://arxiv.org/abs/1901.02350

DAFE-FD: Density Aware Feature Enrichment for Face Detection

https://arxiv.org/abs/1901.05375

Improved Selective Refinement Network for Face Detection

- intro: Chinese Academy of Sciences & JD AI Research
- arxiv: https://arxiv.org/abs/1901.06651


Revisiting a single-stage method for face detection

https://arxiv.org/abs/1902.01559

MSFD:Multi-Scale Receptive Field Face Detector

- intro: ICPR 2018
- arxiv: https://arxiv.org/abs/1903.04147

LFFD: A Light and Fast Face Detector for Edge Devices

https://arxiv.org/abs/1904.10633

BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs

- intro: CVPR Workshop on Computer Vision for Augmented and Virtual Reality, 2019
- arxiv: https://arxiv.org/abs/1907.05047


## Detect Small Faces
### Finding Tiny Faces

- intro: CVPR 2017. CMU
- project page: http://www.cs.cmu.edu/~peiyunh/tiny/index.html
- arxiv: https://arxiv.org/abs/1612.04402
- github(official, Matlab): https://github.com/peiyunh/tiny
- github(inference-only): https://github.com/chinakook/hr101_mxnet
- github: https://github.com/cydonia999/Tiny_Faces_in_Tensorflow

### Detecting and counting tiny faces

- intro: ENS Paris-Saclay. ExtendedTinyFaces
- intro: Detecting and counting small objects - Analysis, review and application to counting
- arxiv: https://arxiv.org/abs/1801.06504
- github: https://github.com/alexattia/ExtendedTinyFaces

Seeing Small Faces from Robust Anchor’s Perspective

- intro: CVPR 2018
- arxiv: https://arxiv.org/abs/1802.09058

Face-MagNet: Magnifying Feature Maps to Detect Small Faces

- intro: WACV 2018
- keywords: Face Magnifier Network (Face-MageNet)
- arxiv: https://arxiv.org/abs/1803.05258
- github: https://github.com/po0ya/face-magnet

Robust Face Detection via Learning Small Faces on Hard Images

- intro: Johns Hopkins University & Stanford University
- arxiv: https://arxiv.org/abs/1811.11662
- github: https://github.com/bairdzhang/smallhardface

SFA: Small Faces Attention Face Detector

- intro: Jilin University
- arxiv: https://arxiv.org/abs/1812.08402






此外我们可以了解一下商用的一些算法：



### DenseBox Baidu

DenseBox: Unifying Landmark Localization with End to End Object Detection

arxiv: http://arxiv.org/abs/1509.04874
demo: http://pan.baidu.com/s/1mgoWWsS
KITTI result: http://www.cvlibs.net/datasets/kitti/eval_object.php

-------------------------------------------------------------
### Landmark Localization 68 points
-------------------------------------------------------------
从技术实现上可将人脸关键点检测分为2大类：生成式方法（Generative methods） 和 判别式方法（Discriminative methods）。
Generative methods 构建人脸shape和appearance的生成模型。这类方法将人脸对齐看作是一个优化问题，来寻找最优的shape和appearance参数，使得appearance模型能够最好拟合输入的人脸。这类方法包括：

ASM(Active Shape Model) 1995
AAM (Active Appearnce Model) 1998

Discriminative methods直接从appearance推断目标位置。这类方法通常通过学习独立的局部检测器或回归器来定位每个面部关键点，然后用一个全局的形状模型对预测结果进行调整，使其规范化。或者直接学习一个向量回归函数来推断整个脸部的形状。这类方法包括传统的方法以及最新的深度学习方法：

Constrained local models (CLMs) 2006 https://github.com/TadasBaltrusaitis/CLM-framework
Deformable part models (DPMs)
基于级联形状回归的方法(Cascaded regression) 2010 
     CPR(Cascaded Pose Regression) 
     ESR https://github.com/soundsilence/FaceAlignment
     ERT(Ensemble of Regression Trees)  dlib： One Millisecond Face Alignment with an Ensemble of Regression Trees.  http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf        https://github.com/davisking/dlib
     Face Alignment at 3000 FPS cvpr2013, https://github.com/yulequan/face-alignment-in-3000fps

![FacePoint](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/facepoint.png)


深度学习：


## face++  DCNN
2013 香港中文大学汤晓欧，SunYi等人作品，首次将CNN用于人脸关键点检测。总体思想是由粗到细，实现5个人脸关键点的精确定位。网络结构分为3层：level 1、level 2、level 3。每层都包含多个独立的CNN模型，负责预测部分或全部关键点位置，在此基础上平均来得到该层最终的预测结果。






## TCDCN  VanillaCNN TCNN（Tweaked Convolutional Neural Networks） 
TCDCN Facial Landmark Detection by Deep Multi-task Learning
http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
VanillaCNN


## DAN Deep-Alignment-Network

Kowalski, M.; Naruniec, J.; Trzcinski, T.: "Deep Alignment Network: A convolutional neural network for robust face alignment", CVPRW 2017

https://github.com/MarekKowalski/DeepAlignmentNetwork


tensorflow实现

zjjMaiMai's implementatation：https://github.com/zjjMaiMai/Deep-Alignment-Network-A-convolutional-neural-network-for-robust-face-alignment
mariolew's implementatation：https://github.com/mariolew/Deep-Alignment-Network-tensorflow

## LAB (LAB-Look at Boundary A Boundary-Aware Face Alignment Algorithm )
2018cvpr 清华&商汤作品。借鉴人体姿态估计，将边界信息引入关键点回归上。网络包含3个部分：边界热度图估计模块（Boundary heatmap estimator），基于边界的关键点定位模块（ Boundary-aware landmarks regressor ）和边界有效性判别模块（Boundary effectiveness discriminator）

![lab](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/lab.png)

- 边界热度图估计模块：采用stacked hourglass network 和 message passing layers。输入人脸图像，输出人脸边界热度图来表示面部的几何结构。人脸面部的各个器官边界共构成K个边界。每个stack结束时，特征图被分成K个分支，分别送给各个对应类型的边界热度图估计。最终生成的热度图与输入原始图像进行融合，作为关键点定位模块的输入。
- 基于边界的关键点定位模块，利用边界信息，通过4阶res-18网络来定位关键点
- 边界有效性判别模块，由于边界热度图在关键点定位中起着非常重要的作用，因此需要对生成的边界信息的准确性进行评判。该模块采用对抗网络，评判边界热度图的有效性。

-----------------------------------------------------------------------------------------------------------


过去二十年来，人脸识别要解决的关键问题还是如何寻找合适特征的算法，主要经过了四个阶段。

第一个阶段Holistci Learning，通过对图片进行空间转换，得到满足假设的一定分布的低维表示
，如线性子空间，稀疏表示等等。这个想法在20世纪90年代占据了FR的主导地位
2000年。

然而，一个众所周知的问题是这些理论上合理的算法无法解决很多异常的问题，当人脸变化偏离了先前的假设，算法就失效了。
在21世纪初，这个问题引起了以Local handcraft算子为主的研讨。 出现了Gabor 算子和LBP算子，及它们的多层和高维扩展。局部算子的一些不变属性表现出了强大的性能。

不幸的是，手工设计的算子缺乏独特性和紧凑性，在海量数据处理表现出局限性。
在2010年初，基于浅层学习的算法被引入，尝试用两层网络来学习，之后，出现了深度学习的方法，使用多层神经网络来进行特征提取和转换。

Osadchy, Margarita, Yann Le Cun, and Matthew L. Miller. "Synergistic face detection and pose
estimation with energy-based models." The Journal of Machine Learning Research 8 (2007): 1197-
1215.


2014年，DeepFace 和DeepID第一次在不受约束的情景超越了人类的表现。从那时起，研究
重点已转向基于深度学习的方法。 


![FaceDetection1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/facerecognition1.png)




![FaceRecognition](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/FaceRecognition.png)




## face Recognition
DeepFace是FaceBook提出来的，后续有DeepID和FaceNet出现。DeepFace是第一个真正将大数据和深度神经网络应用于人脸识别和验证的方法，人脸识别精度接近人类水平，可以谓之CNN在人脸识别的奠基之作

之后Facenet跳出了分类问题的限制，而是构建了一种框架，通过已有的深度模型，训练一个人脸特征。然后用这个人脸特征来完成人脸识别，人脸验证和人脸聚类。

### Deep Face

DeepFace 在算法上并没有什么特别的创新，它的改进在于对前面人脸预处理对齐的部分做了精细的调整，结果显示会有一定的帮助，但是也有一些疑问，因为你要用 3D Alignment（对齐），在很多情况下，尤其是极端情况下，可能会失败。


DeepFace: Closing the Gap to Human-Level Performance in Face Verification 

![DeepFace1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/deepface.jpg)

![DeepFace2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/deepface.png)

![DeepFacemodel](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/face/deepface.png)


### DeepID

DeepID 还是将人脸作为一个分类问题来解决，而从facenet开始，则是通过设计不同的loss，端对端去学习一个人脸的特征。这个特征 在欧式空间 或者高维空间，能够用距离来代表人脸的相似性。


### VGGFace

![faceloss](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/faceloss.png)


### Facenet [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/Facenet.md) 

和物体分类这种分类问题不同，Facenet是构建了一种框架，通过已有的深度模型，结合不同loss，训练一个很棒的人脸特征。它直接使用端对端的方法去学习一个人脸图像到欧式空间的编码，这样构建的映射空间里的距离就代表了人脸图像的相似性。然后基于这个映射空间，就可以轻松完成人脸识别，人脸验证和人脸聚类。

[CVPR2015] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[J]. arXiv preprint arXiv:1503.03832, 2015.[pdf](https://arxiv.org/pdf/1503.03832.pdf) 

Model name          LFW accuracy  Training dataset  Architecture

[20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE)        0.987      CASIA-WebFace    Inception ResNet v1

[20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk)        0.992       MS-Celeb-1M     Inception ResNet v1

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/face%20detection%20and%20recognition%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB/Facenet.md"> <img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/facenet_struct.png"></a>

它使用现有的模型结构，然后将卷积神经网络去掉sofmax后，经过L2的归一化，然后得到特征表示，之后基于这个特征表示计算Loss。文章中使用的结构是[ZFNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/AlexNet.md)，[GoogLeNet](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20classification%20%E7%89%A9%E4%BD%93%E5%88%86%E7%B1%BB/GoogLeNet.md)，tf代码是改用了Inception_resnet_v1。

Loss的发展：
文中使用的Loss 是 triplet loss。后来相应的改进有ECCV2016的 center loss，SphereFace，2018年的AMSoftmax和ArchFace（InsightFace），现在效果最好的是ArchFace（InsightFace）。


![loss1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/arcface.png)

![loss2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/facepic/insightface.png)

https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py


 tensorflow 源码 :https://github.com/davidsandberg/facenet

 caffe center loss:https://github.com/kpzhang93/caffe-face

 mxnet center loss :https://github.com/pangyupo/mxnet_center_loss
 
 caffe sphereface:  https://github.com/wy1iu/sphereface

 deepinsight： https://github.com/deepinsight/insightface

 AMSoftmax ：https://github.com/happynear/AMSoftmax

github：https://github.com/cmusatyalab/openface
基于谷歌的文章《FaceNet: A Unified Embedding for Face Recognition and Clustering》。openface是卡内基梅隆大学的 Brandon Amos主导的。
B. Amos, B. Ludwiczuk, M. Satyanarayanan,
"Openface: A general-purpose face recognition library with mobile applications,"
CMU-CS-16-118, CMU School of Computer Science, Tech. Rep., 2016.

### SeetaFace 

Detection: Funnel-Structured Cascade for Multi-View Face Detection with Alignment-Awareness

2016 

中科院山世光老师开源的人脸识别引擎—SeetafaceEngine，主要实现下面三个功能： 
SeetaFace Detection 
SeetaFace Alignment 
SeetaFace Identification 

github：https://github.com/seetaface/SeetaFaceEngine

### OpenFace


主要在Landmark Detection，Landmark and head pose tracking，Facial Action Unit Recognition等，其中Facial Action Unit Recognition是个比较有意思的点，该项目给出一个脸部的每个AU的回归分数和分类结果。


Detect faces with a pre-trained models from dlib or OpenCV.
Transform the face for the neural network. This repository uses dlib's real-time pose estimation with OpenCV's affine transformation to try to make the eyes and bottom lip appear in the same location on each image.

github：https://github.com/TadasBaltrusaitis/OpenFace


------------

轻量级人脸识别模型
--------------------------------------
这个研究得比较少，主要是分两个方面：

一种是设计一个小型网络，从头开始训。这种包括LmobileNetE（112M），lightCNN (A light cnn for deep face representation with noisy labels. arXiv preprint)， ShiftFaceNet（性能能有点差 LFW 96%）,MobileFaceNet等

一种是从大模型进行knowledge distillation 知识蒸馏得到小模型。包括从DeepID2 进行teacher-student训练得到MobileID，从FaceNet预训练模型继续训MobileNetV1等。


### MobileFaceNet
这个模型主要就是用类MobileNet V2的结构，加上ArcFace的loss进行训练。

--------------------------------------
## 3D Face
3D人脸重建主要有两种方式，一种是通过多摄像头或者多帧图像的关键点匹配(Stereo matching)，重建人脸的深度信息，或者深度相机，从而得到模型,另一种是通过预先训练好的人脸模型(3d morphable model)，拟合单帧或多帧RGB图像或深度图像，从而得到3d人脸模型的个性化参数。

深度学习在3d face的研究着重在第二个。

由于Blanz和Vetter在1999年提出3D Morphable Model（3DMM）（Blanz, V., Vetter, T.: A morphable model for the synthesis of 3d faces. international
conference on computer graphics and interactive techniques (1999)），成为最受欢迎的单图3D面部重建方法。早期是针对特殊点的对应关系（可以是关键点 也可以是局部特征点）来解非线性优化函数，得到3DMM系数。然而，这些方法严重依赖于高精度手工标记或者特征。

首先，2016年左右，CNN的尝试主要是用级联CNN结构来回归准确3DMM系数，解决大姿态下面部特征点定位问题。但迭代会花费大量时间


### 3DDFA: Face Alignment Across Large Poses- A 3D Solution CVPR2016

http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm


自动化所作品， 解决极端姿态下（如侧脸），一些特征点变了不可见，不同姿态下的人脸表观也存在巨大差异使得关键点定位困难等问题

本文提出一种基于3D人脸形状的定位方法3DDFA，算法框架为：
(1) 输入为100x100的RGB图像和PNCC （Projected Normalized Coordinate Code） 特征，PNCC特征的计算与当前形状相关，可以反映当前形状的信息；算法的输出为3D人脸形状模型参数
(2) 使用卷积神经网络拟合从输入到输出的映射函数，网络包含4个卷积层，3个pooling层和2个全连接层
通过级联多个卷积神经网络直至在训练集上收敛，PNCC特征会根据当前预测的人脸形状更新，并作为下一级卷积神经网络的输入。
(3) 此外，卷积神经网络的损失函数也做了精心的设计，通过引入权重，让网络优先拟合重要的形状参数，如尺度、旋转和平移；当人脸形状接近ground truth时，再考虑拟合其他形状参数
实验证明该损失函数可以提升定位模型的精度。由于参数化形状模型会限制人脸形状变形的能力，作者在使用3DDFA拟合之后，抽取HOG特征作为输入，使用线性回归来进一步提升2D特征点的定位精度。

训练3DDFA模型，需要大量的多姿态人脸样本。为此，作者基于已有的数据集如300W，利用3D信息虚拟生成不同姿态下的人脸图像，核心思想为：先预测人脸图像的深度信息，通过3D旋转来生成不同姿态下的人脸图像


### Large-Pose Face Alignment via CNN-Based Dense 3D Model Fitting PAWF

这篇文章是来自密西根州立大学的Amin Jourabloo和Xiaoming Liu的工作。 
2D的人脸形状U可以看成是3D人脸形状A通过投影变化m得到，如下图所示： 3D人脸形状模型可以表示为平均3D人脸形状 A 0 与若干表征身份、表情的基向量 A id 和 A exp 通过p参数组合而成
面部特征点定位问题（预测U）可以转变为同时预测投影矩阵m和3D人脸形状模型参数p

算法的整体框架通过级联6个卷积神经网络来完成这一任务：
(1) 首先以整张人脸图像作为输入，来预测投影矩阵的更新
(2) 使用更新后的投影矩阵计算当前的2D人脸形状，基于当前的2D人脸形状抽取块特征作为下一级卷积神经网络的输入，下一级卷积神经网络用于更新3D人脸形状
(3) 基于更新后的3D人脸形状，计算可得当前2D人脸形状的预测
(4) 根据新的2D人脸形状预测，抽取块特征输入到卷积神经网络中来更新投影矩阵，交替迭代优化求解投影矩阵m和3D人脸形状模型参数p，直到在训练集收敛

值得一提的是，该方法在预测3D人脸形状和投影矩阵的同时也考虑到计算每一个特征点是否可见。如果特征点不可见，则不使用该特征点上的块特征作为输入，这是普通2D人脸对齐方法难以实现的
此外，作者提出两种pose-invariant的特征Piecewise Affine-Warpped Feature (PAWF)和Direct 3D Projected Feature (D3PF)，可以进一步提升特征点定位的精度






## 密集人脸对齐

用cnn学习2d图像与3d图像之间的密集对应关系 然后使用预测的密集约束计算3DMM参数。


### 3dmm_cnn 

End to end 的方法，将输入图片转换为3DMM参数

Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network 2016
https://github.com/anhttran/3dmm_cnn


### DeFA： Dense Face Alignment /Pose-Invariant Face Alignment (PIFA) ICCV 2017 
http://cvlab.cse.msu.edu/project-pifa.html 

密西根州立大学的Amin Jourabloo和Xiaoming Liu等人的工作，该组其他人脸对齐的工作可参见其项目主页。

摘要： 在人脸对齐方法中，以前的算法主要集中在特定数量的人脸特征点检测，比如5、34或者68个特征点，这些方法都属于稀疏的人脸对齐算法。在本文中，我们提出了一种针对大角度人脸图像的一种3D密集人脸对齐算法。在该模型中，我们通过训练CNN模型利用人脸图像来估计3D人脸shape，利用该shape来fitting相应的3D人脸模型，不仅能够检测到人脸特征点，还能匹配人脸轮廓和SIFT特征点。此外还解决了不同数据库中由于包含不同数量的特征点（5、34或68）而不能交叉验证的问题。可以实时运行


###  DenseReg: Fully Convolutional Dense Shape Regression In-the-Wild

原文： CVPR 2017 https://github.com/ralpguler/DenseReg
摘要： 在本文中，我们提出通过完全卷积网络学习从图像像素到密集模板网格的映射。我们将此任务作为一个回归问题，并利用手动注释的面部标注来训练我们的网络。我们使用这样的标注，在三维对象模板和输入图像之间，建立密集的对应领域，然后作为训练我们的回归系统的基础。我们表明，我们可以将来自语义分割的想法与回归网络相结合，产生高精度的“量化回归”架构。我们的系统叫DenseReg，可以让我们以全卷积的方式估计密集的图像到模板的对应关系。因此，我们的网络可以提供有用的对应信息，而当用作统计可变形模型的初始化时，我们获得了标志性的本地化结果，远远超过当前最具挑战性的300W基准的最新技术。我们对大量面部分析任务的方法进行了全面评估，并且还展示了其用于其他估计任务的用途，如人耳建模。

http://alpguler.com/DenseReg.html

### FAN

How far are we from solving the 2D & 3D Face Alignment problem?)

ICCV 2017 诺丁汉大学作品。在现存2D和3D人脸对齐数据集上，本文研究的这个非常深的神经网络达到接近饱和性能的程度。本文主要做了5个贡献：（1）结合最先进的人脸特征点定位（landmark localization）架构和最先进的残差模块（residual block），首次构建了一个非常强大的基准，在一个超大2D人脸特征点数据集（facial landmark dataset）上训练，并在所有其他人脸特征点数据集上进行评估。（2）我们构建一个将2D特征点标注转换为3D标注，并所有现存数据集进行统一，构建迄今最大、最具有挑战性的3D人脸特征点数据集LS3D-W（约230000张图像）。（3）然后，训练一个神经网络来进行3D人脸对齐（face alignment），并在新的LS3D-W数据集上进行评估。（4）本文进一步研究影响人脸对齐性能的所有“传统”因素，例如大姿态( large pose)，初始化和分辨率，并引入一个“新的”因素，即网络的大小。（5）本文的测试结果显示2D和3D人脸对齐网络都实现了非常高的性能，足以证明非常可能接近所使用的数据集的饱和性能。训练和测试代码以及数据集可以从 https://www.adrianbulat.com/face-alignment/%20下载

### VRN

诺丁汉大学和金斯顿大学 用CNN Regression的方法解决大姿态下的三维人脸重建问题。 
ICCV论文：《Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression》

Volumetric Regression Network(VRN) 本文作者使用的模型，由多个沙漏模型组合在一起形成。 
- VRN模型使用两个沙漏模块堆积而成，并且没有使用hourglass的间接监督结构。 
- VRN-guided 模型是使用了Stacked Hourglass Networks for Human Pose Estimation 的工作作为基础，在前半部分使用两个沙漏模块用来获取68个标记点，后半部分使用两个沙漏模块，以一张RGB图片和68个通道（每个通道一个标记点）的标记点作为输入数据。 
- VRN-Multitask 模型，用了三个沙漏模块，第一个模块后分支两个沙漏模块，一个生成三维模型，一个生成68个标记点。 

github：https://github.com/AaronJackson/vrn


### PRNet：Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network

原文： CVPR 2017
摘要： 本文提出了一个强有力的方法来同时实现3D人脸重构和密集人脸对齐。为实现该目标，我们设计了一个UV位置图，来达到用2D图表示UV 空间内完整人脸的3D形状特征。然后训练了一个简单的CNN来通过单张2D图像回归得到UV图。我们的方法不需要任何先验人脸模型，就可以重构出完整的面部结构。速度9.8ms/帧。


https://github.com/YadiraF/PRNet

PRNet 简单来说，就是以前的一张图片三通道是RGB，表达的是二维的图片， 有没有什么方法简单的将三维问题，转换成和现有解决方案相似的问题来处理。作者将一个三维的人脸，投影到x y z 三个平面上，改用xyz作为三个通道，于是 三维的人脸 就可以还是变成三个通道来进行处理。
简单有效。

PS： 个人用CAS-PEAL-R1数据集测试了作者给的模型，人脸角度偏差在5°以内，胜过其他二维图片68个特征点很多算法的效果。

[翻译](https://blog.csdn.net/u011681952/article/details/82383518)
[jianshu](https://www.jianshu.com/p/b460e99e03b0)


HPEN High-Fidelity Pose and Expression Normalization for Face Recognition in the Wild



表情相关
ExpNet: Landmark-Free, Deep, 3D Facial Expressions

Expression-Net
https://github.com/fengju514/Expression-Net



数据集

UMDFace

MTFL(TCDCN所用)

[300W-3D]: The fitted 3D Morphable Model (3DMM) parameters of 300W samples.

[300W-3D-Face]: The fitted 3D mesh, which is needed if you do not have Basel Face Model (BFM)

### 3D-FAN ：2D-and-3D-face-alignment

How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks) ICCV2017
直接使用CNN预测heatmap以获得3D face landmark

两个github项目，在做同一件事，2d和3d的人脸对齐问题，区别在于前者是Pytorch 的代码，后者是Torch7的。 
github：https://github.com/1adrianb/face-alignment 
github: https://github.com/1adrianb/2D-and-3D-face-alignment

2D-FAN：https://www.adrianbulat.com/downloads/FaceAlignment/2D-FAN-300W.t7

3D-FAN：https://www.adrianbulat.com/downloads/FaceAlignment/3D-FAN.t7

2D-to-3D FAN：https://www.adrianbulat.com/downloads/FaceAlignment/2D-to-3D-FAN.tar.gz

3D-FAN-depth：https://www.adrianbulat.com/downloads/FaceAlignment/3D-FAN-depth


other


参考

https://www.jianshu.com/p/e4b9317a817f