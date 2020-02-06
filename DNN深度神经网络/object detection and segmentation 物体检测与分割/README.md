# Object Detection 物体检测

这里借用一张图，展示Object Detection 基础算法的发展

![total](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/total.png)

其中RCNN FastRCNN FasterRCNN为一脉相承。另外两个方向为Yolo 和SSD。Yolo迭代到Yolo V3，SSD的设计也让它后来在很多方向都有应用。

Christian Szegedy / Google 用AlexNet也做过物体检测的尝试。

   [1] Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "Deep neural networks for object detection." Advances in Neural Information Processing Systems. 2013.  [pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)


不过真正取得巨大突破，引发基于深度学习目标检测的热潮的还是RCNN

但是如果将如何检测出区域，按照回归问题的思路去解决，预测出（x,y,w,h）四个参数的值，从而得出方框的位置。回归问题的训练参数收敛时间要长很多，于是将回归问题转成分类问题来解决。总共两个步骤：

第一步：将图片转换成不同大小的框，
第二步：对框内的数据进行特征提取，然后通过分类器判定，选区分最高的框作为物体定位框。

![old](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/old.png)


![scorecompare](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/compare.png)


评价标准: IoU(Intersection over Union)； mAP(Mean Average Precision) 速度：帧率FPS
![iou](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/iou.png)


![obj](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/obj.png)

[link](https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html#non-maximum-suppression-nms)

- [Method](##Method)
  -[SPPNet]

- [Two-Stage Object Detection】
  - [R-CNN]
  - [Fast R-CNN]
  - [Faster R-CNN]

- [Single-Shot Object Detection]
  - [YOLO]
  - [YOLOv2]
  - [YOLOv3]
  - [SSD]
  - [RetinaNet]

- [Great improvement]
 - [R-FCN]
 - Feature Pyramid Network (FPN)


## Method

### SPPNet 何凯明 [He Kaiming](http://kaiminghe.com/) /MSRA
* SPPNet Spatial Pyramid Pooling（空间金字塔池化）
   [3] He, Kaiming, et al. "Spatial pyramid pooling in deep convolutional networks for visual recognition." European Conference on Computer Vision. Springer International Publishing, 2014. [pdf](http://arxiv.org/pdf/1406.4729)

一般CNN后接全连接层或者分类器，他们都需要固定的输入尺寸，因此不得不对输入数据进行crop或者warp，这些预处理会造成数据的丢失或几何的失真。SPP Net的提出，将金字塔思想加入到CNN，实现了数据的多尺度输入。此时网络的输入可以是任意尺度的，在SPP layer中每一个pooling的filter会根据输入调整大小，而SPP的输出尺度始终是固定的。

![spp](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/spp.png)

这样打破了之前大家认为需要先提出检测框，然后resize到一个固定尺寸再通过CNN的模式，而可以图片先通过CNN获取到特征后，在特征图上使用不同的检测框提取特征。之后pooling到同样尺寸进行后续步骤。这样可以提高物体检测速度。

- intro: ECCV 2014 / TPAMI 2015
- keywords: SPP-Net
- arxiv: http://arxiv.org/abs/1406.4729
- github: https://github.com/ShaoqingRen/SPP_net
- notes: http://zhangliliang.com/2014/09/13/paper-note-sppnet/


## Two-Stage Object Detection

### RCNN  Ross B. Girshick(RBG) [link](https://people.eecs.berkeley.edu/~rbg/index.html) / UC-Berkeley

* RCNN R-CNN框架，取代传统目标检测使用的滑动窗口+手工设计特征，而使用CNN来进行特征提取。这是深度神经网络的应用。

Traditional region proposal methods + CNN classifier

也就是将第二步改成了深度神经网络提取特征。
然后通过线性svm分类器识别对象的的类别，再通过回归模型用于收紧边界框；
创新点：将CNN用在物体检测上，提高了检测率。
缺点： 基于选择性搜索算法为每个图像提取2,000个候选区域，使用CNN为每个图像区域提取特征，重复计算，速度慢，40-50秒。

R-CNN在PASCAL VOC2007上的检测结果提升到66%(mAP)

![rcnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/rcnn.png)


   [2] SGirshick, Ross, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)

github: https://github.com/rbgirshick/rcnn

intro: R-CNN
arxiv: http://arxiv.org/abs/1311.2524
supp: http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf
slides: http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf
slides: http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf
github: https://github.com/rbgirshick/rcnn
notes: http://zhangliliang.com/2014/07/23/paper-note-rcnn/
caffe-pr(“Make R-CNN the Caffe detection example”): https://github.com/BVLC/caffe/pull/482


### Fast RCNN Ross B. Girshick
* Fast RCNN
   [4] Girshick, Ross. "Fast r-cnn." Proceedings of the IEEE International Conference on Computer Vision. 2015.

如果RCNN的卷积计算只需要计算一次，那么速度就可以很快降下来了。

Ross Girshick将SPPNet的方法应用到RCNN中，提出了一个可以看做单层sppnet的网络层，叫做ROI Pooling，这个网络层可以把不同大小的输入映射到一个固定尺度的特征向量.将图像输出到CNN生成卷积特征映射。使用这些特征图结合候选区域算法提取候选区域。然后，使用RoI池化层将所有可能的区域重新整形为固定大小，以便将其馈送到全连接网络中。

1.首先将图像作为输入；
2.将图像传递给卷积神经网络，计算卷积后的特征。
3.然后通过之前proposal的方法提取ROI，在所有的感兴趣的区域上应用RoI池化层，并调整区域的尺寸。然后，每个区域被传递到全连接层的网络中；
4.softmax层用于全连接网以输出类别。与softmax层一起，也并行使用线性回归层，以输出预测类的边界框坐标。
      
![fastrcnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/fastrcnn.png)


Fast R-CNN

arxiv: http://arxiv.org/abs/1504.08083
slides: http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf
github: https://github.com/rbgirshick/fast-rcnn
github(COCO-branch): https://github.com/rbgirshick/fast-rcnn/tree/coco
webcam demo: https://github.com/rbgirshick/fast-rcnn/pull/29
notes: http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/
notes: http://blog.csdn.net/linj_m/article/details/48930179
github(“Fast R-CNN in MXNet”): https://github.com/precedenceguo/mx-rcnn
github: https://github.com/mahyarnajibi/fast-rcnn-torch
github: https://github.com/apple2373/chainer-simple-fast-rnn
github: https://github.com/zplizzi/tensorflow-fast-rcnn

A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection

intro: CVPR 2017
arxiv: https://arxiv.org/abs/1704.03414
paper: http://abhinavsh.info/papers/pdfs/adversarial_object_detection.pdf
github(Caffe): https://github.com/xiaolonw/adversarial-frcnn

### Faster RCNN 何凯明 [He Kaiming](http://kaiminghe.com/)
* Faster RCNN
Fast RCNN的区域提取还是使用的传统方法，而Faster RCNN将Region Proposal Network和特征提取、目标分类和边框回归统一到了一个框架中。

Faster R-CNN = Region Proposal Network +Fast R-CNN

![fasterrcnn1](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/fasterrcnn1.png)


![fasterrcnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/fasterrcnn.png)


![fasterrcnn2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/fasterrcnn2.png)

将区域提取通过一个CNN完成。这个CNN叫做Region Proposal Network，RPN的运用使得region proposal的额外开销就只有一个两层网络。关于RPN可以参考[link](https://cloud.tencent.com/developer/article/1347839)


![rpn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/rpn.png)


Faster R-CNN设计了提取候选区域的网络RPN，代替了费时的Selective Search（选择性搜索），使得检测速度大幅提升，下表对比了R-CNN、Fast R-CNN、Faster R-CNN的检测速度：

![speed](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/speed.png)


   [5] Ren, Shaoqing, et al. "Faster R-CNN: Towards real-time object detection with region proposal networks." Advances in neural information processing systems. 2015.


Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

- intro: NIPS 2015
- arxiv: http://arxiv.org/abs/1506.01497
- gitxiv: http://www.gitxiv.com/posts/8pfpcvefDYn2gSgXk/faster-r-cnn-towards-real-time-object-detection-with-region
- slides: http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf
- github(official, Matlab): https://github.com/ShaoqingRen/faster_rcnn
- github: https://github.com/rbgirshick/py-faster-rcnn
- github(MXNet): https://github.com/msracver/Deformable-ConvNets/tree/master/faster_rcnn
- github: https://github.com//jwyang/faster-rcnn.pytorch
- github: https://github.com/mitmul/chainer-faster-rcnn
- github: https://github.com/andreaskoepf/faster-rcnn.torch
- github: https://github.com/ruotianluo/Faster-RCNN-Densecap-torch
- github: https://github.com/smallcorgi/Faster-RCNN_TF
- github: https://github.com/CharlesShang/TFFRCNN
- github(C++ demo): https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus
- github: https://github.com/yhenon/keras-frcnn
- github: https://github.com/Eniac-Xie/faster-rcnn-resnet
- github(C++): https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev

R-CNN minus R
- intro: BMVC 2015
- arxiv: http://arxiv.org/abs/1506.06981


Faster R-CNN in MXNet with distributed implementation and data parallelization
- github: https://github.com/dmlc/mxnet/tree/master/example/rcnn

Contextual Priming and Feedback for Faster R-CNN
- intro: ECCV 2016. Carnegie Mellon University
- paper: http://abhinavsh.info/context_priming_feedback.pdf
- poster: http://www.eccv2016.org/files/posters/P-1A-20.pdf

An Implementation of Faster RCNN with Study for Region Sampling
- intro: Technical Report, 3 pages. CMU
- arxiv: https://arxiv.org/abs/1702.02138
- github: https://github.com/endernewton/tf-faster-rcnn

Interpretable R-CNN
- intro: North Carolina State University & Alibaba
- keywords: AND-OR Graph (AOG)
- arxiv: https://arxiv.org/abs/1711.05226

Light-Head R-CNN: In Defense of Two-Stage Object Detector
- intro: Tsinghua University & Megvii Inc
- arxiv: https://arxiv.org/abs/1711.07264
- github(official, Tensorflow): https://github.com/zengarden/light_head_rcnn
- github: https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784

Cascade R-CNN: Delving into High Quality Object Detection
- intro: CVPR 2018. UC San Diego
- arxiv: https://arxiv.org/abs/1712.00726
- github(Caffe, official): https://github.com/zhaoweicai/cascade-rcnn

Cascade R-CNN: High Quality Object Detection and Instance Segmentation
- https://arxiv.org/abs/1906.09756
- github(Caffe, official): https://github.com/zhaoweicai/cascade-rcnn
- github(official): https://github.com/zhaoweicai/Detectron-Cascade-RCNN

SMC Faster R-CNN: Toward a scene-specialized multi-object detector
- https://arxiv.org/abs/1706.10217

Domain Adaptive Faster R-CNN for Object Detection in the Wild
- intro: CVPR 2018. ETH Zurich & ESAT/PSI
- arxiv: https://arxiv.org/abs/1803.03243
- github(official. Caffe): https://github.com/yuhuayc/da-faster-rcnn

Robust Physical Adversarial Attack on Faster R-CNN Object Detector
- https://arxiv.org/abs/1804.05810

Auto-Context R-CNN
- intro: Rejected by ECCV18
- arxiv: https://arxiv.org/abs/1807.02842

Grid R-CNN
- intro: SenseTime
- arxiv: https://arxiv.org/abs/1811.12030

Grid R-CNN Plus: Faster and Better
- intro: SenseTime Research & CUHK & Beihang University
- arxiv: https://arxiv.org/abs/1906.05688
- github: https://github.com/STVIR/Grid-R-CNN

Few-shot Adaptive Faster R-CNN
- intro: CVPR 2019
- arxiv: https://arxiv.org/abs/1903.09372

Libra R-CNN: Towards Balanced Learning for Object Detection

- intro: CVPR 2019
- arxiv: https://arxiv.org/abs/1904.02701

Rethinking Classification and Localization in R-CNN

- intro: Northeastern University & Microsoft
- arxiv: https://arxiv.org/abs/1904.06493

Reprojection R-CNN: A Fast and Accurate Object Detector for 360° Images

- intro: Peking University
- arxiv: https://arxiv.org/abs/1907.11830
- Rethinking Classification and Localization for Cascade R-CNN

- intro: BMVC 2019
- arxiv: https://arxiv.org/abs/1907.11914

## Single-Shot Object Detection

### Yolo
* Yolo(You only look once)

   ![yolologo](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/yolologo.png)

   YOLO的检测思想不同于R-CNN系列的思想，它将目标检测作为回归任务来解决。YOLO 的核心思想就是利用整张图作为网络的输入，直接在输出层回归 bounding box（边界框） 的位置及其所属的类别。

   ![yolo](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/yolo.jpg)

   ![yolo](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/yolo.png)

   [6] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." arXiv preprint arXiv:1506.02640 (2015). [pdf](https://arxiv.org/pdf/1506.02640.pdf)YOLO,Oustanding Work, really practical
  [PPT](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_1822)

c 官方:  https://pjreddie.com/darknet/yolo/   v3
         https://pjreddie.com/darknet/yolov2/ v2
         https://pjreddie.com/darknet/yolov1/ v1

pytorch (tencent) v1, v2, v3 :https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet

yolo 介绍 可以参考[介绍](https://blog.csdn.net/App_12062011/article/details/77554288)

- arxiv: http://arxiv.org/abs/1506.02640
- code: http://pjreddie.com/darknet/yolo/
- github: https://github.com/pjreddie/darknet
- blog: https://pjreddie.com/publications/yolo/
- slides: https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p
- reddit: https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/
- github: https://github.com/gliese581gg/YOLO_tensorflow
- github: https://github.com/xingwangsfu/caffe-yolo
- github: https://github.com/frankzhangrui/Darknet-Yolo
- github: https://github.com/BriSkyHekun/py-darknet-yolo
- github: https://github.com/tommy-qichang/yolo.torch
- github: https://github.com/frischzenger/yolo-windows
- github: https://github.com/AlexeyAB/yolo-windows
- github: https://github.com/nilboy/tensorflow-yolo

darkflow - translate darknet to tensorflow. Load trained weights, retrain/fine-tune them using tensorflow, export constant graph def to C++

- blog: https://thtrieu.github.io/notes/yolo-tensorflow-graph-buffer-cpp
- github: https://github.com/thtrieu/darkflow


Start Training YOLO with Our Own Data
- intro: train with customized data and class numbers/labels. Linux / Windows version for darknet.
- blog: http://guanghan.info/blog/en/my-works/train-yolo/
- github: https://github.com/Guanghan/darknet

YOLO: Core ML versus MPSNNGraph
- intro: Tiny YOLO for iOS implemented using CoreML but also using the new MPS graph API.
- blog: http://machinethink.net/blog/yolo-coreml-versus-mps-graph/
- github: https://github.com/hollance/YOLO-CoreML-MPSNNGraph

TensorFlow YOLO object detection on Android
- intro: Real-time object detection on Android using the YOLO network with TensorFlow
- github: https://github.com/natanielruiz/android-yolo

Computer Vision in iOS – Object Detection
- blog: https://sriraghu.com/2017/07/12/computer-vision-in-ios-object-detection/
- github:https://github.com/r4ghu/iOS-CoreML-Yolo

### YOLOv2

YOLO9000: Better, Faster, Stronger

- arxiv: https://arxiv.org/abs/1612.08242
- code: http://pjreddie.com/yolo9000/
- github(Chainer): https://github.com/leetenki/YOLOv2
- github(Keras): https://github.com/allanzelener/YAD2K
- github(PyTorch): https://github.com/longcw/yolo2-pytorch
- github(Tensorflow): https://github.com/hizhangp/yolo_tensorflow
- github(Windows): https://github.com/AlexeyAB/darknet
- github: https://github.com/choasUp/caffe-yolo9000
- github: https://github.com/philipperemy/yolo-9000

darknet_scripts
- intro: Auxilary scripts to work with (YOLO) darknet deep learning famework. AKA -> How to generate YOLO anchors?
- github: https://github.com/Jumabek/darknet_scripts

Yolo_mark: GUI for marking bounded boxes of objects in images for training Yolo v2
- github: https://github.com/AlexeyAB/Yolo_mark

LightNet: Bringing pjreddie’s DarkNet out of the shadows
- https://github.com//explosion/lightnet

YOLO v2 Bounding Box Tool
- intro: Bounding box labeler tool to generate the training data in the format YOLO v2 requires.
- github: https://github.com/Cartucho/yolo-boundingbox-labeler-GUI

### YOLOv3
YOLOv3: An Incremental Improvement

- project page: https://pjreddie.com/darknet/yolo/
- paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf
- arxiv: https://arxiv.org/abs/1804.02767
- github: https://github.com/DeNA/PyTorch_YOLOv3
- github: https://github.com/eriklindernoren/PyTorch-YOLOv3

Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving
- https://arxiv.org/abs/1904.04620

YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers
- https://arxiv.org/abs/1811.05588

Spiking-YOLO: Spiking Neural Network for Real-time Object Detection
- https://arxiv.org/abs/1903.06530

### SSD(The Single Shot Detector) [详解 detail](https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20detection%20%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B/SSD.md)

* SSD SSD是一种直接预测bounding box的坐标和类别的object detection算法，没有生成proposal的过程。它使用object classification的模型作为base network，如VGG16网络，


   ![ssd](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/ssd.jpg)

   <a href="https://github.com/weslynn/graphic-deep-neural-network/blob/master/object%20detection%20%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B/SSD.md"><img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/objdetection/ssd.png" width="805"></a>

   [7] Liu, Wei, et al. "SSD: Single Shot MultiBox Detector." arXiv preprint arXiv:1512.02325 (2015). [pdf](https://arxiv.org/pdf/1512.02325.pdf)  


   tensorflow 源码 https://github.com/balancap/SSD-Tensorflow/blob/master/nets/ssd_vgg_300.py

   caffe ：https://github.com/weiliu89/caffe/tree/ssd

- intro: ECCV 2016 Oral
- arxiv: http://arxiv.org/abs/1512.02325
- paper: http://www.cs.unc.edu/~wliu/papers/ssd.pdf
- slides: http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf
- github(Official): https://github.com/weiliu89/caffe/tree/ssd
- video: http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973
- github: https://github.com/zhreshold/mxnet-ssd
- github: https://github.com/zhreshold/mxnet-ssd.cpp
- github: https://github.com/rykov8/ssd_keras
- github: https://github.com/balancap/SSD-Tensorflow
- github: https://github.com/amdegroot/ssd.pytorch
- github(Caffe): https://github.com/chuanqi305/MobileNet-SSD

What’s the diffience in performance between this new code you pushed and the previous code? 
- https://github.com/weiliu89/caffe/issues/327

DSSD : Deconvolutional Single Shot Detector
- intro: UNC Chapel Hill & Amazon Inc
- arxiv: https://arxiv.org/abs/1701.06659
- github: https://github.com/chengyangfu/caffe/tree/dssd
- github: https://github.com/MTCloudVision/mxnet-dssd
- demo: http://120.52.72.53/www.cs.unc.edu/c3pr90ntc0td/~cyfu/dssd_lalaland.mp4

Enhancement of SSD by concatenating feature maps for object detection
- intro: rainbow SSD (R-SSD)
- arxiv: https://arxiv.org/abs/1705.09587

Context-aware Single-Shot Detector
- keywords: CSSD, DiCSSD, DeCSSD, effective receptive fields (ERFs), theoretical receptive fields (TRFs)
- arxiv: https://arxiv.org/abs/1707.08682

Feature-Fused SSD: Fast Detection for Small Objects
- https://arxiv.org/abs/1709.05054

FSSD: Feature Fusion Single Shot Multibox Detector
- https://arxiv.org/abs/1712.00960

Weaving Multi-scale Context for Single Shot Detector
- intro: WeaveNet
- keywords: fuse multi-scale information
- arxiv: https://arxiv.org/abs/1712.03149

Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network
- keywords: ESSD
- arxiv: https://arxiv.org/abs/1801.05918

Tiny SSD: A Tiny Single-shot Detection Deep Convolutional Neural Network for Real-time Embedded Object Detection
- https://arxiv.org/abs/1802.06488

MDSSD: Multi-scale Deconvolutional Single Shot Detector for small objects
- intro: Zhengzhou University
- arxiv: https://arxiv.org/abs/1805.07009

Accurate Single Stage Detector Using Recurrent Rolling Convolution
- intro: CVPR 2017. SenseTime
- keywords: Recurrent Rolling Convolution (RRC)
- arxiv: https://arxiv.org/abs/1704.05776
- github: https://github.com/xiaohaoChen/rrc_detection

Residual Features and Unified Prediction Network for Single Stage Detection
- https://arxiv.org/abs/1707.05031


### FPN
FPN（feature pyramid networks）特征金字塔，是一种融合了多层特征信息的特征提取方法，可以结合各种深度神经网络使用。
SSD的多尺度特征融合的方式，没有上采样过程，没有用到足够低层的特征（在SSD中，最低层的特征是VGG网络的conv4_3）

   ![fpn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/fpn.JPG)


Feature Pyramid Networks for Object Detection [pdf](https://arxiv.org/pdf/1612.03144.pdf)

Feature Pyramid Networks for Object Detection

- intro: Facebook AI Research
- arxiv: https://arxiv.org/abs/1612.03144

Action-Driven Object Detection with Top-Down Visual Attentions

arxiv: https://arxiv.org/abs/1612.06704

Beyond Skip Connections: Top-Down Modulation for Object Detection

- intro: CMU & UC Berkeley & Google Research
- arxiv: https://arxiv.org/abs/1612.06851

Wide-Residual-Inception Networks for Real-time Object Detection

- intro: Inha University
- arxiv: https://arxiv.org/abs/1702.01243

Attentional Network for Visual Object Detection

- intro: University of Maryland & Mitsubishi Electric Research Laboratories
- arxiv: https://arxiv.org/abs/1702.01478

Learning Chained Deep Features and Classifiers for Cascade in Object Detection

- keykwords: CC-Net
- intro: chained cascade network (CC-Net). 81.1% mAP on PASCAL VOC 2007
- arxiv: https://arxiv.org/abs/1702.07054

DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling

- intro: ICCV 2017 (poster)
- arxiv: https://arxiv.org/abs/1703.10295

Discriminative Bimodal Networks for Visual Localization and Detection with Natural Language Queries

- intro: CVPR 2017
- arxiv: https://arxiv.org/abs/1704.03944

Spatial Memory for Context Reasoning in Object Detection

- arxiv: https://arxiv.org/abs/1704.04224

Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection

https://arxiv.org/abs/1704.05775

LCDet: Low-Complexity Fully-Convolutional Neural Networks for Object Detection in Embedded Systems

- intro: Embedded Vision Workshop in CVPR. UC San Diego & Qualcomm Inc
- arxiv: https://arxiv.org/abs/1705.05922

Point Linking Network for Object Detection

- intro: Point Linking Network (PLN)
- arxiv: https://arxiv.org/abs/1706.03646

Perceptual Generative Adversarial Networks for Small Object Detection

https://arxiv.org/abs/1706.05274

Few-shot Object Detection

https://arxiv.org/abs/1706.08249

Yes-Net: An effective Detector Based on Global Information

https://arxiv.org/abs/1706.09180

Towards lightweight convolutional neural networks for object detection

https://arxiv.org/abs/1707.01395

RON: Reverse Connection with Objectness Prior Networks for Object Detection

- intro: CVPR 2017
- arxiv: https://arxiv.org/abs/1707.01691
- github: https://github.com/taokong/RON

Deformable Part-based Fully Convolutional Network for Object Detection

- intro: BMVC 2017 (oral). Sorbonne Universités & CEDRIC
- arxiv: https://arxiv.org/abs/1707.06175

Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors

- intro: ICCV 2017
- arxiv: https://arxiv.org/abs/1707.06399

Recurrent Scale Approximation for Object Detection in CNN

- intro: ICCV 2017
- keywords: Recurrent Scale Approximation (RSA)
- arxiv: https://arxiv.org/abs/1707.09531
- github: https://github.com/sciencefans/RSA-for-object-detection

DSOD: Learning Deeply Supervised Object Detectors from Scratch

- intro: ICCV 2017. Fudan University & Tsinghua University & Intel Labs China
- arxiv: https://arxiv.org/abs/1708.01241
- github: https://github.com/szq0214/DSOD

Object Detection from Scratch with Deep Supervision

https://arxiv.org/abs/1809.09294

CoupleNet: Coupling Global Structure with Local Parts for Object Detection

- intro: ICCV 2017
- arxiv: https://arxiv.org/abs/1708.02863

Incremental Learning of Object Detectors without Catastrophic Forgetting

- intro: ICCV 2017. Inria
- arxiv: https://arxiv.org/abs/1708.06977

Zoom Out-and-In Network with Map Attention Decision for Region Proposal and Object Detection

https://arxiv.org/abs/1709.04347

StairNet: Top-Down Semantic Aggregation for Accurate One Shot Detection

https://arxiv.org/abs/1709.05788

Dynamic Zoom-in Network for Fast Object Detection in Large Images

https://arxiv.org/abs/1711.05187

Zero-Annotation Object Detection with Web Knowledge Transfer

- intro: NTU, Singapore & Amazon
- keywords: multi-instance multi-label domain adaption learning framework
- arxiv: https://arxiv.org/abs/1711.05954

MegDet: A Large Mini-Batch Object Detector

- intro: Peking University & Tsinghua University & Megvii Inc
- arxiv: https://arxiv.org/abs/1711.07240

Receptive Field Block Net for Accurate and Fast Object Detection

- intro: RFBNet
- arxiv: https://arxiv.org/abs/1711.07767
- github: https://github.com//ruinmessi/RFBNet

An Analysis of Scale Invariance in Object Detection - SNIP

- intro: CVPR 2018
- arxiv: https://arxiv.org/abs/1711.08189
- github: https://github.com/bharatsingh430/snip

Feature Selective Networks for Object Detection

https://arxiv.org/abs/1711.08879

Learning a Rotation Invariant Detector with Rotatable Bounding Box

- arxiv: https://arxiv.org/abs/1711.09405
- github(official, Caffe): https://github.com/liulei01/DRBox

Scalable Object Detection for Stylized Objects

- intro: Microsoft AI & Research Munich
- arxiv: https://arxiv.org/abs/1711.09822

Learning Object Detectors from Scratch with Gated Recurrent Feature Pyramids

- arxiv: https://arxiv.org/abs/1712.00886
- github: https://github.com/szq0214/GRP-DSOD

Deep Regionlets for Object Detection

- keywords: region selection network, gating network
- arxiv: https://arxiv.org/abs/1712.02408

Training and Testing Object Detectors with Virtual Images

- intro: IEEE/CAA Journal of Automatica Sinica
- arxiv: https://arxiv.org/abs/1712.08470

Large-Scale Object Discovery and Detector Adaptation from Unlabeled Video

- keywords: object mining, object tracking, unsupervised object discovery by appearance-based clustering, self-supervised detector adaptation
- arxiv: https://arxiv.org/abs/1712.08832

Spot the Difference by Object Detection

- intro: Tsinghua University & JD Group
- arxiv: https://arxiv.org/abs/1801.01051

Localization-Aware Active Learning for Object Detection

- arxiv: https://arxiv.org/abs/1801.05124

Object Detection with Mask-based Feature Encoding

https://arxiv.org/abs/1802.03934

LSTD: A Low-Shot Transfer Detector for Object Detection

- intro: AAAI 2018
- arxiv: https://arxiv.org/abs/1803.01529

Pseudo Mask Augmented Object Detection

https://arxiv.org/abs/1803.05858

Revisiting RCNN: On Awakening the Classification Power of Faster RCNN

- intro: ECCV 2018
- keywords: DCR V1
- arxiv: https://arxiv.org/abs/1803.06799
- github(official, MXNet): https://github.com/bowenc0221/Decoupled-Classification-Refinement

Decoupled Classification Refinement: Hard False Positive Suppression for Object Detection

- keywords: DCR V2
- arxiv: https://arxiv.org/abs/1810.04002
- github(official, MXNet): https://github.com/bowenc0221/Decoupled-Classification-Refinement

Learning Region Features for Object Detection

- intro: Peking University & MSRA
- arxiv: https://arxiv.org/abs/1803.07066

Object Detection for Comics using Manga109 Annotations

- intro: University of Tokyo & National Institute of Informatics, Japan
- arxiv: https://arxiv.org/abs/1803.08670

Task-Driven Super Resolution: Object Detection in Low-resolution Images

https://arxiv.org/abs/1803.11316

Transferring Common-Sense Knowledge for Object Detection

https://arxiv.org/abs/1804.01077

Multi-scale Location-aware Kernel Representation for Object Detection

- intro: CVPR 2018
- arxiv: https://arxiv.org/abs/1804.00428
- github: https://github.com/Hwang64/MLKP

Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors

- intro: National University of Defense Technology
- arxiv: https://arxiv.org/abs/1804.04606

DetNet: A Backbone network for Object Detection

- intro: Tsinghua University & Megvii Inc
- arxiv: https://arxiv.org/abs/1804.06215

AdvDetPatch: Attacking Object Detectors with Adversarial Patches

https://arxiv.org/abs/1806.02299

Attacking Object Detectors via Imperceptible Patches on Background

https://arxiv.org/abs/1809.05966

Physical Adversarial Examples for Object Detectors

- intro: WOOT 2018
- arxiv: https://arxiv.org/abs/1807.07769

Object detection at 200 Frames Per Second

- intro: United Technologies Research Center-Ireland
- arxiv: https://arxiv.org/abs/1805.06361

Object Detection using Domain Randomization and Generative Adversarial Refinement of Synthetic Images

- intro: CVPR 2018 Deep Vision Workshop
- arxiv: https://arxiv.org/abs/1805.11778

SNIPER: Efficient Multi-Scale Training

- intro: University of Maryland
- keywords: SNIPER (Scale Normalization for Image Pyramid with Efficient Resampling)
- arxiv: https://arxiv.org/abs/1805.09300
- github: https://github.com/mahyarnajibi/SNIPER

Soft Sampling for Robust Object Detection

https://arxiv.org/abs/1806.06986

MetaAnchor: Learning to Detect Objects with Customized Anchors

- intro: Megvii Inc (Face++) & Fudan University
- arxiv: https://arxiv.org/abs/1807.00980

Localization Recall Precision (LRP): A New Performance Metric for Object Detection

- intro: ECCV 2018. Middle East Technical University
- arxiv: https://arxiv.org/abs/1807.01696
- github: https://github.com/cancam/LRP

Pooling Pyramid Network for Object Detection

- intro: Google AI Perception
- arxiv: https://arxiv.org/abs/1807.03284

Modeling Visual Context is Key to Augmenting Object Detection Datasets

- intro: ECCV 2018
- arxiv: https://arxiv.org/abs/1807.07428

Acquisition of Localization Confidence for Accurate Object Detection

- intro: ECCV 2018
- arxiv: https://arxiv.org/abs/1807.11590
- gihtub: https://github.com/vacancy/PreciseRoIPooling

CornerNet: Detecting Objects as Paired Keypoints

- intro: ECCV 2018
- keywords: IoU-Net, PreciseRoIPooling
- arxiv: https://arxiv.org/abs/1808.01244
- github: https://github.com/umich-vl/CornerNet

Unsupervised Hard Example Mining from Videos for Improved Object Detection

- intro: ECCV 2018
- arxiv: https://arxiv.org/abs/1808.04285

SAN: Learning Relationship between Convolutional Features for Multi-Scale Object Detection

https://arxiv.org/abs/1808.04974

A Survey of Modern Object Detection Literature using Deep Learning

https://arxiv.org/abs/1808.07256

Tiny-DSOD: Lightweight Object Detection for Resource-Restricted Usages

- intro: BMVC 2018
- arxiv: https://arxiv.org/abs/1807.11013
- github: https://github.com/lyxok1/Tiny-DSOD

Deep Feature Pyramid Reconfiguration for Object Detection

- intro: ECCV 2018
- arxiv: https://arxiv.org/abs/1808.07993


MDCN: Multi-Scale, Deep Inception Convolutional Neural Networks for Efficient Object Detection

- intro: ICPR 2018
- arxiv: https://arxiv.org/abs/1809.01791

Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks

https://arxiv.org/abs/1809.03193

Deep Learning for Generic Object Detection: A Survey

https://arxiv.org/abs/1809.02165

Training Confidence-Calibrated Classifier for Detecting Out-of-Distribution Samples

- intro: ICLR 2018
- arxiv: https://github.com/alinlab/Confident_classifier

Fast and accurate object detection in high resolution 4K and 8K video using GPUs

- intro: Best Paper Finalist at IEEE High Performance Extreme Computing Conference (HPEC) 2018
- intro: Carnegie Mellon University
- arxiv: https://arxiv.org/abs/1810.10551

Hybrid Knowledge Routed Modules for Large-scale Object Detection

- intro: NIPS 2018
- arxiv: https://arxiv.org/abs/1810.12681
- github(official, PyTorch): https://github.com/chanyn/HKRM

BAN: Focusing on Boundary Context for Object Detection

https://arxiv.org/abs/1811.05243

R2CNN++: Multi-Dimensional Attention Based Rotation Invariant Detector with Robust Anchor Strategy

- arxiv: https://arxiv.org/abs/1811.07126
- github: https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow

DeRPN: Taking a further step toward more general object detection

- intro: AAAI 2019
- intro: South China University of Technology
- ariv: https://arxiv.org/abs/1811.06700
- github: https://github.com/HCIILAB/DeRPN

Fast Efficient Object Detection Using Selective Attention

https://arxiv.org/abs/1811.07502

Sampling Techniques for Large-Scale Object Detection from Sparsely Annotated Objects

https://arxiv.org/abs/1811.10862

Efficient Coarse-to-Fine Non-Local Module for the Detection of Small Objects

https://arxiv.org/abs/1811.12152

Deep Regionlets: Blended Representation and Deep Learning for Generic Object Detection

https://arxiv.org/abs/1811.11318

Transferable Adversarial Attacks for Image and Video Object Detection

https://arxiv.org/abs/1811.12641

Anchor Box Optimization for Object Detection

- intro: University of Illinois at Urbana-Champaign & Microsoft Research
- arxiv: https://arxiv.org/abs/1812.00469

AutoFocus: Efficient Multi-Scale Inference

- intro: University of Maryland
- arxiv: https://arxiv.org/abs/1812.01600

Few-shot Object Detection via Feature Reweighting

https://arxiv.org/abs/1812.01866

Practical Adversarial Attack Against Object Detector

https://arxiv.org/abs/1812.10217

Scale-Aware Trident Networks for Object Detection

- intro: University of Chinese Academy of Sciences & TuSimple
- arxiv: https://arxiv.org/abs/1901.01892
- github: https://github.com/TuSimple/simpledet

Region Proposal by Guided Anchoring

- intro: CUHK - SenseTime Joint Lab & Amazon Rekognition & Nanyang Technological University
- arxiv: https://arxiv.org/abs/1901.03278

Bottom-up Object Detection by Grouping Extreme and Center Points

- keywords: ExtremeNet
- arxiv: https://arxiv.org/abs/1901.08043
- github: https://github.com/xingyizhou/ExtremeNet

Bag of Freebies for Training Object Detection Neural Networks

- intro: Amazon Web Services
- arxiv: https://arxiv.org/abs/1902.04103

Augmentation for small object detection

https://arxiv.org/abs/1902.07296

Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression

- intro: CVPR 2019
- arxiv: https://arxiv.org/abs/1902.09630

SimpleDet: A Simple and Versatile Distributed Framework for Object Detection and Instance Recognition

- intro: TuSimple
- arxiv: https://arxiv.org/abs/1903.05831
- github: https://github.com/tusimple/simpledet

BayesOD: A Bayesian Approach for Uncertainty Estimation in Deep Object Detectors

- intro: University of Toronto
- arxiv: https://arxiv.org/abs/1903.03838

DetNAS: Neural Architecture Search on Object Detection

- intro: Chinese Academy of Sciences & Megvii Inc
- arxiv: https://arxiv.org/abs/1903.10979

ThunderNet: Towards Real-time Generic Object Detection

https://arxiv.org/abs/1903.11752

Feature Intertwiner for Object Detection

- intro: ICLR 2019
- intro: CUHK & SenseTime & The University of Sydney
- arxiv: https://arxiv.org/abs/1903.11851

Improving Object Detection with Inverted Attention

https://arxiv.org/abs/1903.12255

What Object Should I Use? - Task Driven Object Detection

- intro: CVPR 2019
- arxiv: https://arxiv.org/abs/1904.03000

Towards Universal Object Detection by Domain Attention

- intro: CVPR 2019
- arxiv: https://arxiv.org/abs/1904.04402

Prime Sample Attention in Object Detection

https://arxiv.org/abs/1904.04821

BAOD: Budget-Aware Object Detection

https://arxiv.org/abs/1904.05443


An Analysis of Pre-Training on Object Detection

- intro: University of Maryland
- arxiv: https://arxiv.org/abs/1904.05871

DuBox: No-Prior Box Objection Detection via Residual Dual Scale Detectors

- intro: Baidu Inc.
- arxiv: https://arxiv.org/abs/1904.06883

NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection

- intro: CVPR 2019
- intro: Google Brain
- arxiv: https://arxiv.org/abs/1904.07392


Objects as Points

- intro: Object detection, 3D detection, and pose estimation using center point detection
- arxiv: https://arxiv.org/abs/1904.07850
- github: https://github.com/xingyizhou/CenterNet

CenterNet: Object Detection with Keypoint Triplets

CenterNet: Keypoint Triplets for Object Detection

- arxiv: https://arxiv.org/abs/1904.08189
- github: https://github.com/Duankaiwen/CenterNet

CornerNet-Lite: Efficient Keypoint Based Object Detection

- intro: Princeton University
- arxiv: https://arxiv.org/abs/1904.08900
- github: https://github.com/princeton-vl/CornerNet-Lite

Automated Focal Loss for Image based Object Detection

https://arxiv.org/abs/1904.09048

Exploring Object Relation in Mean Teacher for Cross-Domain Detection

- intro: CVPR 2019
- arxiv: https://arxiv.org/abs/1904.11245

An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection

- intro: CVPR 2019 CEFRL Workshop
- arxiv: https://arxiv.org/abs/1904.09730

RepPoints: Point Set Representation for Object Detection

- intro: ICCV 2019
- intro: Peking University & Tsinghua University & Microsoft Research Asia
- arxiv: https://arxiv.org/abs/1904.11490
- github: https://github.com/microsoft/RepPoints

Object Detection in 20 Years: A Survey

https://arxiv.org/abs/1905.05055

Light-Weight RetinaNet for Object Detection

https://arxiv.org/abs/1905.10011


Learning Data Augmentation Strategies for Object Detection

- intro: Google Research, Brain Team
- arxiv: https://arxiv.org/abs/1906.11172
- github: https://github.com/tensorflow/tpu/tree/master/models/official/detection

Towards Adversarially Robust Object Detection

- intro: ICCV 2019
- intro: Baidu Research, Sunnyvale USA
- arxiv: https://arxiv.org/abs/1907.10310

Object as Distribution

- intro: NeurIPS 2019
- intro: MIT
- arxiv: https://arxiv.org/abs/1907.12929

Detecting 11K Classes: Large Scale Object Detection without Fine-Grained Bounding Boxes

- intro: ICCV 2019
- arxiv: https://arxiv.org/abs/1908.05217

Relation Distillation Networks for Video Object Detection

- intro: ICCV 2019
- arxiv: https://arxiv.org/abs/1908.09511

FreeAnchor: Learning to Match Anchors for Visual Object Detection

- intro: NeurIPS 2019
- arxiv: https://arxiv.org/abs/1909.02466

Efficient Neural Architecture Transformation Search in Channel-Level for Object Detection

https://arxiv.org/abs/1909.02293

Self-Training and Adversarial Background Regularization for Unsupervised Domain Adaptive One-Stage Object Detection

- intro: ICCV 2019 oral
- arxiv: https://arxiv.org/abs/1909.00597

### R-FCN
* R-FCN
R-FCN是对faster rcnn的改进。因为Faster RCNN的roi pooling中的全连接层计算量大，但是丢弃全连接层（起到了融合特征和特征映射的作用），直接将roi pooling的生成的feature map 连接到最后的分类和回归层检测结果又很差，《Deep residual learning for image recognition》认为：图像分类具有图像移动不敏感性；而目标检测领域是图像移动敏感的，因此在roi pooling中加入位置相关性设计。

   ![rfcn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/rfcn.png)

   [8] Dai, Jifeng, et al. "R-FCN: Object Detection via Region-based Fully Convolutional Networks." arXiv preprint arXiv:1605.06409 (2016). [pdf](https://arxiv.org/abs/1605.06409)

[介绍](https://blog.csdn.net/App_12062011/article/details/79737363)

- arxiv: http://arxiv.org/abs/1605.06409
- github: https://github.com/daijifeng001/R-FCN
- github(MXNet): https://github.com/msracver/Deformable-ConvNets/tree/master/rfcn
- github: https://github.com/Orpine/py-R-FCN
- github: https://github.com/PureDiors/pytorch_RFCN
- github: https://github.com/bharatsingh430/py-R-FCN-multiGPU
- github: https://github.com/xdever/RFCN-tensorflow

R-FCN-3000 at 30fps: Decoupling Detection and Classification

https://arxiv.org/abs/1712.01802

### Mask R-CNN
* Mask R-CNN


ICCV 2017的最佳论文，在Mask R-CNN的工作中，它主要完成了三件事情：目标检测，目标分类，像素级分割。它在Faster R-CNN的结构基础上加上了Mask预测分支，并且改良了ROI Pooling，提出了ROI Align。这是第一次将目标检测和目标分割任务统一起来。

   ![maskrcnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/detectpic/maskrcnn.png)

   [9] He, Gkioxari, et al. "Mask R-CNN" arXiv preprint arXiv:1703.06870 (2017). [pdf] 


[介绍](https://blog.csdn.net/jiongnima/article/details/79094159)
[zhihu](https://zhuanlan.zhihu.com/p/37998710)


## Video Object Detection

Learning Object Class Detectors from Weakly Annotated Video

- intro: CVPR 2012
- paper: https://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00905.pdf

Analysing domain shift factors between videos and images for object detection

arxiv: https://arxiv.org/abs/1501.01186

Video Object Recognition

slides: http://vision.princeton.edu/courses/COS598/2015sp/slides/VideoRecog/Video%20Object%20Recognition.pptx

Deep Learning for Saliency Prediction in Natural Video

- intro: Submitted on 12 Jan 2016
- keywords: Deep learning, saliency map, optical flow, convolution network, contrast features
- paper: https://hal.archives-ouvertes.fr/hal-01251614/document

T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos

- intro: Winning solution in ILSVRC2015 Object Detection from Video(VID) Task
- arxiv: http://arxiv.org/abs/1604.02532
- github: https://github.com/myfavouritekk/T-CNN

Object Detection from Video Tubelets with Convolutional Neural Networks

- intro: CVPR 2016 Spotlight paper
- arxiv: https://arxiv.org/abs/1604.04053
- paper: http://www.ee.cuhk.edu.hk/~wlouyang/Papers/KangVideoDet_CVPR16.pdf
- gihtub: https://github.com/myfavouritekk/vdetlib

Object Detection in Videos with Tubelets and Multi-context Cues

- intro: SenseTime Group
- slides: http://www.ee.cuhk.edu.hk/~xgwang/CUvideo.pdf
- slides: http://image-net.org/challenges/talks/Object%20Detection%20in%20Videos%20with%20Tubelets%20and%20Multi-context%20Cues%20-%20Final.pdf

Context Matters: Refining Object Detection in Video with Recurrent Neural Networks

- intro: BMVC 2016
- keywords: pseudo-labeler
- arxiv: http://arxiv.org/abs/1607.04648
- paper: http://vision.cornell.edu/se3/wp-content/uploads/2016/07/video_object_detection_BMVC.pdf

CNN Based Object Detection in Large Video Images

- intro: WangTao @ 爱奇艺
- keywords: object retrieval, object detection, scene classification
- slides: http://on-demand.gputechconf.com/gtc/2016/presentation/s6362-wang-tao-cnn-based-object-detection-large-video-images.pdf

Object Detection in Videos with Tubelet Proposal Networks

arxiv: https://arxiv.org/abs/1702.06355

Flow-Guided Feature Aggregation for Video Object Detection

- intro: MSRA
- arxiv: https://arxiv.org/abs/1703.10025

Video Object Detection using Faster R-CNN

- blog: http://andrewliao11.github.io/object_detection/faster_rcnn/
- github: https://github.com/andrewliao11/py-faster-rcnn-imagenet

Improving Context Modeling for Video Object Detection and Tracking

http://image-net.org/challenges/talks_2017/ilsvrc2017_short(poster).pdf

Temporal Dynamic Graph LSTM for Action-driven Video Object Detection

- intro: ICCV 2017
- arxiv: https://arxiv.org/abs/1708.00666

Mobile Video Object Detection with Temporally-Aware Feature Maps

https://arxiv.org/abs/1711.06368

Towards High Performance Video Object Detection

https://arxiv.org/abs/1711.11577

Impression Network for Video Object Detection

https://arxiv.org/abs/1712.05896

Spatial-Temporal Memory Networks for Video Object Detection

https://arxiv.org/abs/1712.06317

3D-DETNet: a Single Stage Video-Based Vehicle Detector

https://arxiv.org/abs/1801.01769

Object Detection in Videos by Short and Long Range Object Linking

https://arxiv.org/abs/1801.09823

Object Detection in Video with Spatiotemporal Sampling Networks

- intro: University of Pennsylvania, 2Dartmouth College
- arxiv: https://arxiv.org/abs/1803.05549

Towards High Performance Video Object Detection for Mobiles

- intro: Microsoft Research Asia
- arxiv: https://arxiv.org/abs/1804.05830

Optimizing Video Object Detection via a Scale-Time Lattice

- intro: CVPR 2018
- project page: http://mmlab.ie.cuhk.edu.hk/projects/ST-Lattice/
- arxiv: https://arxiv.org/abs/1804.05472
- github: https://github.com/hellock/scale-time-lattice

Pack and Detect: Fast Object Detection in Videos Using Region-of-Interest Packing

https://arxiv.org/abs/1809.01701

Fast Object Detection in Compressed Video

https://arxiv.org/abs/1811.11057

Tube-CNN: Modeling temporal evolution of appearance for object detection in video

- intro: INRIA/ENS
- arxiv: https://arxiv.org/abs/1812.02619

AdaScale: Towards Real-time Video Object Detection Using Adaptive Scaling

- intro: SysML 2019 oral
- arxiv: https://arxiv.org/abs/1902.02910

SCNN: A General Distribution based Statistical Convolutional Neural Network with Application to Video Object Detection

- intro: AAAI 2019
- arxiv: https://arxiv.org/abs/1903.07663

Looking Fast and Slow: Memory-Guided Mobile Video Object Detection

- intro: Cornell University & Google AI
- arxiv: https://arxiv.org/abs/1903.10172

Progressive Sparse Local Attention for Video object detection

- intro: NLPR,CASIA & Horizon Robotics
- arxiv: https://arxiv.org/abs/1903.09126

Sequence Level Semantics Aggregation for Video Object Detection

https://arxiv.org/abs/1907.06390

Object Detection in Video with Spatial-temporal Context Aggregation

- intro: Huazhong University of Science and Technology & Horizon Robotics Inc.
- rxiv: https://arxiv.org/abs/1907.04988

A Delay Metric for Video Object Detection: What Average Precision Fails to Tell

- intro: ICCV 2019
- arxiv: https://arxiv.org/abs/1908.06368

Minimum Delay Object Detection From Video

- intro: ICCV 2019
- arxiv: https://arxiv.org/abs/1908.11092

## Object Detection on Mobile Devices

Pelee: A Real-Time Object Detection System on Mobile Devices

- intro: ICLR 2018 workshop track
- intro: based on the SSD
- arxiv: https://arxiv.org/abs/1804.06882
- github: https://github.com/Robert-JunWang/Pelee

## Object Detection in 3D

Vote3Deep: Fast Object Detection in 3D Point Clouds Using Efficient Convolutional Neural Networks

arxiv: https://arxiv.org/abs/1609.06666

VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

- intro: Apple Inc
- arxiv: https://arxiv.org/abs/1711.06396

Complex-YOLO: Real-time 3D Object Detection on Point Clouds

- intro: Valeo Schalter und Sensoren GmbH & Ilmenau University of Technology
- arxiv: https://arxiv.org/abs/1803.06199

Focal Loss in 3D Object Detection

- arxiv: https://arxiv.org/abs/1809.06065
- github: https://github.com/pyun-ram/FL3D

3D Object Detection Using Scale Invariant and Feature Reweighting Networks

- intro: AAAI 2019
- arxiv: https://arxiv.org/abs/1901.02237

** 3D Backbone Network for 3D Object Detection**

https://arxiv.org/abs/1901.08373

Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds

https://arxiv.org/abs/1904.07537

Monocular 3D Object Detection and Box Fitting Trained End-to-End Using Intersection-over-Union Loss

https://arxiv.org/abs/1906.08070

IoU Loss for 2D/3D Object Detection

- intro: 3d vision 2019
- arxiv: https://arxiv.org/abs/1908.03851

Fast Point R-CNN

- intro: CUHK & Tencent YouTu Lab
- arxiv: https://arxiv.org/abs/1908.02990

## Object Detection on RGB-D

Learning Rich Features from RGB-D Images for Object Detection and Segmentation

arxiv: http://arxiv.org/abs/1407.5736

Differential Geometry Boosts Convolutional Neural Networks for Object Detection

- intro: CVPR 2016
- paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w23/html/Wang_Differential_Geometry_Boosts_CVPR_2016_paper.html

A Self-supervised Learning System for Object Detection using Physics Simulation and Multi-view Pose Estimation

https://arxiv.org/abs/1703.03347

Cross-Modal Attentional Context Learning for RGB-D Object Detection

- intro: IEEE Transactions on Image Processing
- arxiv: https://arxiv.org/abs/1810.12829

## Zero-Shot Object Detection

Zero-Shot Detection

- intro: Australian National University
- keywords: YOLO
- arxiv: https://arxiv.org/abs/1803.07113

Zero-Shot Object Detection

https://arxiv.org/abs/1804.04340

Zero-Shot Object Detection: Learning to Simultaneously Recognize and Localize Novel Concepts

- intro: Australian National University
- arxiv: https://arxiv.org/abs/1803.06049

Zero-Shot Object Detection by Hybrid Region Embedding

- intro: Middle East Technical University & Hacettepe University
- arxiv: https://arxiv.org/abs/1805.06157

## Visual Relationship Detection

Visual Relationship Detection with Language Priors

- intro: ECCV 2016 oral
- paper: https://cs.stanford.edu/people/ranjaykrishna/vrd/vrd.pdf
- github: https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection

ViP-CNN: A Visual Phrase Reasoning Convolutional Neural Network for Visual Relationship Detection

- intro: Visual Phrase reasoning Convolutional Neural Network (ViP-CNN), Visual Phrase Reasoning Structure (VPRS)
- arxiv: https://arxiv.org/abs/1702.07191

Visual Translation Embedding Network for Visual Relation Detection

arxiv: https://www.arxiv.org/abs/1702.08319

Deep Variation-structured Reinforcement Learning for Visual Relationship and Attribute Detection

- intro: CVPR 2017 spotlight paper
- arxiv: https://arxiv.org/abs/1703.03054

Detecting Visual Relationships with Deep Relational Networks

- intro: CVPR 2017 oral. The Chinese University of Hong Kong
- arxiv: https://arxiv.org/abs/1704.03114

Identifying Spatial Relations in Images using Convolutional Neural Networks

https://arxiv.org/abs/1706.04215

PPR-FCN: Weakly Supervised Visual Relation Detection via Parallel Pairwise R-FCN

- intro: ICCV
- arxiv: https://arxiv.org/abs/1708.01956

Natural Language Guided Visual Relationship Detection

https://arxiv.org/abs/1711.06032

Detecting Visual Relationships Using Box Attention

- intro: Google AI & IST Austria
- arxiv: https://arxiv.org/abs/1807.02136

Google AI Open Images - Visual Relationship Track

- intro: Detect pairs of objects in particular relationships
- kaggle: https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track

Context-Dependent Diffusion Network for Visual Relationship Detection

- intro: 2018 ACM Multimedia Conference
- arxiv: https://arxiv.org/abs/1809.06213

A Problem Reduction Approach for Visual Relationships Detection

- intro: ECCV 2018 Workshop
- arxiv: https://arxiv.org/abs/1809.09828

Exploring the Semantics for Visual Relationship Detection

https://arxiv.org/abs/1904.02104


---------------------------------------------------------------------------------
## Object Segmentation 物体分割
目标识别网络（分类网络）尽管表面上来看可以接受任意尺寸的图片作为输入，但是由于网络结构最后全连接层的存在，使其丢失了输入的空间信息，因此，这些网络并没有办法直接用于解决诸如分割等稠密估计的问题。于是FCN用卷积层和池化层替代了分类网络中的全连接层，从而使得网络结构可以适应像素级的稠密估计任务。该工作被视为里程碑式的进步，因为它阐释了CNN如何可以在语义分割问题上被端对端的训练，而且高效的学习了如何基于任意大小的输入来为语义分割问题产生像素级别的标签预测。

在深度学习统治计算机视觉领域之前，有Texton Forests和Random Forest based classifiers等方法来进行语义分割。

其他人对这部分工作的收集：

https://github.com/mrgloom/awesome-semantic-segmentation

https://github.com/amusi/awesome-object-detection

https://github.com/hoya012/deep_learning_object_detection



### FCN

FCN(Fully Convolutional Networks for Semantic Segmentation)成为了深度学习技术应用于语义分割问题的基石：

它利用了现存的CNN网络作为其模块之一来产生层次化的特征。作者将现存的知名的分类模型包括AlexNet、VGG-16、GoogLeNet和ResNet等转化为全卷积模型：将其全连接层均替换为卷积层，输出空间映射而不是分类分数。这些映射由小步幅卷积上采样（又称反卷积）得到，来产生密集的像素级别的标签。

![fcn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/segpic/fcn.png)

![fcn2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/segpic/fcn2.png)

输入：整幅图像。
输出：空间尺寸与输入图像相同，通道数等于全部类别个数。
真值：通道数为1（或2）的分割图像。

 [1]  J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation.” in CVPR, 2015. pp. 3431-3440 [pdf](https://arxiv.org/pdf/1605.06211v1.pdf) CVPR 2015 Best paper

   ![fcn8s](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/segpic/fcn8s.png)


   ![fcn8sdata](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/seg/fcn8.png)

caffe https://github.com/shelhamer/fcn.berkeleyvision.org 官方

tf ： https://github.com/shekkizh/FCN.tensorflow

 尽管FCN模型强大而普适，它任然有着多个缺点从而限制其对于某些问题的应用：

 1 固有的空间不变性导致其没有考虑到有用的全局上下文信息，

 2 并没有默认考虑对实例的辨识，

 3 效率在高分辨率场景下还远达不到实时操作的能力，

 4 不完全适合非结构性数据如3D点云，或者非结构化模型。

 [参考](https://blog.csdn.net/mieleizhi0522/article/details/82902359)给出了这个综述的总结，他们所基于的架构、主要的贡献、以及基于其任务目标的分级：准确率、效率、训练难度、序列数据处理、多模式输入以及3D数据处理能力等。每个目标分为3个等级，依赖于对应工作对该目标的专注程度，叉号则代表该目标问题并没有被该工作考虑进来。

   ![fcn35](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/segpic/fcn35.png)

   ![fcn3](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/segpic/fcn3.png)

[参考](https://blog.csdn.net/mieleizhi0522/article/details/82902359)

### U-NET
http://www.arxiv.org/pdf/1505.04597.pdf
 
SegNet



Dilated Convolutions



DeepLab (v1 & v2)


http://liangchiehchen.com/projects/DeepLab.html
RefineNet

ParseNet
PSPNet


Large Kernel Matters

DeepLab v3



### SegNet

### DeconvNet

### Deeplab 

### RefineNet

### BlitzNet



### DeepMask

https://github.com/facebookresearch/deepmask


### Mask Scoring R-CNN
MS R-CNN对Mask R-CNN进行了修正,在结构中添加了Mask-IoU。Mask R-CNN的评价函数只对目标检测的候选框进行打分，而不是分割模板打分，所以会出现分割模板效果很差但是打分很高的情况。所以增加了对模板进行打分的Mask-IoU Head

   ![msrcnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/segpic/msrcnn.png)



## 参考

https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html#feature-pyramid-network-fpn