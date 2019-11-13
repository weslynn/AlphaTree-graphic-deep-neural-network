# Salient Object Detection 物体显著性检测

Salient Object Detection: A Surve (http://mmcheng.net/zh/paperreading/)

这里借用一张图，展示Salient Object Detection 算法的发展

![total](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/detectpic/salientobjectdetection.jpg)


1) 1998 Itti等人提出的最早、经典的的显著模型。掀起了跨认知心理学、神经科学和计算机视觉等多个学科的第一波热潮。
A model of saliency-based visual attention for rapid scene analysis ([Matlab](http://www.saliencytoolbox.net/), 9000+ citations)
[pdf](http://ilab.usc.edu/publications/doc/Itti_etal98pami.pdf)


2) 2007 第二波热潮由刘等人和Achanta等人掀起，他们将显著性检测定义为二元分割问题，自此出现了大量的显著性检测模型。
 Saliency detection: A spectral residual approach, ([Matlab](http://www.klab.caltech.edu/~xhou/projects/spectralResidual/spectralresidual.html), 2600+ citations)
 http://www.houxiaodi.com/assets/papers/cvpr07.pdf   这篇论文影响力很大，一个重要的原因是简单，出奇的简单！这篇论文一共5行matlab代码  

 Frequency-tuned salient region detection, (C++, 2400+ citations)
  http://ivrlwww.epfl.ch/supplementary_material/RK_CVPR09/ 

3) 第三波热潮，卷积神经网络（CNN）
  基于CNN的模型通常包含数十万个可调参数和具有可变感受野大小的神经元。神经元具有较大的接受范围提供全局信息，可以帮助更好地识别图像中最显著的区域。CNN成为显著性物体检测的主流方向。


良好的显著性检测模型应至少满足以下三个标准：
- 良好的检测：丢失实际显著区域的可能性以及将背景错误地标记为显著区域应该是低的
- 高分辨率：显著图应该具有高分辨率或全分辨率以准确定位突出物体并保留原始图像信息
- 计算效率：作为其他复杂过程的前端，这些模型应该快速检测显著区域。

https://blog.csdn.net/qq_32493539/article/details/79530118

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

这篇论文虽然只是个short paper，但是在这个领域有着不可磨灭的绝对重要性。其最大的贡献在于将Visual attention的问题用计算模型表达出来，并展示出来这个问题可以在一定程度上得到有意义的结果。其中提到的Center-Surround difference在后续的很多工作中都被以不同的形式表现出来。除了生成saliency map （后续的很多方法只生成saliency map），这篇文章也探讨了注视点的转移机制。总之，说这篇论文是saliency Detection computation的开山之作也不为过，此文对后续工作有着深刻的影响。体现了最牛的一种创新境界“提出新问题”。

2.1.1 具有内在线索的基于块的模型

有两个缺点：1）高对比度边缘通常突出而不是突出物体；2）凸显物体的边界不能很好地保存。为了克服这些问题，一些方法提出基于区域来计算显著性。两个主要优点：1）区域的数量远少于区块的数量，这意味着开发高效和快速算法的潜力；2）更多的信息功能可以从区域中提取，领先以更好的表现。

2.1.2 具有内在线索的基于区域的模型（图4）

基于区域的显著性模型的主要优势：1）采用互补先验，以提高整体性能，这是主要优势；2）与像素和色块相比，区域提供更复杂的线索（如颜色直方图），以更好地捕捉场景的显著对象；3）由于图像中的区域数量远小于像素数量，因此在生成全分辨率显著图时，区域级别的计算显著性可以显著降低计算成本。

2.1.3 具有外部线索的模型（图5）

2.1.4 其他经典模型（图6）

局部化模型、分割模型、监督模式与无监督模式、聚合和优化模型

2.2 基于深度学习的模型

2.2.1 基于CNN（经典卷积网络）的模型

CNN大大降低了计算成本，多级特征允许CNN更好地定位检测到显著区域的边界，即使存在阴影或反射。但CNN特征的空间信息因为使用了MLP（多层感知器）而无法保留。

2.2.2 基于FCN（完全卷积网络）的模型

该模型具有保存空间信息的能力，可实现点对点学习和端到端训练策略，与CNN相比大大降低了时间成本。但在具有透明物体的场景、前景和背景之间的相同对比度以及复杂的背景等情况无法检测显著物体。
————————————————
版权声明：本文为CSDN博主「qq_32493539」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_32493539/article/details/79530118




## 参考

https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html#feature-pyramid-network-fpn