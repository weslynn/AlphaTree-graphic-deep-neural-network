
# CTPN
把RNN引入检测问题，使用CNN + RNN 进行文本检测与定位。采用Top-down的方式，先用CNN得到深度特征，然后用固定宽度的anchor来检测text proposal，并把同一行anchor对应的特征串成序列，输入到RNN中，最后用全连接层来分类或回归，并将正确的text proposal进行合并成文本线。

paper ：[ECCV2016] Zhi Tian, Weilin Huang, Tong He, Pan He, Yu Qiao，Detecting Text in Natural Image with Connectionist Text Proposal Network [pdf](https://arxiv.org/pdf/1609.03605.pdf) 

CTPN 原始结构如图

![CTPN](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/ocrpic/ctpn.png)

分为如下几个部分：

第一个部分为CNN：卷积网络部分是基于VGG框架设计，用VGG16的前5个Conv stage 得到feature map

第二个部分，在Conv5的feature map的每个位置上取3×3的窗口的特征，这些特征将用于预测该位置k个anchor（anchor的定义和Faster RCNN类似）对应的类别信息，位置信息，然后传递给双向LSTM

第三个部分，特征经过FC层，输出到三个输出：vertical coordinate，scores ，side-refinement。2k vertical coordinate表示的是bounding box的高度和中心的y轴坐标（可以决定上下边界），k个side-refinement表示的bounding box的水平平移量，它们是用来回归k个anchor的位置信息，每个anchor的width是16（VGG16的conv5的stride是16）。

之后用文本线构造算法，把分类得到的文字的proposal（图Fig.1（b）中的细长的矩形）合并成文本线

paper中的模型是可以应用到各种水平文本检测。

用不同节点表示如图：

![cptn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/ctpn.png)


作者给出的caffe代码模型结构有一点改变，包括输出：

![cptncaffe](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/ctpn_caffe.png)


Caffe 源码：https://github.com/tianzhi0549/CTPN 官方


# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)
