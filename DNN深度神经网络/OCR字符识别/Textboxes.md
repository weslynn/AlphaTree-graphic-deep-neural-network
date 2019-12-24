
# TextBoxes 

TextBoxes 这个算法是基于SSD来实现的,解决水平文字检测问题，将原来3×3的kernel改成了更适应文字的long conv kernels 3×3 -> 1×5。default boxes 也做了修改。SSD原来为多类检测问题，现在转为单类检测问题

paper ：M. Liao et al. TextBoxes: A Fast Text Detector with a Single Deep Neural Network. AAAI, 2017. [pdf](https://arxiv.org/pdf/1611.06779.pdf) 


TextBoxes的原始结构如图：
![TextBoxes](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/ocrpic/textboxes.png)

TextBoxes从SSD修改而来，但是SSD中每一个支线上3×3的卷积都被换成了1×5的卷积。在每一个特征位置，有12个default box（12个default box 为1 2 3 5 7 10 不同比率的6个box ，以及增加0.5个vertical offset后的6个box），通过text-box预测72维向量，这是文本出现的得分（text presence scores）（2维）和default box的位置偏移（offsets）（4维） 72 = 12×（2+4）。7×7的图像就预测7×7×12个得分(2)和bounding box (4)。 最后所有的输出通过一个Non-maximum suppression（NMS）得到最终的输出。
 

用不同节点表示如图：

![textboxes](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/textboxes.png)
<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/ocr/textboxes.png)</p>

对应Text_box层如图（以score输出为例 输出2）：

![textboxes_cal](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/textboxes_cal.png)


作者caffe中模型结构做了一点小小的修改，如图：


![textboxes_caffe](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/textboxes_caffe.png)

<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/textboxes_caffe.png)</p>


对应Text_box层如图（score 2维， bounding box offset 4维 ）：

![textboxes_caffecal](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/textboxes_caffecal.png)


Caffe 源码：https://github.com/MhLiao/TextBoxes_plusplus 官方


# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)
