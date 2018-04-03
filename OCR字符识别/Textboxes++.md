
# TextBoxes++

TextBoxes++ 是[TextBoxes](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/TextBoxes.md)的拓展，这个算法也是基于SSD来实现的,解决对多方向文字的检测。boundingbox的输出从4维的水平的boundingbox扩展到4+8=12维的输出。long conv kernels 从 1×5 改成了 3×5。default boxes 也做了修改。

paper ：M. Liao et al. TextBoxes++: Multi-oriented text detection [pdf](https://arxiv.org/pdf/1801.02765.pdf)


TextBoxes的原始结构如图：
![TextBoxes++](https://github.com/weslynn/graphic-deep-neural-network/blob/master/ocrpic/textboxes++.png)

TextBoxes从SSD修改而来，但是SSD中每一个支线上3×3的卷积都被换成了1×5的卷积。在每一个特征位置，通过text-box预测72维向量，这是文本出现的得分（text presence scores）（2维）和default box的位置偏移（offsets）（4维） 72 = 12×（2+4）。7×7的图像就预测7×7×12个得分(2)和bounding box (4)。 最后所有的输出通过一个Non-maximum suppression（NMS）得到最终的输出。
 

用不同节点表示如图：

![textboxes](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/textboxes.png)
<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/textboxes.png)</p>

对应网络中图像尺寸变化如图（以score输出为例 输出2）：

![textboxes_cal](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/textboxes_cal.png)


作者caffe中模型结构做了一点小小的修改，如图：


![textboxes_caffe](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/textboxes_caffe.png)

<p align="right">[大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/textboxes_caffe.png)</p>


对应网络中图像尺寸变化如图（score 2维， bounding box offset 4维 ）：

![textboxes_caffecal](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/textboxes_caffecal.png)


Caffe 源码：https://github.com/MhLiao/TextBoxes 官方


# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)
