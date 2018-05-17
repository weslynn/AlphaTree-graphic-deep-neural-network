
# TextBoxes++

TextBoxes++ 是[TextBoxes](https://github.com/weslynn/graphic-deep-neural-network/blob/master/OCR%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB/Textboxes.md)的拓展，这个算法也是基于SSD来实现的,解决对多方向文字的检测。boundingbox的输出从4维的水平的boundingbox扩展到4+8=12维的输出。long conv kernels 从 1×5 改成了 3×5。default boxes 也做了修改。

paper ：M. Liao et al. TextBoxes++: Multi-oriented text detection [pdf](https://arxiv.org/pdf/1801.02765.pdf)


TextBoxes的原始结构如图：
![TextBoxes++](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/ocrpic/textboxes++.png)

TextBoxes++将TextBoxes的水平文字检测拓展到多方向，修改了输出的维度，offset从4维，拓展到4+8=12维（也可以是4+5，不过caffe源码中为4+8，5是rotated rectangele bounding box offsets，8是quadrilateral bounding box offsets），caffe源码中输出的12个default boxes拓展到20个，因此输出也从TextBoxes的72维向量拓展到20×（2+12）=280。最后所有的输出通过一个级联的Non-maximum suppression（先选0.5的高阈值，然后选0.2的低阈值）得到最终的输出。
 

用不同节点表示如图（按照作者caffe代码绘制）：

![textboxes++](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/textboxes++.png)
<p align="right"> [大图](https://raw.githubusercontent.com/weslynn/graphic-deep-neural-network/master/modelpic/ocr/textboxes++.png) </p>

对应Text_box层如图（score 2维， bounding box offset 12维）：

![textboxes_cal](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/textboxes++_cal.png)



Caffe 源码：https://github.com/MhLiao/TextBoxes_plusplus  官方


# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)
