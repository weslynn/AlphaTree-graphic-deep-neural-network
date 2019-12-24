
# CRNN
CNN+RNN，End-to-End Trainable Neural Network
白翔团队将特征提取，序列建模和转录整合到统一框架，完成端对端的识别任务

paper：[2015-CoRR] An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition [pdf]
(http://arxiv.org/pdf/1507.05717v1.pdf) 

CRNN 原始结构如图

![CRNN](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/ocrpic/crnn.jpg)

细节如图：

![CRNND](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/ocrpic/crnn-detail.png)


分为三个部分：

第一个部分为CNN：

卷积网络部分是基于VGG框架设计，分为7个卷积层，4个maxpooling层，其中两个尺寸改成1×2，2个bn层

第二个部分为RNN：

通过feature sequence extraction 将一张图像的特征转换成特征序列，传给两层双向lstm。

![CRNNFS](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/ocrpic/featuresequence.png)

第三部分为transcription，通过CTC(Connectionist Temporal Classification)层得到label序列的概率

任意一个label序列的概率 = 它的不同对齐方式的概率之和

（主要是空格以及重复字母的影响，注意映射的时候是先删除重复的字母，后删除空格）



结果如图

![CRNNR](https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/ocrpic/crnn-result.png)

用不同节点表示如图：
![crnn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/ocr/crnn.png)


code： http://mclab.eic.hust.edu.cn/~xbai/CRNN/crnn_code.zip

github：https://github.com/bgshih/crnn Torch7 官方

https://github.com/meijieru/crnn.pytorch pytorch 



# [返回首页](https://github.com/weslynn/graphic-deep-neural-network/)