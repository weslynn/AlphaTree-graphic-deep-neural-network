
## Neural Style 风格迁移

Neural Style Transfer: A Review 

https://github.com/ycjing/Neural-Style-Transfer-Papers 

包含图像风格化综述论文对应论文、源码和预训练模型。 [中文](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247489172&idx=1&sn=42f567fb57d2886da71a07dd16388022&chksm=96e9c914a19e40025bf88e89514d5c6f575ee94545bd5d854c01de2ca333d4738b433d37d1f5#rd)

![neuralstyle](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/overview.jpg)

### 风格迁移 Neural Style

它将待风格化图片和风格化样本图放入VGG中进行前向运算。其中待风格化图像提取relu4特征图，风格化样本图提取relu1,relu2,relu3,relu4,relu5的特征图。我们要把一个随机噪声初始化的图像变成目标风格化图像，将其放到VGG中计算得到特征图，然后分别计算内容损失和风格损失。

用这个训练好的 VGG 提取风格图片代表风格的高层语义信息，具体为，把风格图片作为 VGG 的输入，然后提取在风格语义选取层激活值的格拉姆矩阵（Gramian Matrix）。值得一提的是，格拉姆矩阵的数学意义使得其可以很好地捕捉激活值之间的相关性，所以能很好地表现图片的风格特征；
用 VGG 提取被风格化图片代表内容的高层语义信息，具体为，把该图片作为 VGG 的输入，然后提取内容语义提取层的激活值。这个方法很好地利用了卷积神经网络的性质，既捕捉了图片元素的结构信息，又对细节有一定的容错度；
随机初始化一张图片，然后用2，3介绍的方法提取其风格，内容特征，然后将它们分别与风格图片的风格特征，内容图片的内容特征相减，再按一定的权重相加，作为优化的目标函数。
保持 VGG 的权重不不变，直接对初始化的图⽚做梯度下降，直至目标函数降至一个比较小的值。
这个方法的风格化效果震惊了学术界，但它的缺点也是显而易见的，由于这种风格化方式本质上是一个利用梯度下降迭代优化的过程，所以尽管其效果不不错，但是风格化的速度较慢，处理一张图片在GPU上大概需要十几秒。deepart.io这个网站就是运用这个技术来进行图片纹理转换的。 




Ulyanov的Texture Networks: Feed-forward Synthesis of Textures and Stylized Images
李飞飞老师的Perceptual Losses for Real-Time Style Transfer and Super-Resolution。 

后面两篇都是将原来的求解全局最优解问题转换成用前向网络逼近最优解
原版的方法每次要将一幅内容图进行风格转换，就要进行不断的迭代，而后两篇的方法是先将其进行训练，训练得到前向生成网络，以后再来一张内容图，直接输入到生成网络中，即可得到具有预先训练的风格的内容图。 


### 多风格及任意风格转换

A Learned Representation for Artistic Style

condition instance normalization。

该论文在IN的基础上做了改进，加入了类似BN的γ和β缩放和平移因子，也就是风格特征的方差和均值，称为CIN（条件IN）。这样一来，网络只要在学习风格化的同时学习多种不同风格的γ和β，并保存起来。在要风格化某一风格的时候，只要将网络的所有γ和β（网络中所有有用到CIN层的地方）替换成对应风格的γ和β。 该方法还能实现同一内容图像风格化成多种风格的融合，这只要将多种风格特征的γ和β进行相应的线性融合便可，具体参考论文的实验部分。 该论文只能同时风格化有限的风格种类（论文中为32种），因为其需要保存所有风格种类的γ和β参数。



Diversified Texture Synthesis with Feed-forward Networks
是通过加入不同的风格图片ID，并加入嵌入层，来达到实现多种风格的目的。有点类似语音合成中的基于说话人ID搞成词向量作为网络的输入信息之一。

Fast Patch-based Style Transfer of Arbitrary Style
      这篇论文实现了图像的任意风格转换，不在局限于单个风格的训练。同时支持优化和前向网络的方法。生成时间：少于 10 秒。网络核心部分是一个style swap layer，即在这一层，对content的feature maps的每一块使用最接近的style feature 来替换。


style swap
论文分为2个部分，第一部分就是常规的迭代方式，第二个是将常规的改成一次前向的方法。


Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
https://arxiv.org/pdf/1703.06868.pdf
https://github.com/xunhuang1995/AdaIN-style
http://www.ctolib.com/AdaIN-style.html
支持使用一个前向网络来实现任意的风格转换，同时还保证的效率，能达到实时的效果。运行时间少于1s, 该论文在CIN的基础上做了一个改进，提出了AdaIN（自适应IN层）。顾名思义，就是自己根据风格图像调整缩放和平移参数，不在需要像CIN一样保存风格特征的均值和方差，而是在将风格图像经过卷积网络后计算出均值和方差。



### 语义合成图像：涂鸦拓展

### Neural Doodle 
纹理转换的另外一个非常有意思的应用是Neural Doodle，运用这个技术，我们可以让三岁的小孩子都轻易地像莫奈一样成为绘画大师。这个技术本质上其实就是先对一幅世界名画（比如皮埃尔-奥古斯特·雷诺阿的Bank of a River）做一个像素分割，得出它的语义图，让神经网络学习每个区域的风格。 
然后，我们只需要像小孩子一样在这个语义图上面涂鸦（比如，我们想要在图片的中间画一条河，在右上方画一棵树），神经网络就能根据语义图上的区域渲染它，最后得出一幅印象派的大作。

Champandard（2016） “Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks”

基于 Chuan Li 和 Michael Wand（2016）在论文“Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis”中提出的 Neural Patches 算法。这篇文章中深入解释了这个项目的动机和灵感来源：https://nucl.ai/blog/neural-doodles/

doodle.py 脚本通过使用1个，2个，3个或4个图像作为输入来生成新的图像，输入的图像数量取决于你希望生成怎样的图像：原始风格及它的注释（annotation），以及带有注释（即你的涂鸦）的目标内容图像（可选）。该算法从带风格图像中提取 annotated patches，然后根据它们匹配的紧密程度用这些 annotated patches 渐进地改变目标图像的风格。

Github 地址：https://github.com/alexjc/neural-doodle

Faster
https://github.com/DmitryUlyanov/fast-neural-doodle
实时
https://github.com/DmitryUlyanov/online-neural-doodle





###  Controlling Perceptual Factors in Neural Style Transfer
颜色控制颜色控制
在以前的风格转换中，生成图的颜色都会最终变成style图的颜色，但是很多时候我们并不希望这样。其中一种方法是，将RGB转换成YIQ，只在Y上进行风格转换，因为I和Q通道主要是保存了颜色信息。 




###  Deep Photo Style Transfer
本文在 Neural Style algorithm [5] 的基础上进行改进，主要是在目标函数进行了修改，加了一项 Photorealism regularization，修改了一项损失函数引入 semantic segmentation 信息使其在转换风格时 preserve the image structure
贡献是将从输入图像到输出图像的变换约束在色彩空间的局部仿射变换中，将这个约束表示成一个完全可微的参数项。我们发现这种方法成功地抑制了图像扭曲，在各种各样的场景中生成了满意的真实图像风格变换，包括一天中时间变换，天气，季节和艺术编辑风格变换。


以前都是整幅图stransfer的，然后他们想只对一幅图的单个物体进行stransfer，比如下面这幅图是电视剧Son of Zorn的剧照，设定是一个卡通人物生活在真实世界。他们还说这种技术可能在增强现实起作用，比如Pokemon go. 




### Visual Attribute Transfer through Deep Image Analogy SIGGRAPH 2017 paper
https://github.com/msracver/Deep-Image-Analogy



