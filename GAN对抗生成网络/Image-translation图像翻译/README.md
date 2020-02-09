# GANs Applications in CV

## 3.1 图像翻译 (Image Translation)

图像翻译，指从一副（源域）输入的图像到另一副（目标域）对应的输出图像的转换。它代表了图像处理的很多问题，比如灰度图、梯度图、彩色图之间的转换等。可以类比机器翻译，一种语言转换为另一种语言。翻译过程中会保持源域图像内容不变，但是风格或者一些其他属性变成目标域。

有标注数据的，被称为Paired Image-to-Image Translation，没有标注数据的，被称为 Unpaired Image-to-Image Translation。
一张图可以同时进行多领域转换的，称为Multiple Domain
- [ Paired two domain data](#1-Paired-Image-to-Image-Translation)
- [ Unpaired two domain data](#2-Unpaired-Image-to-Image-Translation)

![compare](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/compare.png)


|Title|	Co-authors|	Publication|Links|
|:---:|:---:|:---:|:---:|
|Pix2Pix |	Zhu & Park & et al.|CVPR 2017|[demo](https://affinelayer.com/pixsrv/) [code](https://phillipi.github.io/pix2pix/) [paper](https://arxiv.org/pdf/1611.07004v1.pdf)|
|Pix2Pix HD|NVIDIA UC Berkeley | CVPR 2018|[paper](https://arxiv.org/pdf/1711.11585v2.pdf) [code](https://github.com/NVIDIA/pix2pixHD)|
|SPADE|Nvidia|2019|[paper](https://arxiv.org/abs/1903.07291) [code](https://github.com/NVlabs/SPADE)|


|Title|	Co-authors|	Publication|Links|
|:---:|:---:|:---:|:---:|
|CoupledGan||2016|[paper](https://arxiv.org/abs/1606.07536) [code](https://github.com/mingyuliutw/CoGAN)|
|DTN||2017|[paper](https://arxiv.org/abs/1611.02200v1) [code](https://github.com/yunjey/domain-transfer-network)|
|CycleGan| |2017|[code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [paper](https://arxiv.org/pdf/1703.10593.pdf)|
|DiscoGan ||2017|[paper](https://arxiv.org/abs/1703.05192)|
|DualGan||2017|[paper](https://arxiv.org/abs/1704.02510)|
|UNIT||2017||
|XGAN||2018||
|OST||2018||
|FUNIT||ICCV 2019|[paper](https://arxiv.org/pdf/1905.01723.pdf) [code](https://github.com/NVlabs/FUNIT) [demo](http://nvidia-research-mingyuliu.com/petswap)|

Multiple Domain

|Title|	Co-authors|	Publication|Links|
|:---:|:---:|:---:|:---:|
|Domain-Bank||2017||
|ComboGAN||2017||
|StarGan|| 2018||


## 1. Paired two domain data

成对图像翻译典型的例子就是 pix2pix，pix2pix 使用成对数据训练了一个条件 GAN，Loss 包括 GAN 的 loss 和逐像素差 loss。而 PAN 则使用特征图上的逐像素差作为感知损失替代图片上的逐像素差，以生成人眼感知上更加接近源域的图像。

## Pix2Pix

Image-to-Image Translation with Conditional Adversarial Networks

https://arxiv.org/pdf/1611.07004v1.pdf

传统的GAN也不是万能的，它有下面两个不足：

1. 没有用户控制（user control）能力
在传统的GAN里，输入一个随机噪声，就会输出一幅随机图像。但用户是有想法滴，我们想输出的图像是我们想要的那种图像，和我们的输入是对应的、有关联的。比如输入一只猫的草图，输出同一形态的猫的真实图片（这里对形态的要求就是一种用户控制）。

2. 低分辨率（Low resolution）和低质量（Low quality）问题
尽管生成的图片看起来很不错，但如果你放大看，就会发现细节相当模糊。
 ----------------朱俊彦（Jun-Yan Zhu） Games2018 Webinar 64期 ：[Siggraph 2018优秀博士论文报告](https://games-cn.org/games-webinar-20180913-64/)

Pix2Pix对传统的CGAN做了个小改动，它不再输入随机噪声，而是输入用户给的图片：

![pix2pix](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/pix2pix.png)


通过pix2pix来完成成对的图像转换(Labels to Street Scene, Aerial to Map,Day to Night等)，可以得到比较清晰的结果。

![pix2pixr](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/pix2pixr.png)

代码：

官方project：https://phillipi.github.io/pix2pix/

官方torch代码：https://github.com/phillipi/pix2pix

官方pytorch代码（CycleGAN、pix2pix）：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

第三方的tensorflow版本：https://github.com/yenchenlin/pix2pix-tensorflow

https://github.com/affinelayer/pix2pix-tensorflow


demo: https://affinelayer.com/pixsrv/

Edges2cats： http://affinelayer.com/pixsrv/index.html

http://paintschainer.preferred.tech/index_zh.html

# Pix2Pix HD

High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs

https://arxiv.org/pdf/1711.11585v2.pdf

Ming-Yu Liu在介入过许多CV圈内耳熟能详的项目,包括vid2vid、pix2pixHD、CoupledGAN、FastPhotoStyle、MoCoGAN


1 模型结构
2 Loss设计
3 使用Instance-map的图像进行训练。

![pix2pixhd](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/pix2pixhd.png)

![pix2pixhd](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/pix2pixhd.gif)

官方代码 ：https://github.com/NVIDIA/pix2pixHD

pix2pix的核心是有了对应关系，这种网络的应用范围还是比较广泛的，如草稿变图片，自动上色，交互式上色等。

## 2. Unpaired two domain data

对于无成对训练数据的图像翻译问题，一个典型的例子是 CycleGAN。CycleGAN 使用两对 GAN，将源域数据通过一个 GAN 网络转换到目标域之后，再使用另一个 GAN 网络将目标域数据转换回源域，转换回来的数据和源域数据正好是成对的，构成监督信息。

## CoGAN (CoupledGAN)

CoGAN:Coupled Generative Adversarial Networks

CoGAN会训练两个GAN而不是一个单一的GAN。

当然，GAN研究人员不停止地将此与那些警察和伪造者的博弈理论进行类比。所以这就是CoGAN背后的想法，用作者自己的话说就是：

在游戏中，有两个团队，每个团队有两个成员。生成模型组成一个团队，在两个不同的域中合作共同合成一对图像，用以混淆判别模型。判别模型试图将从各个域中的训练数据分布中绘制的图像与从各个生成模型中绘制的图像区分开。同一团队中，参与者之间的协作是根据权重分配约束建立的。这样就有了一个GAN的多人局域网竞赛

CoupledGAN 通过部分权重共享学习到多个域图像的联合分布。生成器前半部分权重共享，目的在于编码两个域高层的，共有信息，后半部分没有进行共享，则是为了各自编码各自域的数据。判别器前半部分不共享，后半部分用于提取高层特征共享二者权重。对于训练好的网络，输入一个随机噪声，输出两张不同域的图片。

值得注意的是，上述模型学习的是联合分布 P(x,y)，如果使用两个单独的 GAN 分别取训练，那么学习到的就是边际分布 P(x) 和 P(y)。。


论文：

https://arxiv.org/abs/1606.07536

代码：

https://github.com/mingyuliutw/CoGAN

博客：

https://wiseodd.github.io/techblog/2017/02/18/coupled_gan/




## CycleGan /DiscoGan /DualGan

CycleGan: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

朱俊彦

https://arxiv.org/abs/1703.10593

同一时期还有两篇非常类似的文章，同样的idea，同样的结构，同样的功能：DualGAN( https://arxiv.org/abs/1704.02510 ) 和 DiscoGAN( Learning to Discover Cross-Domain Relations with Generative Adversarial Networks ： https://arxiv.org/abs/1703.05192)

CycleGan是让两个domain的图片互相转化。传统的GAN是单向生成，而CycleGAN是互相生成，一个A→B单向GAN加上一个B→A单向GAN，网络是个环形，所以命名为Cycle。理念就是，如果从A生成的B是对的，那么从B再生成A也应该是对的。CycleGAN输入的两张图片可以是任意的两张图片，也就是unpaired。

![CycleGan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/cyclegan.png)

![CycleGanr](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/cyclegan.jpg)

官方pytorch代码（CycleGAN、pix2pix）：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

有趣的应用：
地图转换
风格转换
游戏画风转换：Chintan Trivedi的实现：用CycleGAN把《堡垒之夜》转成《绝地求生》写实风。


## FUNIT

Few-Shot Unsupervised Image-to-Image Translation

https://arxiv.org/pdf/1905.01723.pdf

小样本(few-shot)非监督图像到图像转换。

https://github.com/NVlabs/FUNIT

主要解决两个问题，小样本Few-shot和没见过的领域转换Unseen Domains。
人能针对一个新物种，看少量样本，也能进行想象和推算 。关键就是 一个大类型的物种中，信息可以相互转换。

![FUNIT](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/FUNIT.png)

![FUNITr](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/FUNITr.png)

demo http://nvidia-research-mingyuliu.com/petswap


## StarGan

StarGAN的引入是为了解决多领域间的转换问题的，之前的CycleGAN等只能解决两个领域之间的转换，那么对于含有C个领域转换而言，需要学习Cx(C-1)个模型，但StarGAN仅需要学习一个

![starGan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/stargan.png)



https://arxiv.org/pdf/1711.09020.pdf

pytorch 原版github地址：https://github.com/yunjey/StarGAN 
tf版github地址：https://github.com/taki0112/StarGAN-Tensorflow 

