# GAN 生成式对抗网络


![GAN](https://github.com/weslynn/graphic-deep-neural-network/blob/master/map/Art&pic/ganpic.png)

-----------------------------------

“在机器学习过去的10年里，GAN是最有趣的一个想法。”
                                         ——Yann LeCun

----------------------------------
## GAN

2019年，是很重要的一年。在这一年里，GAN有了重大的进展，出现了 BigGan，StyleGan 这样生成高清大图的GAN，也出现了很多对GAN的可解释性方法，包括 苏剑林的OGAN。 还有GAN都被拿来烤Pizza了……(CVPR2019还有个PizzaGAN,[demo](http://pizzagan.csail.mit.edu/) [paper](https://arxiv.org/abs/1906.02839))

这一切预示着GAN这个话题，马上就要被勤勉的科学家们攻克了。

从目标分类的被攻克，人脸识别的特征提取和loss改进，目标检测与分割的统一…… 深度学习的堡垒一个接一个的被攻克。一切都迅速都走上可应用化的道路。


深度学习的发展惊人，如果说互联网过的是狗年，一年抵七年，深度学习的发展一定是在天宫的，天上一天，地上一年。


生成式对抗网络（GAN, Generative Adversarial Networks ）是近年来深度学习中复杂分布上无监督学习最具前景的方法之一。
监督学习需要大量标记样本，而GAN不用。
模型包括两个模块：生成模型（Generative Model）和判别模型（Discriminative Model），通过模型的互相博弈学习产生相当好的输出。原始 GAN 理论中，并不要求 G 和 D 都是神经网络，只需要是能拟合相应生成和判别的函数即可。但实用中一般均使用深度神经网络作为 G 和 D 。


![basic](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/gan/basic.png)

GAN的目标,就是G生成的数据在D看来，和真实数据误差越小越好，目标函数如下：

![basictarget](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/basictarget.png)

从判别器 D 的角度看，它希望自己能尽可能区分真实样本和虚假样本，因此希望 D(x) 尽可能大，D(G(z)) 尽可能小， 即 V(D,G)尽可能大。从生成器 G 的角度看，它希望自己尽可能骗过 D，也就是希望 D(G(z)) 尽可能大，即 V(D,G) 尽可能小。两个模型相对抗，最后达到全局最优。

从数据分布来说，就是开始的噪声noise，在G不断修正后，产生的分布，和目标数据分布达到一致：

![data](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/data.png)


   [1] Ian Goodfellow. "Generative Adversarial Networks." arXiv preprint arXiv:1406.2661v1 (2014). [pdf] (https://arxiv.org/pdf/1406.2661v1.pdf)

   http://www.iangoodfellow.com/

https://github.com/goodfeli/adversarial

![human](https://github.com/weslynn/graphic-deep-neural-network/blob/master/famous/goodfellow.jpg)

-----------------------------------------------------------------------------
“酒后脑洞”的故事…

2014 年的一个晚上，Goodfellow 在酒吧给师兄庆祝博士毕业。一群工程师聚在一起不聊姑娘，而是开始了深入了学术探讨——如何让计算机自动生成照片。



当时研究人员已经在使用神经网络（松散地模仿人脑神经元网络的算法），作为“生成”模型来创建可信的新数据。但结果往往不是很好：计算机生成的人脸图像要么模糊到看不清人脸，要么会出现没有耳朵之类的错误。

针对这个问题，Goodfellow 的朋友们“煞费苦心”，提出了一个计划——对构成照片的元素进行统计分析，来帮助机器自己生成图像。

Goodfellow 一听就觉得这个想法根本行不通，马上给否决掉了。但他已经无心再party了，刚才的那个问题一直盘旋在他的脑海，他边喝酒边思考，突然灵光一现：如果让两个神经网络互相对抗呢？

但朋友们对这个不靠谱的脑洞深表怀疑。Goodfellow 转头回家，决定用事实说话。写代码写到凌晨，然后测试…



Ian Goodfellow：如果你有良好的相关编程基础，那么快速实现自己的想法将变得非常简单。几年来，我和我的同事一直在致力于软件库的开发，我曾用这些软件库来创建第一个 GAN、Theano 和 Pylearn2。第一个 GAN 几乎是复制-粘贴我们早先的一篇论文《Maxout Networks》中的 MNIST 分类器。即使是 Maxout 论文中的超参数对 GAN 也相当有效，所以我不需要做太多的新工作。而且，MNIST 模型训练非常快。我记得第一个 MNIST GAN 只花了我一个小时左右的时间。

参考：

https://baijiahao.baidu.com/s?id=1615737087826316102&wfr=spider&for=pc


http://www.heijing.co/almosthuman2014/2018101111561225242

-----------------------------------------------------------------------------


和监督学习的的网络结构一样，GAN的发展 也主要包含网络结构性的改进 和loss、参数、权重的改进。

Avinash Hindupur建了一个GAN Zoo，他的“动物园”里目前已经收集了近500种有名有姓的GAN。
主要是2014-2018年之间的GAN。
https://github.com/hindupuravinash/the-gan-zoo

那么问题来了：这么多变体，有什么区别？哪个好用？

于是，Google Brain的几位研究员（不包括原版GAN的爸爸Ian Goodfellow）对各种进行了loss，参数，权重修改的GAN做一次“中立、多方面、大规模的”评测。
在此项研究中，Google此项研究中使用了minimax损失函数和用non-saturating损失函数的GAN，分别简称为MM GAN和NS GAN，对比了WGAN、WGAN GP、LS GAN、DRAGAN、BEGAN，另外还对比的有VAE（变分自编码器）。为了很好的说明问题，研究者们两个指标来对比了实验结果，分别是FID和精度（precision、）、召回率（recall）以及两者的平均数F1。

其中FID（Fréchet distance(弗雷歇距离) ）是法国数学家Maurice René Fréchet在1906年提出的一种路径空间相似形描述，直观来说是狗绳距离：主人走路径A，狗走路径B，各自走完这两条路径过程中所需要的最短狗绳长度，所以说，FID与生成图像的质量呈负相关。

为了更容易说明对比的结果，研究者们自制了一个类似mnist的数据集，数据集中都是灰度图，图像中的目标是不同形状的三角形。

最后，他们得出了一个有点丧的结论：

No evidence that any of the tested algorithms consistently outperforms the original one.
：

都差不多……都跟原版差不多……


Are GANs Created Equal? A Large-Scale Study
Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, Olivier Bousquet
https://arxiv.org/abs/1711.10337


http://www.dataguru.cn/article-12637-1.html

这些改进是否一无是处呢？当然不是，之前的GAN 训练很难， 而他们的优点，主要就是让训练变得更简单了。 

那对于GAN这种无监督学习的算法，不同的模型结构改进，和不同的应用领域，才是GAN大放异彩的地方。


此外，谷歌大脑发布了一篇全面梳理 GAN 的论文，该研究从损失函数、对抗架构、正则化、归一化和度量方法等几大方向整理生成对抗网络的特性与变体。
作者们复现了当前最佳的模型并公平地对比与探索 GAN 的整个研究图景，此外研究者在 TensorFlow Hub 和 GitHub 也分别提供了预训练模型与对比结果。
https://arxiv.org/pdf/1807.04720.pdf

原名：The GAN Landscape: Losses, Architectures, Regularization, and Normalization

现名：A Large-Scale Study on Regularization and Normalization in GANs

Github：http://www.github.com/google/compare_gan

TensorFlow Hub：http://www.tensorflow.org/hub

翻译 参见 http://www.sohu.com/a/241299306_129720

--------------------------------------------------------------

GAN的很多研究，都是对Generative modeling生成模型的一种研究，主要是为了对原有数据的分布进行模拟

怎么知道原有数据的分布呢？

1 Density Estimation 对原有数据进行密度估计，建模，然后使用模型进行估计
2 Sampling 取样，用对数据分布建模，并进行取样，生成符合原有数据分布的新数据。


![gang](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/gang.jpg)


----------------------------------------------
Others' collection:

https://github.com/kozistr/Awesome-GANs

https://github.com/nightrome/really-awesome-gan



------------------------------------------------
参考Mohammad KHalooei的教程，我也将GAN分为4个level，第四个level将按照应用层面进行拓展。 


# Level 0: Definition of GANs



|Level|	Title|	Co-authors|	Publication|	Links|
|:---:|:---:|:---:|:---:|:---:|
|Beginner|	GAN : Generative Adversarial Nets|	Goodfellow & et al.|	NeurIPS (NIPS) 2014	|[link](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) |
|Beginner|	GAN : Generative Adversarial Nets (Tutorial)|	Goodfellow & et al.|	NeurIPS (NIPS) 2016 Tutorial|	[link](https://arxiv.org/pdf/1701.00160.pdf)|
|Beginner|	CGAN : Conditional Generative Adversarial Nets|	Mirza & et al.|	-- 2014	|[link](https://gist.github.com/shagunsodhani/5d726334de3014defeeb701099a3b4b3) |
|Beginner|	InfoGAN : Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets|	Chen & et al.|	NeuroIPS (NIPS) 2016	||


模型结构的发展：

在标准的GAN中，生成数据的来源一般是一段连续单一的噪声z, 在半监督式学习CGAN中，会加入c的class 分类。InfoGan 找到了Gan的latent code 使得Gan的数据生成具有了可解释性。

![ganmodule](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/ganmodule.png)





## CGAN

[1411.1784]Mirza M, Osindero S,Conditional Generative Adversarial Nets [pdf](https://arxiv.org/pdf/1411.1784.pdf) 

通过GAN可以生成想要的样本，以MNIST手写数字集为例，可以任意生成0-9的数字。

但是如果我们想指定生成的样本呢？譬如指定生成1，或者2，就可以通过指定C condition来完成。

条件BN首先出现在文章 Modulating early visual processing by language 中，后来又先后被用在 cGANs With Projection Discriminator 中，目前已经成为了做条件 GAN（cGAN）的标准方案，包括 SAGAN、BigGAN 都用到了它。


https://github.com/znxlwm/tensorflow-MNIST-cGAN-cDCGAN
![cgan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/gan/cgan.png)

应用方向 数字生成， 图像自动标注等

## LAPGAN 
Emily Denton & Soumith Chintala, arxiv: 1506.05751

是第一篇将层次化或者迭代生成的思想运用到 GAN 中的工作。在原始 GAN和后来的 CGAN中，GAN 还只能生成32X32 这种低像素小尺寸的图片。而这篇工作[16] 是首次成功实现 64X64 的图像生成。思想就是，与其一下子生成这么大的（包含信息量这么多），不如一步步由小转大，这样每一步生成的时候，可以基于上一步的结果，而且还只需要“填充”和“补全”新大小所需要的那些信息。这样信息量就会少很多，而为了进一步减少信息量，他们甚至让 G 每次只生成“残差”图片，生成后的插值图片与上一步放大后的图片做加法，就得到了这一步生成的图片。


## IcGAN
Invertible Conditional GANs for image editing

通常GAN的生成网络输入为一个噪声向量z,IcGAN是对cGAN的z的解释。

利用一个encoder网络,对输入图像提取得到一个特征向量z,将特征向量z,以及需要转换的目标attribute向量y串联输入生成网络,得到生成图像,网络结构如下,

![icgan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/icgan.png)


https://arxiv.org/pdf/1611.06355.pdf
https://github.com/Guim3/IcGAN


## ACGAN

为了提供更多的辅助信息并允许半监督学习，可以向判别器添加额外的辅助分类器，以便在原始任务以及附加任务上优化模型。

和CGAN不同的是，C不直接输入D。D不仅需要判断每个样本的真假，还需要完成一个分类任务即预测C


添加辅助分类器允许我们使用预先训练的模型（例如，在ImageNet上训练的图像分类器），并且在ACGAN中的实验证明这种方法可以帮助生成更清晰的图像以及减轻模式崩溃问题。 使用辅助分类器还可以应用在文本到图像合成和图像到图像的转换。

![acgan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/gan/acgan.png)


## SemiGan /SSGAN  Goodfellow

Salimans, Tim, et al. “Improved techniques for training gans.” Advances in Neural Information Processing Systems. 2016.


![ssgan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/gan/semi.png)



Theano+Lasagne https://github.com/openai/improved-gan

tf: https://github.com/gitlimlab/SSGAN-Tensorflow

https://blog.csdn.net/shenxiaolu1984/article/details/75736407


----------------------
## InfoGan OpenAI

InfoGAN - Xi Chen, arxiv: 1606.03657

提出了latent code。

单一的噪声z，使得人们无法通过控制z的某些维度来控制生成数据的语义特征，也就是说，z是不可解释的。

以MNIST手写数字集为例，每个数字可以分解成多个维度特征：数字的类别、倾斜度、粗细度等等，在标准GAN的框架下，是无法在维度上具体指定生成什么样的数字。但是Info Gan 通过latent code的设定成功让网络学习到了可解释的特征表示（interpretable representation）

把原来的噪声z分解成两部分：一是原来的z；二是由若干个latent variables拼接而成的latent code c，这些latent variables会有一个先验的概率分布，且可以是离散的或连续的，用于代表生成数据的不同特征维度，如数字类别（离散），倾斜度（连续），粗细度（连续）等。通过找到对信息影响最大的c，来得到数据中最重要的特征。


InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets，NIPS 2016。

![info](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/gan/infogan.png)


https://arxiv.org/abs/1606.03657

https://github.com/openai/InfoGAN


----------------------


# Level 1: Improvements of GANs training

然后看看 loss、参数、权重的改进：

|Level|	Title|	Co-authors|	Publication|	Links|
|:---:|:---:|:---:|:---:|:---:|
|Beginner	|LSGAN : Least Squares Generative Adversarial Networks	|Mao & et al.|	ICCV 2017|[link](https://ieeexplore.ieee.org/document/8237566)|	
|Advanced	|Improved Techniques for Training GANs	|Salimans & et al.|	NeurIPS (NIPS) 2016	|[link](https://ceit.aut.ac.ir/http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf)|	
|Advanced	|WGAN : Wasserstein GAN	|Arjovsky & et al.|	ICML 2017|[link](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf)|
|Advanced	|WGAN-GP : improved Training of Wasserstein GANs|	 2017|[link](https://arxiv.org/pdf/1704.00028v3.pdf)|
|Advanced	|Certifying Some Distributional Robustness with Principled Adversarial Training	|Sinha & et al.|ICML 2018|[link](https://arxiv.org/pdf/1710.10571.pdf) [code](https://github.com/duchi-lab/certifiable-distributional-robustness)|

Loss Functions:

## LSGAN(Least Squares Generative Adversarial Networks)

LS-GAN - Guo-Jun Qi, arxiv: 1701.06264

   [2] Mao et al., 2017.4 [pdf](https://arxiv.org/pdf/1611.04076.pdf)

 https://github.com/hwalsuklee/tensorflow-generative-model-collections
 https://github.com/guojunq/lsgan

用了最小二乘损失函数代替了GAN的损失函数,缓解了GAN训练不稳定和生成图像质量差多样性不足的问题。

但缺点也是明显的, LSGAN对离离群点的过度惩罚, 可能导致样本生成的'多样性'降低, 生成样本很可能只是对真实样本的简单模仿和细微改动.

## WGAN
WGAN - Martin Arjovsky, arXiv:1701.07875v1

WGAN：
在初期一个优秀的GAN应用需要有良好的训练方法，否则可能由于神经网络模型的自由性而导致输出不理想。 

为啥难训练？  令人拍案叫绝的Wasserstein GAN 中做了如下解释 ：
原始GAN不稳定的原因就彻底清楚了：判别器训练得太好，生成器梯度消失，生成器loss降不下去；判别器训练得不好，生成器梯度不准，四处乱跑。只有判别器训练得不好不坏才行，但是这个火候又很难把握，甚至在同一轮训练的前后不同阶段这个火候都可能不一样，所以GAN才那么难训练。

https://zhuanlan.zhihu.com/p/25071913

WGAN 针对loss改进 只改了4点：
1.判别器最后一层去掉sigmoid
2.生成器和判别器的loss不取log
3.每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
4.不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

https://github.com/martinarjovsky/WassersteinGAN


## WGAN-GP
Regularization and Normalization of the Discriminator:

![wgangp](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/wgangp.png)

WGAN-GP：

WGAN的作者Martin Arjovsky不久后就在reddit上表示他也意识到没能完全解决GAN训练稳定性，认为关键在于原设计中Lipschitz限制的施加方式不对，并在新论文中提出了相应的改进方案--WGAN-GP ,从weight clipping到gradient penalty,提出具有梯度惩罚的WGAN（WGAN with gradient penalty）替代WGAN判别器中权重剪枝的方法(Lipschitz限制)：

[1704.00028] Gulrajani et al., 2017,improved Training of Wasserstein GANs[pdf](https://arxiv.org/pdf/1704.00028v3.pdf)

Tensorflow实现：https://github.com/igul222/improved_wgan_training

pytorch https://github.com/caogang/wgan-gp



参考 ：

https://www.leiphone.com/news/201704/pQsvH7VN8TiLMDlK.html


----------------------


# Level 2: Implementation skill



![face](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/face.jpg)



GAN的实现

|Title|	Co-authors|	Publication|Links| size |FID/IS|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Keras Implementation of GANs|	Linder-Norén|	Github	|[link](https://github.com/eriklindernoren/Keras-GAN)|||
|GAN implementation hacks|	Salimans paper & Chintala|	World research	|[link](https://github.com/soumith/ganhacks) [paper](https://ceit.aut.ac.ir/~khalooei/tutorials/gan/#gan-hack-paper-2016)|||
|DCGAN : Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks|	Radford & et al.|2015.11-ICLR 2016	|[link](https://github.com/carpedm20/DCGAN-tensorflow) [paper](https://arxiv.org/pdf/1511.06434.pdf)| 64x64 human||
|ProGAN:Progressive Growing of GANs for Improved Quality, Stability, and Variation|Tero Karras|2017.10|[paper](https://arxiv.org/pdf/1710.10196.pdf) [link](https://github.com/tkarras/progressive_growing_of_gans)|1024x1024 human|8.04|
|SAGAN：Self-Attention Generative Adversarial Networks| Han Zhang & Ian Goodfellow|2018.05|[paper](https://arxiv.org/pdf/1805.08318.pdf) [link](https://github.com/taki0112/Self-Attention-GAN-Tensorflow)|128x128 obj|18.65/52.52|
|BigGAN:Large Scale GAN Training for High Fidelity Natural Image Synthesis|Brock et al.|2018.09-ICLR 2019|[demo](https://tfhub.dev/deepmind/biggan-256) [paper](https://arxiv.org/pdf/1809.11096.pdf) [link](https://github.com/AaronLeong/BigGAN-pytorch)|512x512 obj|9.6/166.3|
|StyleGAN:A Style-Based Generator Architecture for Generative Adversarial Networks|Tero Karras|2018.12|[paper](https://arxiv.org/pdf/1812.04948.pdf) [link]( https://github.com/NVlabs/stylegan)|1024x1024 human|4.04|


指标：

1 Inception Score (IS，越大越好) IS用来衡量GAN网络的两个指标：1. 生成图片的质量 和2. 多样性

2 Fréchet Inception Distance (FID，越小越好) 在FID中我们用相同的inception network来提取中间层的特征。然后我们使用一个均值为 μμ 方差为 ΣΣ 的正态分布去模拟这些特征的分布。较低的FID意味着较高图片的质量和多样性。FID对模型坍塌更加敏感。

FID和IS都是基于特征提取，也就是依赖于某些特征的出现或者不出现。但是他们都无法描述这些特征的空间关系。

物体的数据在Imagenet数据库上比较，人脸的 progan 和stylegan 在CelebA-HQ和FFHQ上比较。上表列的为FFHQ指标。
## DCGAN

Deep Convolution Generative Adversarial Networks(深度卷积生成对抗网络)

Alec Radford & Luke Metz提出使用 CNN 结构来稳定 GAN 的训练，并使用了以下一些 trick：

Batch Normalization
使用 Transpose convlution 进行上采样
使用 Leaky ReLu 作为激活函数
上面这些 trick 对于稳定 GAN 的训练有许多帮助

这是CNN在unsupervised learning领域的一次重要尝试，这个架构能极大地稳定GAN的训练，以至于它在相当长的一段时间内都成为了GAN的标准架构，给后面的探索开启了重要的篇章。

![dcgan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/dcgang.jpg)


![dcganr](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/dcganr.jpg)


## ImprovedDCGAN
 GANs的主要问题之一是收敛性不稳定，尽管DCGAN做了结构细化，训练过程仍可能难以收敛。GANs的训练常常是同时在两个目标上使用梯度下降，然而这并不能保证到达均衡点，毕竟目标未必是凸的。也就是说GANs可能永远达不到这样的均衡点，于是就会出现收敛性不稳定。

为了解决这一问题，ImprovedDCGAN针对DCGAN训练过程提出了不同的增强方法。
1 特征匹配(feature mapping)

为了不让生成器尽可能地去蒙骗鉴别器，ImprovedDCGAN希望以特征作为匹配标准，而不是图片作为匹配标准，于是提出了一种新的生成器的目标函数

2 批次判别(minibatch discrimination)

GAN的一个常见的失败就是收敛到同一个点，只要生成一个会被discriminator误认的内容，那么梯度方向就会不断朝那个方向前进。ImprovedDCGAN使用的方法是用minibatch discriminator。也就是说每次不是判别单张图片，而是判别一批图片。
3 历史平均(historical averaging)
4 单侧标签平滑(one-sided label smoothing)

## PGGAN(ProGAN)
Progressive Growing of GANs for Improved Quality, Stability, and Variation

Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen

首次实现了 1024 人脸生成的 Progressive Growing GANs，简称 PGGAN，来自 NVIDIA。

顾名思义，PGGAN 通过一种渐进式的结构，实现了从低分辨率到高分辨率的过渡，从而能平滑地训练出高清模型出来。论文还提出了自己对正则化、归一化的一些理解和技巧，值得思考。当然，由于是渐进式的，所以相当于要串联地训练很多个模型，所以 PGGAN 很慢。


![progan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/progan.gif)
论文地址：https://arxiv.org/pdf/1710.10196.pdf
代码实现地址：https://github.com/tkarras/progressive_growing_of_gans 


CelebA HQ 数据集


"随着 ResNet 在分类问题的日益深入，自然也就会考虑到 ResNet 结构在 GAN 的应用。事实上，目前 GAN 上主流的生成器和判别器架构确实已经变成了 ResNet：PGGAN、SNGAN、SAGAN 等知名 GAN 都已经用上了 ResNet

可以看到，其实基于 ResNet 的 GAN 在整体结构上与 DCGAN 并没有太大差别，主要的特点在于：
1) 不管在判别器还是生成器，均去除了反卷积，只保留了普通卷积层；
2) 通过 AvgPooling2D 和 UpSampling2D 来实现上/下采样，而 DCGAN 中则是通过 stride > 1 的卷积/反卷积实现的；其中 UpSampling2D 相当于将图像的长/宽放大若干倍；
3) 有些作者认为 BN 不适合 GAN，有时候会直接移除掉，或者用 LayerNorm 等代替。

然而，ResNet层数更多、层之间的连接更多，相比 DCGAN，ResNet比 DCGAN 要慢得多，所需要的显存要多得多。
                                           ---苏剑林

## SAGAN Ian Goodfellow
由于卷积的局部感受野的限制，如果要生成大范围相关（Long-range dependency）的区域会出现问题，用更深的卷积网络参数量太大，于是采用将 Self Attention 引入到了生成器（以及判别器）中，使用来自所有特征位置的信息生成图像细节，同时保证判别器能鉴别距离较远的两个特征之间的一致性，获取全局信息。
IS从36.8提到了52.52，并把FID（Fréchet Inception Distance）从27.62降到了18.65。
![sagan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/sagan.jpg)

![sagan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/sagan.png)

SAGAN 使用注意力机制，高亮部位为注意力机制关注的位置。

![saganr](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/saganr.jpg)



论文地址：https://arxiv.org/pdf/1805.08318v2.pdf

https://github.com/taki0112/Self-Attention-GAN-Tensorflow

pytorch https://github.com/heykeetae/Self-Attention-GAN

## SELF-MOD

Self Modulated Generator，来自文章 On Self Modulation for Generative Adversarial Networks

SELF-MOD 考虑到 cGAN 训练的稳定性更好，但是一般情况下 GAN 并没有标签 c 可用，而以噪声 z 自身为标签好了，自己调节自己，不借助于外部标签，但能实现类似的效果。


## BigGAN

BigGAN — Brock et al. (2019) Large Scale GAN Training for High Fidelity Natural Image Synthesis”

https://arxiv.org/pdf/1809.11096.pdf

BigGAN模型是基于ImageNet生成图像质量最高的模型之一。BigGAN作为GAN发展史上的重要里程碑，将精度作出了跨越式提升。在ImageNet （128x128分辨率）训练下，将IS从52.52提升到166.3，FID从18.65降到9.6。
该模型很难在本地机器上实现，而且BigGAN有许多组件，如Self-Attention、 Spectral Normalization和带有投影鉴别器的cGAN，这些组件在各自的论文中都有更好的解释。不过，这篇论文对构成当前最先进技术水平的基础论文的思想提供了很好的概述，论文贡献包括，大batchsize，大channel数，截断技巧，训练平稳性控制等。（暴力出奇迹）

这篇文章提供了 128、256、512 的自然场景图片的生成结果。 自然场景图片的生成可是比 CelebA 的人脸生成要难上很多

![biggan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/biggan.png)

![bigganr](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/bigganr.png)

Github：https://github.com/AaronLeong/BigGAN-pytorch


参考 https://mp.weixin.qq.com/s/9GeryvW5PI93FCmTpuFEPQ
此外 https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247495491&idx=1&sn=978f0afeb0b38affe54fc9e6d6086e3c&chksm=96ea30c3a19db9d52b735bdfee3f535ce68bcc6ace230b452b2ef8d389e66d32bba38e1574e3&scene=21#wechat_redirect


 Spectral Normalization 出自 《Spectral Norm Regularization for Improving the Generalizability of Deep Learning》 和 《Spectral Normalization for Generative Adversarial Networks》，是为了解决GAN训练不稳定的问题，从“层参数”的角度用spectral normalization 的方式施加regularization，从而使判别器D具备Lipschitz连续条件。


## StyleGAN  NVIDIA

A Style-Based Generator Architecture for Generative Adversarial Networks

被很多文章称之为 GAN 2.0，借鉴了风格迁移的模型，所以叫 Style-Based Generator

新数据集 FFHQ。

ProGAN是逐级直接生成图片，特征无法控制，相互关联，我们希望有一种更好的模型，能让我们控制生成图片过程中每一级的特征，要能够特定决定生成图片某些方面的表象，并且相互间的影响尽可能小。于是，在ProGAN的基础上，StyleGAN作出了进一步的改进与提升。

StyleGAN首先重点关注了ProGAN的生成器网络，它发现，渐进层的一个潜在的好处是，如果使用得当，它们能够控制图像的不同视觉特征。层和分辨率越低，它所影响的特征就越粗糙。简要将这些特征分为三种类型：
  1、粗糙的——分辨率不超过8^2，影响姿势、一般发型、面部形状等；
  2、中等的——分辨率为16^2至32^2，影响更精细的面部特征、发型、眼睛的睁开或是闭合等；
  3、高质的——分辨率为64^2到1024^2，影响颜色（眼睛、头发和皮肤）和微观特征；


![stylegan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/stylegan.png)

![stylegan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/stylegan.gif)
![styleganr](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/styleganr.jpg)


![stylegan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/modelpic/gan/stylegan.png)


tf： https://github.com/NVlabs/stylegan

不存在系列：

人 ： https://thispersondoesnotexist.com/

动漫： https://www.thiswaifudoesnotexist.net/ 

https://www.obormot.net/demos/these-waifus-do-not-exist-alt

猫： http://thesecatsdonotexist.com/

换衣服： Generating High-Resolution Fashion Model Images Wearing Custom Outﬁts 1908.08847.pdf

other change clothers: https://arxiv.org/pdf/1710.07346.pdf


INSTAGAN: INSTANCE-AWARE IMAGE-TO-IMAGE TRANSLATION

加入实例 裤子换裙子  https://github.com/sangwoomo/instagan

论文链接：https://arxiv.org/pdf/1812.10889.pdf


Estimated StyleGAN wallclock training times for various resolutions & GPU-clusters (source: StyleGAN repo)

|GPUs |1024x2 |512x2| 256x2| [March 2019 AWS Costs19] |
|:---:|:---:|:---:|:---:|:---:|
|1	|41 days 4 hours [988 GPU-hours]	|24 days 21 hours [597 GPU-hours]|	14 days 22 hours [358 GPU-hours]|	[$320, $194, $115]|
|2	|21 days 22 hours [1,052]	|13 days 7 hours [638]	|9 days 5 hours [442]|	[NA]|
|4	|11 days 8 hours [1,088]	|7 days 0 hours [672] 	|4 days 21 hours [468]|	[NA]|
|8	|6 days 14 hours [1,264]	|4 days 10 hours [848]	|3 days 8 hours [640]|	[$2,730, $1,831, $1,382]|


## StyleGan2


## BigBiGAN DeepMind

https://arxiv.org/pdf/1907.02544.pdf

大规模对抗性表示学习
DeepMind基于最先进的BigGAN模型构建了BigBiGAN模型，通过添加编码器和修改鉴别器将其扩展到表示学习。

BigBiGAN表明，“图像生成质量的进步转化为了表示学习性能的显著提高”。

研究人员广泛评估了BigBiGAN模型的表示学习和生成性能，证明这些基于生成的模型在ImageNet上的无监督表示学习和无条件图像生成方面都达到了state of the art的水平。

[介绍](http://www.sohu.com/a/325681408_100024677)


PS:
O-GAN 可以加入其它的loss 将生成器 变为编码器。

通过简单地修改原来的GAN模型，就可以让判别器变成一个编码器，从而让GAN同时具备生成能力和编码能力，并且几乎不会增加训练成本。这个新模型被称为O-GAN（正交GAN，即Orthogonal Generative Adversarial Network），因为它是基于对判别器的正交分解操作来完成的，是对判别器自由度的最充分利用。

Arxiv链接：https://arxiv.org/abs/1903.01931

开源代码：https://github.com/bojone/o-gan

https://kexue.fm/archives/6409

TL-GAN ： 找到隐藏空间中的特征轴（如BigGAN PGGAN等，然后在特征轴上调节） [zhihu](https://zhuanlan.zhihu.com/p/48053933) [zhihu](https://zhuanlan.zhihu.com/p/47835790)

[日本的开源GAN插件，局部定制毫无压力 | Demo](https://zhuanlan.zhihu.com/p/62569491)



-------------------------------------------------------------
在现有的BigGan基础上，发展方向有了一个新的分支，就是减少标注数据，加入无监督学习。

## S³GAN（CompareGAN）


High-Fidelity Image Generation With Fewer Labels
https://arxiv.org/abs/1903.02271

Github ：https://github.com/google/compare_gan



-------------------------------------------------------



# GAN的应用 Level 3： GANs Applications 

## 3-1 GANs Applications in CV



<table>
    <tr>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/Image-translation图像翻译">图像翻译 (Image Translation)</a></td>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/Super-Resolution超分辨率">超分辨率 (Super-Resolution)</a></td>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/Colourful-Image%20Colorization图像上色%20%20">图像上色(Colourful Image Colorization)</a></td>
    </tr>   
    <tr>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/blob/master/GAN对抗生成网络/Image%20Inpainting图像修复/README.md"> 图像修复(Image Inpainting)</a></td>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/Image-denoising图像去噪">图像去噪(Image denoising)</a></td>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/交互式图像生成">交互式图像生成</a></td>   
</table>


特殊领域与应用

<table>
    <tr>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/comic-anime-manga漫画">漫画 (comic、anime、manga)</a></td>
      <td align="center"><a href="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/face-changing换脸">换脸 (face changing)</a></td>
    </tr>  
</table>

## 3-2 GANs Applications in Video



## 3-3 GANs Applications in NLP/Speech










