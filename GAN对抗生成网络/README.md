## GAN

生成式对抗网络（GAN, Generative Adversarial Networks ）是近年来深度学习中复杂分布上无监督学习最具前景的方法之一。
监督学习需要大量标记样本，而GAN不用。
模型包括两个模块：生成模型（Generative Model）和判别模型（Discriminative Model），通过模型的互相博弈学习产生相当好的输出。原始 GAN 理论中，并不要求 G 和 D 都是神经网络，只需要是能拟合相应生成和判别的函数即可。但实用中一般均使用深度神经网络作为 G 和 D 。

   [1] Ian Goodfellow. "Generative Adversarial Networks." arXiv preprint arXiv:1406.2661v1 (2014). [pdf] (https://arxiv.org/pdf/1406.2661v1.pdf)

   http://www.iangoodfellow.com/

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
和监督学习的的网络结构一样，GAN的发展 也主要包含网络结构性的改进 和loss、参数、权重的改进。我们首先看后者 。


## LSGAN(Least Squares Generative Adversarial Networks)



[1611.04076]Mao et al., 2017.4 [pdf](https://arxiv.org/pdf/1611.04076.pdf)

 https://github.com/hwalsuklee/tensorflow-generative-model-collections
 https://github.com/guojunq/lsgan

用了最小二乘损失函数代替了GAN的损失函数,缓解了GAN训练不稳定和生成图像质量差多样性不足的问题。

但缺点也是明显的, LSGAN对离离群点的过度惩罚, 可能导致样本生成的'多样性'降低, 生成样本很可能只是对真实样本的简单模仿和细微改动.

## WGAN /WGAN-GP

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

WGAN的作者Martin Arjovsky不久后就在reddit上表示他也意识到没能完全解决GAN训练稳定性，认为关键在于原设计中Lipschitz限制的施加方式不对，并在新论文中提出了相应的改进方案--WGAN-GP ,从weight clipping到gradient penalty,提出具有梯度惩罚的WGAN（WGAN with gradient penalty）替代WGAN判别器中权重剪枝的方法(Lipschitz限制)：

[1704.00028] Gulrajani et al., 2017,mproved Training of Wasserstein GANs[pdf](https://arxiv.org/pdf/1704.00028v3.pdf)

Tensorflow实现：https://github.com/igul222/improved_wgan_training

pytorch https://github.com/caogang/wgan-gp




参考 ：

https://www.leiphone.com/news/201704/pQsvH7VN8TiLMDlK.html


模型结构 ： 常见的合成为 数字及人脸。



gan

2D
------------------

stylegan
biggan
http://www.ijiandao.com/2b/baijia/183668.html
3D
-------------------
《Visual Object Networks: Image Generation with Disentangled 3D Representation》，描述了一种用GAN生成3D图片的方法。

这篇文章被近期在蒙特利尔举办的

NeurIPS 2018

## 应用 
1.0 人脸生成

1 侧脸->正脸 多角度 
TP-GAN

2 年龄

agegan？？


论文：https://arxiv.org/pdf/1711.10352.pdf


http://k.sina.com.cn/article_6462307252_1812efbb4001009quq.html
1.1 人脸表情生成

https://github.com/albertpumarola/GANimation

https://www.albertpumarola.com/research/GANimation/
http://tech.ifeng.com/a/20180729/45089543_0.shtml

1.2 动漫头像生成

2.1 图像翻译

所谓图像翻译，指从一副（源域）图像到另一副（目标域）图像的转换。可以类比机器翻译，一种语言转换为另一种语言。翻译过程中会保持源域图像内容不变，但是风格或者一些其他属性变成目标域。

Paired two domain data

成对图像翻译典型的例子就是 pix2pix

pix2pix 使用成对数据训练了一个条件 GAN，Loss 包括 GAN 的 loss 和逐像素差 loss。而 PAN 则使用特征图上的逐像素差作为感知损失替代图片上的逐像素差，以生成人眼感知上更加接近源域的图像。

Unpaired two domain data

对于无成对训练数据的图像翻译问题，一个典型的例子是 CycleGAN。


CycleGAN 使用两对 GAN，将源域数据通过一个 GAN 网络转换到目标域之后，再使用另一个 GAN 网络将目标域数据转换回源域，转换回来的数据和源域数据正好是成对的，构成监督信息。


3.1 超分辨


## SRGAN

SRGAN，2017 年 CVPR 中备受瞩目的超分辨率论文，把超分辨率的效果带到了一个新的高度，而 2017 年超分大赛 NTIRE 的冠军 EDSR 也是基于 SRGAN 的变体。



SRGAN 是基于 GAN 方法进行训练的，有一个生成器和一个判别器，判别器的主体使用 VGG19，生成器是一连串的 Residual block 连接，同时在模型后部也加入了 subpixel 模块，借鉴了 Shi et al 的 Subpixel Network [6] 的思想，重点关注中间特征层的误差，而不是输出结果的逐像素误差。让图片在最后面的网络层才增加分辨率，提升分辨率的同时减少计算资源消耗。

胡志豪提出一个来自工业界的问题
在实际生产使用中，遇到的低分辨率图片并不一定都是 PNG 格式的（无损压缩的图片复原效果最好），而且会带有不同程度的失真（有损压缩导致的 artifacts）。笔者尝试过很多算法，例如 SRGAN、EDSR、RAISR、Fast Neural Style 等等，这类图片目前使用任何一种超分算法都没法在提高分辨率的同时消除失真。


## ESRGAN 

ECCV 2018收录，赢得了PIRM2018-SR挑战赛的第一名。







3.1.4 图像联合分布学习

大部分 GAN 都是学习单一域的数据分布，CoupledGAN 则提出一种部分权重共享的网络，使用无监督方法来学习多个域图像的联合分布。具体结构如下 [11]：



如上图所示，CoupledGAN 使用两个 GAN 网络。生成器前半部分权重共享，目的在于编码两个域高层的，共有信息，后半部分没有进行共享，则是为了各自编码各自域的数据。判别器前半部分不共享，后半部分用于提取高层特征共享二者权重。对于训练好的网络，输入一个随机噪声，输出两张不同域的图片。

值得注意的是，上述模型学习的是联合分布 P(x,y)，如果使用两个单独的 GAN 分别取训练，那么学习到的就是边际分布 P(x) 和 P(y)。


4 根据图像生成描述 /根据描述生成图像



5 视频生成

通常来说，视频有相对静止的背景和运动的前景组成。VideoGAN 使用一个两阶段的生成器，3D CNN 生成器生成运动前景，2D CNN 生成器生成静止的背景。Pose GAN 则使用 VAE 和 GAN 生成视频，首先，VAE 结合当前帧的姿态和过去的姿态特征预测下一帧的运动信息，然后 3D CNN 使用运动信息生成后续视频帧。Motion and Content GAN(MoCoGAN) 则提出在隐空间对运动部分和内容部分进行分离，使用 RNN 去建模运动部分。

6 序列生成

相比于 GAN 在图像领域的应用，GAN 在文本，语音领域的应用要少很多。主要原因有两个：

GAN 在优化的时候使用 BP 算法，对于文本，语音这种离散数据，GAN 没法直接跳到目标值，只能根据梯度一步步靠近。

对于序列生成问题，每生成一个单词，我们就需要判断这个序列是否合理，可是 GAN 里面的判别器是没法做到的。除非我们针对每一个 step 都设置一个判别器，这显然不合理。

为了解决上述问题，强化学习中的策略梯度下降（Policy gredient descent）被引入到 GAN 中的序列生成问题。

6.1 音乐生成

RNN-GAN 使用 LSTM 作为生成器和判别器，直接生成整个音频序列。然而，正如上面提到的，音乐当做包括歌词和音符，对于这种离散数据生成问题直接使用 GAN 存在很多问题，特别是生成的数据缺乏局部一致性。

相比之下，SeqGAN 把生成器的输出作为一个智能体 (agent) 的策略，而判别器的输出作为奖励 (reward)，使用策略梯度下降来训练模型。ORGAN 则在 SeqGAN 的基础上，针对具体的目标设定了一个特定目标函数。


wavenet

GANSynth是一种快速生成高保真音频的新方法
http://www.elecfans.com/d/877752.html


6.2 语言和语音

VAW-GAN(Variational autoencoding Wasserstein GAN) 结合 VAE 和 WGAN 实现了一个语音转换系统。编码器编码语音信号的内容，解码器则用于重建音色。由于 VAE 容易导致生成结果过于平滑，所以此处使用 WGAN 来生成更加清晰的语音信号。

