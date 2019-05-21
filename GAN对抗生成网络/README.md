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


那么问题来了：这么多变体，有什么区别？哪个好用？

于是，Google Brain的几位研究员（不包括原版GAN的爸爸Ian Goodfellow）对各种进行了loss，参数，权重修改的GAN做一次“中立、多方面、大规模的”评测，得出了一个有点丧的结论：

No evidence that any of the tested algorithms consistently outperforms the original one.
：

都差不多……都跟原版差不多……

http://www.dataguru.cn/article-12637-1.html

这些改进是否一无是处呢？当然不是，之前的GAN 训练很难， 而他们的优点，主要就是让训练变得更简单了。 

那对于GAN这种无监督学习的算法，不同的模型结构改进，和不同的应用领域，才是GAN大放异彩的地方。


## LSGAN(Least Squares Generative Adversarial Networks)


   [2] Mao et al., 2017.4 [pdf](https://arxiv.org/pdf/1611.04076.pdf)

 https://github.com/hwalsuklee/tensorflow-generative-model-collections
 https://github.com/guojunq/lsgan

用了最小二乘损失函数代替了GAN的损失函数,缓解了GAN训练不稳定和生成图像质量差多样性不足的问题。

但缺点也是明显的, LSGAN对离离群点的过度惩罚, 可能导致样本生成的'多样性'降低, 生成样本很可能只是对真实样本的简单模仿和细微改动.

## WGAN /WGAN-GP

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

WGAN-GP：

WGAN的作者Martin Arjovsky不久后就在reddit上表示他也意识到没能完全解决GAN训练稳定性，认为关键在于原设计中Lipschitz限制的施加方式不对，并在新论文中提出了相应的改进方案--WGAN-GP ,从weight clipping到gradient penalty,提出具有梯度惩罚的WGAN（WGAN with gradient penalty）替代WGAN判别器中权重剪枝的方法(Lipschitz限制)：

[1704.00028] Gulrajani et al., 2017,mproved Training of Wasserstein GANs[pdf](https://arxiv.org/pdf/1704.00028v3.pdf)

Tensorflow实现：https://github.com/igul222/improved_wgan_training

pytorch https://github.com/caogang/wgan-gp




参考 ：

https://www.leiphone.com/news/201704/pQsvH7VN8TiLMDlK.html


----------------------
模型结构 ： 常见的合成为 数字及人脸。



gan

Deep Learning: State of the Art*
(Breakthrough Developments in 2017 & 2018)

• BERT and Natural Language Processing
• Tesla Autopilot Hardware v2+: Neural Networks at Scale
• AdaNet: AutoML with Ensembles
• AutoAugment: Deep RL Data Augmentation
• Training Deep Networks with Synthetic Data
• Segmentation Annotation with Polygon-RNN++
• DAWNBench: Training Fast and Cheap
• BigGAN: State of the Art in Image Synthesis
• Video-to-Video Synthesis
• Semantic Segmentation
• AlphaZero & OpenAI Five
• Deep Learning Frameworks
BERT和自然语言处理（NLP）

特斯拉Autopilot二代（以上）硬件：规模化神经网络

AdaNet：可集成学习的AutoML

AutoAugment：用强化学习做数据增强

用合成数据训练深度神经网络

用Polygon-RNN++做图像分割自动标注

DAWNBench：寻找快速便宜的训练方法

BigGAN：最领先的图像合成研究

视频到视频合成

语义分割

AlphaZero和OpenAI Five

深度学习框架
https://github.com/lexfridman/mit-deep-learning

2D
------------------
《Progressive Growing of GANs for Improved Quality, Stability, and Variation》
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


 Face Aging With Conditional Generative Adversarial Networks 的作者使用在 IMDB 数据集上预训练模型而获得年龄的预测方法，然后研究者基于条件 GAN 修改生成图像的面部年龄。



agegan？？


论文：https://arxiv.org/pdf/1711.10352.pdf


http://k.sina.com.cn/article_6462307252_1812efbb4001009quq.html
1.1 人脸表情生成

https://github.com/albertpumarola/GANimation

https://www.albertpumarola.com/research/GANimation/
http://tech.ifeng.com/a/20180729/45089543_0.shtml

1.2 动漫头像生成


domain-transfer-net



twin—gan

　　5、paGAN：用单幅照片实时生成超逼真动画人物头像

　　最新引起很大反响的“换脸”技术来自华裔教授黎颢的团队，他们开发了一种新的机器学习技术paGAN，能够以每秒1000帧的速度对对人脸进行跟踪，用单幅照片实时生成超逼真动画人像，论文已被SIGGRAPH 2018接收。具体技术细节请看新智元昨天的头条报道。

　　Pinscreen拍摄了《洛杉矶时报》记者David Pierson的一张照片作为输入（左），并制作了他的3D头像（右）。 这个生成的3D人脸通过黎颢的动作（中）生成表情。这个视频是6个月前制作的，Pinscreen团队称其内部早就超越了上述结果。

　https://tech.sina.com.cn/csj/2018-08-08/doc-ihhkuskt7977099.shtml


1.3 换脸
　从Deepfake到HeadOn：换脸技术发展简史

　　DAPAR的担忧并非空穴来风，如今的变脸技术已经达到威胁安全的地步。最先，可能是把特朗普和普京弄来表达政治观点；但后来，出现了比如DeepFake，令普通人也可以利用这样的技术制造虚假色情视频和假新闻。技术越来越先进，让AI安全也产生隐患。

　　1、Deepfake
https://github.com/deepfakes/faceswap
　　我们先看看最大名鼎鼎的Deepfake是何方神圣。

　　Deepfake即“deep learning”和“fake”的组合词，是一种基于深度学习的人物图像合成技术。它可以将任意的现有图像和视频组合并叠加到源图像和视频上。

　　Deepfake允许人们用简单的视频和开源代码制作虚假的色情视频、假新闻、恶意内容等。后来，deepfakes还推出一款名为Fake APP的桌面应用程序，允许用户轻松创建和分享换脸的视频，进一步把技术门槛降低到C端。

特朗普的脸被换到希拉里身上特朗普的脸被换到希拉里身上
　　由于其恶意使用引起大量批评，Deepfake已经被Reddit、Twitter等网站封杀。

　　2、Face2Face

　　Face2Face同样是一项引起巨大争议的“换脸”技术。它比Deepfake更早出现，由德国纽伦堡大学科学家Justus Thies的团队在CVPR 2016发布。这项技术可以非常逼真的将一个人的面部表情、说话时面部肌肉的变化、嘴型等完美地实时复制到另一个人脸上。它的效果如下：


　　Face2Face被认为是第一个能实时进行面部转换的模型，而且其准确率和真实度比以前的模型高得多。

　　3、HeadOn

　　HeadOn可以说是Face2Face的升级版，由原来Face2Face的团队创造。研究团队在Face2Face上所做的工作为HeadOn的大部分能力提供了框架，但Face2Face只能实现面部表情的转换，HeadOn增加了身体运动和头部运动的迁移。

　　也就是说，HeadOn不仅可以“变脸”，它还可以“变人”——根据输入人物的动作，实时地改变视频中人物的面部表情、眼球运动和身体动作，使得图像中的人看起来像是真的在说话和移动一样。

HeadOn技术的图示HeadOn技术的图示
　　研究人员在论文里将这个系统称为“首个人体肖像视频的实时的源到目标（source-to-target）重演方法，实现了躯干运动、头部运动、面部表情和视线注视的迁移”。

　　4、Deep Video Portraits

　　Deep Video Portraits 是斯坦福大学、慕尼黑技术大学等的研究人员提交给今年 8 月SIGGRAPH 大会的一篇论文，描述了一种经过改进的 “换脸” 技术，可以在视频中用一个人的脸再现另一人脸部的动作、面部表情和说话口型。


　　例如，将普通人的脸换成奥巴马的脸。Deep Video Portraits 可以通过一段目标人物的视频（在这里就是奥巴马），来学习构成脸部、眉毛、嘴角和背景等的要素以及它们的运动形式。 



图像修复  



EdgeConnect
TL-GAN

Glow


基于流的生成模型在 2014 年已经被提出，但是一直被忽视。由 OpenAI 带来的 Glow 展示了流生成模型强大的图像生成能力。文章使用可逆 1 x 1 卷积在已有的流模型 NICE 和 RealNVP 基础上进行扩展，精确的潜变量推断在人脸属性上展示了惊艳的实验效果。



■ 论文 | Glow: Generative Flow with Invertible 1x1 Convolutions

■ 链接 | https://www.paperweekly.site/papers/2101

■ 源码 | https://github.com/openai/glow

图像生成在 GAN 和 VAE 诞生后得到了很快的发展，现在围绕 GAN 的论文十分火热。生成模型只能受限于 GAN 和 VAE 吗？OpenAI 给出了否定的答案，OpenAI 带来了 Glow，一种基于流的生成模型。

Recycle-GAN




工具 ：

https://github.com/iperov/DeepFaceLab
2.1 图像翻译

所谓图像翻译，指从一副（源域）图像到另一副（目标域）图像的转换。可以类比机器翻译，一种语言转换为另一种语言。翻译过程中会保持源域图像内容不变，但是风格或者一些其他属性变成目标域。

Paired two domain data

成对图像翻译典型的例子就是 pix2pix

pix2pix 使用成对数据训练了一个条件 GAN，Loss 包括 GAN 的 loss 和逐像素差 loss。而 PAN 则使用特征图上的逐像素差作为感知损失替代图片上的逐像素差，以生成人眼感知上更加接近源域的图像。

 Ming-Yu Liu在介入过许多CV圈内耳熟能详的项目,包孕vid2vid、pix2pixHD、CoupledGAN、FastPhotoStyle、MoCoGAN

Unpaired two domain data

对于无成对训练数据的图像翻译问题，一个典型的例子是 CycleGAN。


CycleGAN 使用两对 GAN，将源域数据通过一个 GAN 网络转换到目标域之后，再使用另一个 GAN 网络将目标域数据转换回源域，转换回来的数据和源域数据正好是成对的，构成监督信息。


3.1 超分辨
超分辨率技术（Super-Resolution）是指从观测到的低分辨率图像重建出相应的高分辨率图像，在监控设备、卫星图像和医学影像等领域都有重要的应用价值。SR可分为两类:从多张低分辨率图像重建出高分辨率图像和从单张低分辨率图像重建出高分辨率图像。基于深度学习的SR，主要是基于单张低分辨率的重建方法，即Single Image Super-Resolution (SISR)。

SISR是一个逆问题，对于一个低分辨率图像，可能存在许多不同的高分辨率图像与之对应，因此通常在求解高分辨率图像时会加一个先验信息进行规范化约束。在传统的方法中，这个先验信息可以通过若干成对出现的低-高分辨率图像的实例中学到。而基于深度学习的SR通过神经网络直接学习分辨率图像到高分辨率图像的端到端的映射函数。

较新的基于深度学习的SR方法，包括SRCNN，DRCN, ESPCN，VESPCN和SRGAN等。


## SRGAN

SRGAN，2017 年 CVPR 中备受瞩目的超分辨率论文，把超分辨率的效果带到了一个新的高度，而 2017 年超分大赛 NTIRE 的冠军 EDSR 也是基于 SRGAN 的变体。



SRGAN 是基于 GAN 方法进行训练的，有一个生成器和一个判别器，判别器的主体使用 VGG19，生成器是一连串的 Residual block 连接，同时在模型后部也加入了 subpixel 模块，借鉴了 Shi et al 的 Subpixel Network [6] 的思想，重点关注中间特征层的误差，而不是输出结果的逐像素误差。让图片在最后面的网络层才增加分辨率，提升分辨率的同时减少计算资源消耗。

胡志豪提出一个来自工业界的问题
在实际生产使用中，遇到的低分辨率图片并不一定都是 PNG 格式的（无损压缩的图片复原效果最好），而且会带有不同程度的失真（有损压缩导致的 artifacts）。笔者尝试过很多算法，例如 SRGAN、EDSR、RAISR、Fast Neural Style 等等，这类图片目前使用任何一种超分算法都没法在提高分辨率的同时消除失真。


## ESRGAN 

ECCV 2018收录，赢得了PIRM2018-SR挑战赛的第一名。




其他 
waifu2x, 属于是Image scaling领域的内容.

waifu2x
Github: nagadomi/waifu2x · GitHub

Single-Image Super-Resolution for anime/fan-art using Deep Convolutional Neural Networks.
使用卷积神经网络(Convolutional Neural Network, CNN)针对漫画风格的图片进行放大. 
效果还是相当不错的, 下面是官方的Demo图:
https://raw.githubusercontent.com/nagadomi/waifu2x/master/images/slide.png



3.1.4 图像联合分布学习

大部分 GAN 都是学习单一域的数据分布，CoupledGAN 则提出一种部分权重共享的网络，使用无监督方法来学习多个域图像的联合分布。具体结构如下 [11]：



如上图所示，CoupledGAN 使用两个 GAN 网络。生成器前半部分权重共享，目的在于编码两个域高层的，共有信息，后半部分没有进行共享，则是为了各自编码各自域的数据。判别器前半部分不共享，后半部分用于提取高层特征共享二者权重。对于训练好的网络，输入一个随机噪声，输出两张不同域的图片。

值得注意的是，上述模型学习的是联合分布 P(x,y)，如果使用两个单独的 GAN 分别取训练，那么学习到的就是边际分布 P(x) 和 P(y)。



3.2 填补图像

DeepCreamPy自动去码



4 根据图像生成描述 /根据描述生成图像



5 视频生成

通常来说，视频有相对静止的背景和运动的前景组成。VideoGAN 使用一个两阶段的生成器，3D CNN 生成器生成运动前景，2D CNN 生成器生成静止的背景。Pose GAN 则使用 VAE 和 GAN 生成视频，首先，VAE 结合当前帧的姿态和过去的姿态特征预测下一帧的运动信息，然后 3D CNN 使用运动信息生成后续视频帧。Motion and Content GAN(MoCoGAN) 则提出在隐空间对运动部分和内容部分进行分离，使用 RNN 去建模运动部分。



标题：视频到视频的合成Video-to-Video Synthesis
作者：Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, Bryan Catanzaro
https://arxiv.org/abs/1808.06601
论文摘要
本文研究的问题是视频到视频(Video-to-Video)的合成，其目标是学习一个映射函数从一个输入源视频(例如，语义分割掩码序列)到一个输出逼真的视频，准确地描述了源视频的内容。
与之对应的图像到图像的合成问题是一个热门话题，而视频到视频的合成问题在文献中研究较少。在不了解时间动态的情况下，直接将现有的图像合成方法应用于输入视频往往会导致视频在时间上不连贯，视觉质量低下。
本文提出了一种在生成对抗学习框架下的视频合成方法。通过精心设计的生成器和鉴别器架构，再加上时空对抗目标，可以在一组不同的输入格式(包括分割掩码、草图和姿势)上获得高分辨率、逼真的、时间相干的视频结果。
在多个基准上的实验表明，与强基线相比，本文的方法具有优势。特别是该模型能够合成长达30秒的街道场景的2K分辨率视频，大大提高了视频合成的技术水平。最后，将该方法应用于未来的视频预测，表现优于几个最先进的系统。
概要总结
英伟达的研究人员引入了一种新的视频合成方法。该框架基于条件甘斯。具体地说，该方法将精心设计的发生器和鉴别器与时空对抗性目标相结合。实验表明，所提出的vid2vid方法可以在不同的输入格式(包括分割掩码、草图和姿势)上合成高分辨率、逼真、时间相干的视频。它还可以预测下一帧，其结果远远优于基线模型。

市场营销和广告可以从vid2vid方法创造的机会中获益(例如，在视频中替换面部甚至整个身体)。然而，这应该谨慎使用，需要想到道德伦理方面的一些顾虑。
代码
英伟达团队提供了本研究论文在GitHub上的原始实现的代码：
https://github.com/NVIDIA/vid2vid

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


7 
唇读



唇读（lipreading）是指根据说话人的嘴唇运动解码出文本的任务。传统的方法是将该问题分成两步解决：设计或学习视觉特征、以及预测。最近的深度唇读方法是可以端到端训练的（Wand et al., 2016; Chung & Zisserman, 2016a）。目前唇读的准确度已经超过了人类。







Google DeepMind 与牛津大学合作的一篇论文《Lip Reading Sentences in the Wild》介绍了他们的模型经过电视数据集的训练后，性能超越 BBC 的专业唇读者。



该数据集包含 10 万个音频、视频语句。音频模型：LSTM，视频模型：CNN + LSTM。这两个状态向量被馈送至最后的 LSTM，然后生成结果（字符）。



 人工合成奥巴马：嘴唇动作和音频的同步



华盛顿大学进行了一项研究，生成美国前总统奥巴马的嘴唇动作。选择奥巴马的原因在于网络上有他大量的视频（17 小时高清视频）。



# Art

## Neural Style 风格迁移
1.1 风格迁移 
Neural Style

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


1.2

多风格及任意风格转换
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



1.3  语义合成图像：涂鸦拓展

1 Neural Doodle 
纹理转换的另外一个非常有意思的应用是Neural Doodle，运用这个技术，我们可以让三岁的小孩子都轻易地像莫奈一样成为绘画大师。这个技术本质上其实就是先对一幅世界名画（比如皮埃尔-奥古斯特·雷诺阿的Bank of a River）做一个像素分割，得出它的语义图，让神经网络学习每个区域的风格。 
然后，我们只需要像小孩子一样在这个语义图上面涂鸦（比如，我们想要在图片的中间画一条河，在右上方画一棵树），神经网络就能根据语义图上的区域渲染它，最后得出一幅印象派的大作。
Faster
https://github.com/DmitryUlyanov/fast-neural-doodle
实时
https://github.com/DmitryUlyanov/online-neural-doodle



英伟达出品的GauGAN：你画一幅涂鸦，用颜色区分每一块对应着什么物体，它就能照着你的大作，合成以假乱真的真实世界效果图。在AI界，你的涂鸦有个学名，叫“语义布局”。

要实现这种能力，GauGAN靠的是空间自适应归一化合成法SPADE架构。这种算法的论文Semantic Image Synthesis with Spatially-Adaptive Normalization已经被CVPR 2019接收，而且还是口头报告（oral）。

这篇论文的一作，照例还是实习生。另外几位作者来自英伟达和MIT，CycleGAN的创造者华人小哥哥朱俊彦也在其中。

在基于语义合成图像这个领域里，这可是目前效果最强的方法。


在正在举行的英伟达GTC 19大会上，GauGAN已表态了。美国时候周三周五Ting-Chun Wang和Ming-Yu Liu还将举行相干演讲。

论文地点：https://arxiv.org/abs/1903.07291

GitHub地点（代码行将上线）：https://github.com/NVlabs/SPADE

项目地点：https://nvlabs.github.io/SPADE/

https://36kr.com/p/5187136

1.4

Controlling Perceptual Factors in Neural Style Transfer
颜色控制颜色控制
在以前的风格转换中，生成图的颜色都会最终变成style图的颜色，但是很多时候我们并不希望这样。其中一种方法是，将RGB转换成YIQ，只在Y上进行风格转换，因为I和Q通道主要是保存了颜色信息。 




1.5 Deep Photo Style Transfer
本文在 Neural Style algorithm [5] 的基础上进行改进，主要是在目标函数进行了修改，加了一项 Photorealism regularization，修改了一项损失函数引入 semantic segmentation 信息使其在转换风格时 preserve the image structure
贡献是将从输入图像到输出图像的变换约束在色彩空间的局部仿射变换中，将这个约束表示成一个完全可微的参数项。我们发现这种方法成功地抑制了图像扭曲，在各种各样的场景中生成了满意的真实图像风格变换，包括一天中时间变换，天气，季节和艺术编辑风格变换。


以前都是整幅图stransfer的，然后他们想只对一幅图的单个物体进行stransfer，比如下面这幅图是电视剧Son of Zorn的剧照，设定是一个卡通人物生活在真实世界。他们还说这种技术可能在增强现实起作用，比如Pokemon go. 



1.5 Colourful Image Colorization

Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification

Colourful Image Colourization 
2016 ECCV 里加州大学伯克利分校的一篇文章介绍的方法。这个方法与之前方法的不同之处在于，它把照片上色看成是一个分类问题——预测三百多种颜色在图片每一个像素点上的概率分布。这种方法tackle了这个任务本身的不确定性，例如，当你看到一个黑白的苹果时，你可能会觉得它是红色的，但如果这个苹果是青色的，其实也并没有多少违和感。大家也可以到作者的网站网站来试用他们的demo。 
https://richzhang.github.io/colorization/


Real-Time User-Guided Image Colorization with Learned Deep Priors

UC Berkeley 的研究人员近日推出了一种利用深度学习对黑白图像进行实时上色的模型，并开源了相关代码。该研究的论文将出现在 7 月 30 日在洛杉矶举行的 SIGGRAPH 2017 计算机图像和交互技术大会上。
论文链接：https://arxiv.org/abs/1705.02999
Demo 和代码链接：https://richzhang.github.io/ideepcolor/

在计算机图形学领域中，一直存在两种为图片上色的方向：用户引导上色和数据驱动的自动上色方式。第一种范式是由 Levin 等人在 2004 年开创的，用户通过彩色画笔在灰度图像中进行引导性上色，随后优化算法会生成符合用户逻辑的上色结果。这种方法可以保留人工上色的部分性质，因而经常会有绝佳的表现，但往往需要密集的用户交互次数（有时超过五十次）。




Deep Painterly Harmonization
开源地址：https://github.com/luanfujun/deep-painterly-harmonization

首先从我最喜爱的一个开源项目讲起。我希望你花点时间仅仅来欣赏一下上面的图像。你能分辨出哪张是由人类做的，哪张是由机器生成的吗？我确定你不能。这里，第一个画面是输入图像（原始的），而第三个画面是由这项技术所生成的。
很惊讶，是吗？这个算法将你选择的外部物体添加到了任意一张图像上，并成功让它看上去好像本来就应该在那里一样。你不妨查看这个代码，然后尝试亲自到一系列不同的图像上去操作这项技术。
Image Outpainting
开源地址：https://github.com/bendangnuksung/Image-OutPainting

如果我给你一张图像，并让你通过想象图像在图中完整场景呈现时的样子，来扩展它的画面边界，你会怎么办？正常来说，你可能会把这个图导入到某个图像编辑软件里进行操作。但是现在有了一个非常棒的新软件——你可以用几行代码就实现这项操作。
这个项目是斯坦福大学「Image Outpainting」论文（论文地址：https://cs230.stanford.edu/projects_spring_2018/posters/8265861.pdf ，
这是一篇无比惊艳并配有示例说明的论文——这就是大多数研究论文所应有的样子！）的 Keras 实现。你或者可以从头开始创建模型，或者也可以使用这个开源项目作者所提供的模型。深度学习从来不会停止给人们带来惊喜。


字体合成
英文 ：
Handwriting Synthesis（手写体合成）


1
这个项目来自亚历克斯 · 格雷夫斯（Alex Graves）撰写的论文（Generating Sequences with Recurrent Neural Networks）《用 RNN 生成序列》，正如存储库的名称所示，您可以生成不同风格的手写，是其中手写体合成实验的实现，它可以生成不同风格的手写字迹。模型包括初始化和偏置两个部分，其中初始化控制样例的风格，偏置控制样例的整洁度。
作者在 GitHub 页面上呈现的样本的多样性真的很吸引人。他正在寻找贡献者来加强存储库，所以如果您有兴趣，可以研究去看看。


2
论文：Multi-Content GAN for Few-Shot Font Style Transfer

论文链接：https://arxiv.org/abs/1712.00516
GitHub 链接：https://github.com/azadis/MC-GAN
原文链接：http://bair.berkeley.edu/blog/2018/03/13/mcgan/

论文为cvpr2018，伯克利的BAIR实验室和adobe合作的论文。


3 汉字
Github 用户 kaonashi-tyc 将 字体设计 的过程转化为一个“风格迁移”（style transfer）的问题，使用条件 GAN 自动将输入的汉字转化为另一种字体（风格）的汉字，效果相当不错。


动机

创造字体一直是一件难事，中文字体更难，毕竟汉字有26000多个，要完成一整套设计需要很长的时间。

而 Github 用户 kaonashi-tyc 想到的解决方案是只手工设计一部分字体，然后通过深度学习算法训练出剩下的字体，毕竟汉字也是各种「零件」组成的。

于是，作者将字体设计的过程转化为一个“风格迁移”（style transfer）的问题。他用两种不同字体作为训练数据，训练了一个神经网络，训练好的神经网络自动将输入的汉字转化为另一种字体（风格）的汉字。

作者使用风格迁移解决中文字体生成的问题，同时加上了条件生成对抗网络（GAN）的力量。

项目地址：kaonashi-tyc/zi2zi






艺术字 
本文首次提出用端到端的方案来解决从少量相同风格的字体中合成其他艺术字体，例如 A-Z 26 个相同风格的艺术字母，已知其中 A-D 的艺术字母，生成剩余 E-Z 的艺术字母。

本文研究的问题看上去没啥亮点，但在实际应用中，很多设计师在设计海报或者电影标题的字体时，只会创作用到的字母，但想将风格迁移到其他设计上时，其他的一些没设计字母需要自己转化，造成了不必要的麻烦。

如何从少量（5 个左右）的任意类型的艺术字中泛化至全部 26 个字母是本文的难点。本文通过对传统 Condition GAN 做扩展，提出了 Stack GAN 的两段式架构，首先通过 Conditional GAN #1 根据已知的字体生成出所有 A-Z 的字体，之后通过 Conditional GAN #2 加上颜色和艺术点缀。

关于作者：黄亢，卡耐基梅隆大学硕士，研究方向为信息抽取和手写识别，现为波音公司数据科学家。

■ 论文 | Multi-Content GAN for Few-Shot Font Style Transfer

■ 链接 | https://www.paperweekly.site/papers/1781

■ 源码 | https://github.com/azadis/MC-GAN

其他 

 SketchRNN：教机器画画



你可能看过谷歌的 Quick, Draw! 数据集，其目标是 20 秒内绘制不同物体的简笔画。谷歌收集该数据集的目的是教神经网络画画。

专业摄影作品



谷歌已经开发了另一个非常有意思的 GAN 应用，即摄影作品的选择和改进。开发者在专业摄影作品数据集上训练 GAN，其中生成器试图改进照片的表现力（如更好的拍摄参数和减少对滤镜的依赖等），判别器用于区分「改进」的照片和真实的作品。
训练后的算法会通过 Google Street View 搜索最佳构图，获得了一些专业级的和半专业级的作品评分。

：Creatism: A deep-learning photographer capable of creating professional work（https://arxiv.org/abs/1707.03491）。
Showcase：https://google.github.io/creatism/

https://www.sohu.com/a/157091073_473283



　DeepMasterPrint 万能指纹


人人来跳舞
标题：人人都在跳舞
作者：Caroline Chan, Shiry Ginosar, Tinghui Zhou, Alexei A. Efros
https://arxiv.org/abs/1808.07371
论文摘要
本文提出了一种简单的“按我做”的动作转移方法：给定一个人跳舞的源视频，我们可以在目标对象执行标准动作几分钟后将该表演转换为一个新的(业余)目标。
本文提出这个问题作为每帧图像到图像的转换与时空平滑。利用位姿检测作为源和目标之间的中间表示，我们调整这个设置为时间相干视频生成，包括现实的人脸合成。学习了从位姿图像到目标对象外观的映射。视频演示可以在https://youtu.be/PCBTZh41Ris找到。
概要总结
加州大学伯克利分校的研究人员提出了一种简单的方法，可以让业余舞蹈演员像专业舞蹈演员一样表演，从而生成视频。如果你想参加这个实验，你所需要做的就是录下你自己表演一些标准动作的几分钟的视频，然后拿起你想要重复的舞蹈的视频。
神经网络将完成主要工作：它将问题解决为具有时空平滑的每帧图像到图像的转换。通过将每帧上的预测调整为前一时间步长的预测以获得时间平滑度并应用专门的GAN进行逼真的面部合成，该方法实现了非常惊人的结果。

核心思想
“跟我做”动传递被视为每帧图像到图像的平移，姿势棒图作为源和目标之间的中间表示
预先训练的最先进的姿势检测器根据源视频创建姿势棒图；
应用全局姿势标准化来解释框架内的体形和位置中的源和目标主体之间的差异；
标准化的姿势棒图被映射到目标对象。
为了使视频流畅，研究人员建议在先前生成的帧上调节发生器，然后将两个图像提供给鉴别器。 姿势关键点上的高斯平滑允许进一步减少抖动。
为了生成更逼真的面部，该方法包括额外的面部特定GAN，其在主生成完成之后刷新面部。
最重要的成果
根据定性和定量评估，提出了一种优于强基线(pix2pixHD)的运动传输新方法。
演示特定于人脸的GAN为输出视频添加了相当多的细节。
AI社区的评价
谷歌大脑的技术人员汤姆·布朗(Tom Brown)说：“总的来说，我觉得这真的很有趣，而且执行得很好。期待代码的公布，这样我就可以开始训练我的舞步了。”
Facebook人工智能研究工程师Soumith Chintala说：“卡洛琳·陈(Caroline Chan)、阿廖沙·埃夫罗斯(Alyosha Efros)和团队将舞蹈动作从一个主题转移到另一个主题。只有这样我才能跳得好。了不起的工作! ! !”
未来研究方向
用时间相干的输入和专门为运动传输优化的表示来替换姿态棒图。
可能的应用
“跟我做”在制作营销和宣传视频时，可能会应用动作转移来替换主题。
代码
本研究论文的PyTorch实现可在GitHub上获得：
https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow


http://www.sohu.com/a/294911565_100024677