

## 3.8 字体合成


### 英文手写体 Handwriting Synthesis

### Generating Sequences with Recurrent Neural Networks
这个项目来自亚历克斯 · 格雷夫斯（Alex Graves）撰写的论文（Generating Sequences with Recurrent Neural Networks）《用 RNN 生成序列》，正如存储库的名称所示，您可以生成不同风格的手写，是其中手写体合成实验的实现，它可以生成不同风格的手写字迹。模型包括初始化和偏置两个部分，其中初始化控制样例的风格，偏置控制样例的整洁度。
作者在 GitHub 页面上呈现的样本的多样性真的很吸引人。他正在寻找贡献者来加强存储库，所以如果您有兴趣，可以研究去看看。

https://github.com/sherjilozair/char-rnn-tensorflow

https://github.com/karpathy/char-rnn

https://github.com/hardmaru/write-rnn-tensorflow

### 艺术字体 

### Multi-Content GAN for Few-Shot Font Style Transfer

cvpr2018，伯克利的BAIR实验室和adobe合作的论文

本文首次提出用端到端的方案来解决从少量相同风格的字体中合成其他艺术字体，例如 A-Z 26 个相同风格的艺术字母，已知其中 A-D 的艺术字母，生成剩余 E-Z 的艺术字母。

本文研究的问题看上去没啥亮点，但在实际应用中，很多设计师在设计海报或者电影标题的字体时，只会创作用到的字母，但想将风格迁移到其他设计上时，其他的一些没设计字母需要自己转化，造成了不必要的麻烦。

如何从少量（5 个左右）的任意类型的艺术字中泛化至全部 26 个字母是本文的难点。本文通过对传统 Condition GAN 做扩展，提出了 Stack GAN 的两段式架构，首先通过 Conditional GAN #1 根据已知的字体生成出所有 A-Z 的字体，之后通过 Conditional GAN #2 加上颜色和艺术点缀。

作者：黄亢，卡耐基梅隆大学硕士，研究方向为信息抽取和手写识别，现为波音公司数据科学家。

论文链接： https://arxiv.org/abs/1712.00516  https://www.paperweekly.site/papers/1781

GitHub 链接：https://github.com/azadis/MC-GAN

原文链接：http://bair.berkeley.edu/blog/2018/03/13/mcgan/




### 汉字

Github 用户 kaonashi-tyc 将 字体设计 的过程转化为一个“风格迁移”（style transfer）的问题，使用条件 GAN 自动将输入的汉字转化为另一种字体（风格）的汉字，效果相当不错。


动机

创造字体一直是一件难事，中文字体更难，毕竟汉字有26000多个，要完成一整套设计需要很长的时间。

而 Github 用户 kaonashi-tyc 想到的解决方案是只手工设计一部分字体，然后通过深度学习算法训练出剩下的字体，毕竟汉字也是各种「零件」组成的。

于是，作者将字体设计的过程转化为一个“风格迁移”（style transfer）的问题。他用两种不同字体作为训练数据，训练了一个神经网络，训练好的神经网络自动将输入的汉字转化为另一种字体（风格）的汉字。

作者使用风格迁移解决中文字体生成的问题，同时加上了条件生成对抗网络（GAN）的力量。

项目地址：https://github.com/kaonashi-tyc/zi2zi


----------------

## 3.9 视频生成

通常来说，视频有相对静止的背景和运动的前景组成。VideoGAN 使用一个两阶段的生成器，3D CNN 生成器生成运动前景，2D CNN 生成器生成静止的背景。Pose GAN 则使用 VAE 和 GAN 生成视频，首先，VAE 结合当前帧的姿态和过去的姿态特征预测下一帧的运动信息，然后 3D CNN 使用运动信息生成后续视频帧。Motion and Content GAN(MoCoGAN) 则提出在隐空间对运动部分和内容部分进行分离，使用 RNN 去建模运动部分。

### vid2vid

视频到视频的合成 ： Video-to-Video Synthesis
作者：Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, Bryan Catanzaro

https://arxiv.org/abs/1808.06601

英伟达的研究人员引入了一种新的视频合成方法。该框架基于生成对抗学习框架。实验表明，所提出的vid2vid方法可以在不同的输入格式(包括分割掩码、草图和姿势)上合成高分辨率、逼真、时间相干的视频。它还可以预测下一帧，其结果远远优于基线模型。

https://github.com/NVIDIA/vid2vid



### 人人来跳舞

作者：Caroline Chan, Shiry Ginosar, Tinghui Zhou, Alexei A. Efros

https://youtu.be/PCBTZh41Ris

加州大学伯克利分校的研究人员提出了一种简单的方法，可以让业余舞蹈演员像专业舞蹈演员一样表演，从而生成视频。如果你想参加这个实验，你所需要做的就是录下你自己表演一些标准动作的几分钟的视频，然后拿起你想要重复的舞蹈的视频。

神经网络将完成主要工作：它将问题解决为具有时空平滑的每帧图像到图像的转换。利用位姿检测作为源和目标之间的中间表示,通过将每帧上的预测调整为前一时间步长的预测以获得时间平滑度并应用专门的GAN进行逼真的面部合成，该方法实现了非常惊人的结果。

https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow

http://www.sohu.com/a/294911565_100024677

https://ceit.aut.ac.ir/~khalooei/tutorials/gan/


### Recycle-GAN

-------------------

### 3.10 3D
《Visual Object Networks: Image Generation with Disentangled 3D Representation》，描述了一种用GAN生成3D图片的方法。

这篇文章被近期在蒙特利尔举办的NeurIPS 2018

--------------------------

# Level 4: GANs Applications in Others

## 4.1 NLP 自然语言处理领域
相比于 GAN 在图像领域的应用，GAN 在文本，语音领域的应用要少很多。主要原因有两个：

GAN 在优化的时候使用 BP 算法，对于文本，语音这种离散数据，GAN 没法直接跳到目标值，只能根据梯度一步步靠近。
对于序列生成问题，每生成一个单词，我们就需要判断这个序列是否合理，可是 GAN 里面的判别器是没法做到的。除非我们针对每一个 step 都设置一个判别器，这显然不合理。为了解决上述问题，强化学习中的策略梯度下降（Policy gredient descent）被引入到 GAN 中的序列生成问题。

GAN在自然语言处理上的应用可以分为两类：生成文本、根据文本生成图像。其中，生成文本包括两种：根据隐向量（噪声）生成一段文本；对话生成。

 
BERT和自然语言处理（NLP）

### 4.2.1 对话生成
 Li J等2017年发表的Adversarial Learning for Neural Dialogue Generation[16]显示了GAN在对话生成领域的应用。实验效果如图11。可以看出，生成的对话具有一定的相关性，但是效果并不是很好，而且这只能做单轮对话。



如图11 Li J对话生成效果

文本到图像的翻译（text to image）

文本到图像的翻译指GAN的输入是一个描述图像内容的一句话，比如“一只有着粉色的胸和冠的小鸟”，那么所生成的图像内容要和这句话所描述的内容相匹配。



在ICML 2016会议上，Scott Reed等[17]人提出了基于CGAN的一种解决方案将文本编码作为generator的condition输入；对于discriminator，文本编码在特定层作为condition信息引入，以辅助判断输入图像是否满足文本描述。作者提出了两种基于GAN的算法，GAN-CLS和GAN-INT。



### 4.2.2 Text2image

Torch 版本：https://github.com/reedscot/icml2016

TensorFlow+Theano 版本：https://github.com/paarthneekhara/text-to-image


 Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman and Alexei A. Efros. “Generative Visual Manipulation on the Natural Image Manifold”, ECCV 2016.

从 Text 生成 Image，比如从图片标题生成一个具体的图片。这个过程需要不仅要考虑生成的图片是否真实，还应该考虑生成的图片是否符合标题里的描述。比如要标题形容了一个黄色的鸟，那么就算生成的蓝色鸟再真实，也是不符合任务需求的。为了捕捉或者约束这种条件，他们提出了 matching-aware discriminator 的思想，让本来的 D 的目标函数中的两项，扩大到了三项：


StackGAN


Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaolei Huang, Xiaogang Wang, Dimitris Metaxas. “StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks”. arXiv preprint 2016.
第三篇这方面的工作[20]可以粗略认为是 LAPGAN[16] 和 matching-aware[18] 的结合。他们提出的 StackGAN[20] 做的事情从标题生成鸟类，但是生成的过程则是像 LAPGAN 一样层次化的，从而实现了 256X256 分辨率的图片生成过程。StackGAN 将图片生成分成两个阶段，阶段一去捕捉大体的轮廓和色调，阶段二加入一些细节上的限制从而实现精修。这个过程效果很好，甚至在某些数据集上以及可以做到以假乱真：

ObjGAN，可以通过关注文本描述中最相关的单词和预先生成的语义布局(semantic layout)来合成显著对象。

https://www.microsoft.com/en-us/research/uploads/prod/2019/06/1902.10740.pdf


其他：
textGAN MailGAN

MaskGAN


## 4.2 语音方向

### 4.2.1 音乐生成

RNN-GAN 使用 LSTM 作为生成器和判别器，直接生成整个音频序列。然而，正如上面提到的，音乐当做包括歌词和音符，对于这种离散数据生成问题直接使用 GAN 存在很多问题，特别是生成的数据缺乏局部一致性。

相比之下，SeqGAN 把生成器的输出作为一个智能体 (agent) 的策略，而判别器的输出作为奖励 (reward)，使用策略梯度下降来训练模型。ORGAN 则在 SeqGAN 的基础上，针对具体的目标设定了一个特定目标函数。

RNN-GAN 使用 LSTM 作为生成器和判别器，直接生成整个音频序列。然而，正如上面提到的，音乐当做包括歌词和音符，对于这种离散数据生成问题直接使用 GAN 存在很多问题，特别是生成的数据缺乏局部一致性。

相比之下，SeqGAN 把生成器的输出作为一个智能体 (agent) 的策略，而判别器的输出作为奖励 (reward)，使用策略梯度下降来训练模型。ORGAN 则在 SeqGAN 的基础上，针对具体的目标设定了一个特定目标函数。


wavenet

GANSynth是一种快速生成高保真音频的新方法
http://www.elecfans.com/d/877752.html



audio (currently no meta)   

https://github.com/CorentinJ/Real-Time-Voice-Cloning

https://github.com/andabi/deep-voice-conversion

https://github.com/r9y9/wavenet_vocoder

https://github.com/kuleshov/audio-super-res

https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses

https://github.com/drethage/speech-denoising-wavenet

### 4.2.2 语言和语音


VAW-GAN(Variational autoencoding Wasserstein GAN) 结合 VAE 和 WGAN 实现了一个语音转换系统。编码器编码语音信号的内容，解码器则用于重建音色。由于 VAE 容易导致生成结果过于平滑，所以此处使用 WGAN 来生成更加清晰的语音信号。

Generative Adversarial Text to Image Synthesis	Reed & et al.	ICML 2016
Conditional Generative Adversarial Networks for Speech Enhancement and Noise-Robust Speaker Verification	Michelsanti & Tan	Interspeech 2017	
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network	Ledig & et al.	CVPR 2017	
SalGAN: Visual Saliency Prediction with Generative Adversarial Networks	Pan & et al.	CVPR 2017	
SAGAN: Self-Attention Generative Adversarial Networks	Zhang & et al.	NIPS 2018	
Speaker Adaptation for High Fidelity WaveNet Vocoder with GAN	Tian & et al.	arXiv Nov 2018	
MTGAN: Speaker Verification through Multitasking Triplet Generative Adversarial Networks	Ding & et al.	arXiv Mar 2018	
Adversarial Learning and Augmentation for Speaker Recognition	Zhang & et al.	Speaker Odyssey 2018 / ISCA 2018	
Investigating Generative Adversarial Networks based Speech Dereverberation for Robust Speech Recognition	Wang & et al.	Interspeech 2018	
On Enhancing Speech Emotion Recognition using Generative Adversarial Networks	Sahu & et al.	Interspeech 2018	
Robust Speech Recognition Using Generative Adversarial Networks	Sriram & et al.	ICASSP 2018	
Adversarially Learned One-Class Classifier for Novelty Detection	Sabokrou & khalooei & et al.	CVPR 2018	
Generalizing to Unseen Domains via Adversarial Data Augmentation	Volpi & et al.	NeurIPS (NIPS) 2018	
Generative Adversarial Networks for Unpaired Voice Transformation on Impaired Speech	Chen & lee & et al.	Submitted on ICASSP 2019	
Generative Adversarial Speaker Embedding Networks for Domain Robust End-to-End Speaker Verification	Bhattacharya & et al.	Submitted on ICASSP 2019	


### 4.2.3 唇读

唇读（lipreading）是指根据说话人的嘴唇运动解码出文本的任务。传统的方法是将该问题分成两步解决：设计或学习视觉特征、以及预测。最近的深度唇读方法是可以端到端训练的（Wand et al., 2016; Chung & Zisserman, 2016a）。目前唇读的准确度已经超过了人类。

Google DeepMind 与牛津大学合作的一篇论文《Lip Reading Sentences in the Wild》介绍了他们的模型经过电视数据集的训练后，性能超越 BBC 的专业唇读者。

该数据集包含 10 万个音频、视频语句。音频模型：LSTM，视频模型：CNN + LSTM。这两个状态向量被馈送至最后的 LSTM，然后生成结果（字符）。

 人工合成奥巴马：嘴唇动作和音频的同步



华盛顿大学进行了一项研究，生成美国前总统奥巴马的嘴唇动作。选择奥巴马的原因在于网络上有他大量的视频（17 小时高清视频）。


## Speech Denoising

https://github.com/drethage/speech-denoising-wavenet
https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses
https://github.com/auspicious3000/WaveNet-Enhancement
https://github.com/lucianodato/speech-denoiser

## Adversarial Programming
https://github.com/Prinsphield/Adversarial_Reprogramming
https://github.com/kcelia/adversarial_reprogramming_of_neural_network
https://github.com/lizhuorong/Adversarial-Reprogramming-tensorflow
https://github.com/rajatvd/AdversarialReprogramming
https://github.com/savan77/Adversarial-Reprogramming
https://github.com/mishig25/adversarial-reprogramming-keras



--------------------------------

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







---------------------------

其他：

Deep Learning: State of the Art*
(Breakthrough Developments in 2017 & 2018)

• AdaNet: AutoML with Ensembles
• AutoAugment: Deep RL Data Augmentation
• Training Deep Networks with Synthetic Data
• Segmentation Annotation with Polygon-RNN++
• DAWNBench: Training Fast and Cheap
• Video-to-Video Synthesis
• Semantic Segmentation
• AlphaZero & OpenAI Five
• Deep Learning Frameworks


AdaNet：可集成学习的AutoML

AutoAugment：用强化学习做数据增强

用合成数据训练深度神经网络

用Polygon-RNN++做图像分割自动标注

DAWNBench：寻找快速便宜的训练方法



语义分割

AlphaZero和OpenAI Five

深度学习框架
https://github.com/lexfridman/mit-deep-learning



SAC-X






-----------------------------------------------------------------------------

https://github.com/wiseodd/generative-models


PPGAN - Anh Nguyen, arXiv:1612.00005v1



SeqGAN - Lantao Yu, arxiv: 1609.05473

EBGAN - Junbo Zhao, arXiv:1609.03126v2

VAEGAN - Anders Boesen Lindbo Larsen, arxiv: 1512.09300

......

特定任务中提出来的模型，如GAN-CLS、GAN-INT、SRGAN、iGAN、IAN 等


IAN

Theano 版本：https://github.com/ajbrock/Neural-Photo-Editor


TensorFlow 版本：https://github.com/yenchenlin/pix2pix-tensorflow

GAN for Neural dialogue generation

Torch 版本：https://github.com/jiweil/Neural-Dialogue-Generation


GAN for Imitation Learning

Theano 版本：https://github.com/openai/imitation

SeqGAN

TensorFlow 版本：https://github.com/LantaoYu/SeqGAN


Qi G J. Loss-Sensitive Generative Adversarial Networks onLipschitz Densities[J]. arXiv preprint arXiv:1701.06264, 2017.

Li J, Monroe W, Shi T, et al. Adversarial Learning for NeuralDialogue Generation[J]. arXiv preprint arXiv:1701.06547, 2017.

Sønderby C K, Caballero J, Theis L, et al. Amortised MAPInference for Image Super-resolution[J]. arXiv preprint arXiv:1610.04490, 2016.

Ravanbakhsh S, Lanusse F, Mandelbaum R, et al. Enabling DarkEnergy Science with Deep Generative Models of Galaxy Images[J]. arXiv preprintarXiv:1609.05796, 2016.

Ho J, Ermon S. Generative adversarial imitationlearning[C]//Advances in Neural Information Processing Systems. 2016:4565-4573.

Zhu J Y, Krähenbühl P, Shechtman E, et al. Generative visualmanipulation on the natural image manifold[C]//European Conference on ComputerVision. Springer International Publishing, 2016: 597-613.

Isola P, Zhu J Y, Zhou T, et al. Image-to-image translationwith conditional adversarial networks[J]. arXiv preprint arXiv:1611.07004,2016.

Shrivastava A, Pfister T, Tuzel O, et al. Learning fromSimulated and Unsupervised Images through Adversarial Training[J]. arXivpreprint arXiv:1612.07828, 2016.

Ledig C, Theis L, Huszár F, et al. Photo-realistic singleimage super-resolution using a generative adversarial network[J]. arXivpreprint arXiv:1609.04802, 2016.

Nguyen A, Yosinski J, Bengio Y, et al. Plug & playgenerative networks: Conditional iterative generation of images in latentspace[J]. arXiv preprint arXiv:1612.00005, 2016.

Yu L, Zhang W, Wang J, et al. Seqgan: sequence generativeadversarial nets with policy gradient[J]. arXiv preprint arXiv:1609.05473,2016.

Lotter W, Kreiman G, Cox D. Unsupervised learning of visualstructure using predictive generative networks[J]. arXiv preprintarXiv:1511.06380, 2015.

Reed S, Akata Z, Yan X, et al. Generative adversarial textto image synthesis[C]//Proceedings of The 33rd International Conference onMachine Learning. 2016, 3.

Brock A, Lim T, Ritchie J M, et al. Neural photo editingwith introspective adversarial networks[J]. arXiv preprint arXiv:1609.07093,2016.

Pfau D, Vinyals O. Connecting generative adversarialnetworks and actor-critic methods[J]. arXiv preprint arXiv:1610.01945, 2016.



Neural Dialogue Generation
https://github.com/jiweil/Neural-Dialogue-Generation








