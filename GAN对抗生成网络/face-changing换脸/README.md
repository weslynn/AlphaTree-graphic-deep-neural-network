# GANs Applications in CV

### 3.6.3 换脸
　
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

  5、 ZAO


工具 ：

https://github.com/iperov/DeepFaceLab


