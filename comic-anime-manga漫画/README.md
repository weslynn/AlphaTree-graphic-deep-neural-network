# 动漫二次元相关

动漫常用的数据库有如下几个：
- nico-opendata
nico-opendata https://nico-opendata.jp/en/index.html
At Niconico, we are providing a wide variety of data from our services,
to be used for academic purposes.  仅限用于学术目的。

- Nico-Illust    超过40万张图像（插图） https://nico-opendata.jp/en/seigadata/index.html
This dataset contains over 400,000 images (illustraions) from Niconico Seiga and Niconico Shunga.
Niconico Seiga is a community for posting illustrations.  一个发布插图的社区。http://seiga.nicovideo.jp/
Niconico Shunga is a community for posting illustrations, where submission of explicit content is allowed. Viewers under age 18 are prohibited in this website.  本网站禁止18岁以下观众观看。  http://seiga.nicovideo.jp/shunga/

- Danbooru 

Danbooru2018 is a large-scale anime image database with 3.33m+ images annotated with 99.7m+ tags; it can be useful for machine learning purposes such as image recognition and generation.
[2018](https://www.gwern.net/Danbooru2018) [2017](http://www.gwern.net/Danbooru2017)

大神 gwern 的工作，数据来自 danbooru。

| Site | Intro  |Link|
| --- | --- | ---|
|danbooru:D站 |最老的恶魔，各系列的图片都很齐备||
|gelbooru :G站| danbooru以外最大的备份||
|3dbooru | cosplay与写真收集||
|yandere | 高清扫图，曾经的萌妹||
|konachan | 高质量二次元壁纸||
|iqdb | 图片逆向搜索||
|saucenao | 图片逆向搜索||
|whatanime | 动画截图逆向搜索||
|safebooru:S站 |A LARGE-SCALE CROWDSOURCED AND TAGGED ANIME ILLUSTRATION DATASET 一个大规模的众包和标记动画插图数据集|[link](https://safebooru.donmai.us/) [link](https://safebooru.org/)|
|getchu |游戏人设网站|[link](https://www.getchu.com/) [爬虫](https://github.com/One-sixth/getchu_character_picture_grabber)|
|Manga109||[link](http://www.manga109.org/en/index.html)|



国内数据库

iCartoonFace： 爱奇艺
- arxiv:https://arxiv.org/abs/1907.13394
- link:https://www.arxiv-vanity.com/papers/1907.13394/
![ANIMEface](https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/intro_icartoonface.jpg)

新出的数据集，需要邮件申请，不过申请似乎没有回应。

## 动漫人脸检测

### LBP
目前常用的动漫人脸检测方法为LBP,主要是因为识别出来的人脸，准确率较高，但是检出率相对较低。如果希望拥有较高检出率，可以使用一些tiny face 的检测方法。

https://github.com/nagadomi/lbpcascade_animeface

## 动漫人脸特征点检测

![ANIMEface](https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/animeface2009.png)
在animeface2009上做的基本的眼嘴鼻特征点检测
- website：http://anime.udp.jp/
- github:https://github.com/nagadomi/animeface-2009

Manga face detection based on deep neural networks fusing global and local information

https://www.sciencedirect.com/science/article/abs/pii/S0031320318303066

Manga FaceNet: Face Detection in Manga based on Deep Neural Network
- link:https://www.cs.ccu.edu.tw/~wtchu/projects/MangaFace/

Facial Landmark Detection for Manga Images

用的Manga109的数据，需要邮件申请。模型不开源，开源了manga109识别出的人脸标注点数据
<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/mangaface.jpg" width="600">

- arxiv:https://arxiv.org/pdf/1811.03214
- github:https://github.com/oaugereau/FacialLandmarkManga
- link:https://www.groundai.com/project/facial-landmark-detection-for-manga-images/

## 动漫人脸识别

### Finding Familiar Faces with a Tensorflow Object Detector, Pytorch Feature Extractor, and Spotify’s Annoy
- link: https://towardsdatascience.com/finding-familiar-faces-with-a-tensorflow-object-detector-pytorch-feature-extractor-and-spotifys-3e78858a8148
- github: https://github.com/sugi-chan/fgo_face_similarity

### transfer learning anime
- link: http://freedomofkeima.com/blog/posts/flag-15-image-recognition-for-anime-characters
- github:https://github.com/freedomofkeima/transfer-learning-anime

## 动漫的头像生成

### makegirlsmoe

最早玩人脸生成效果比较好的一家，作者总的训练框架来自于DRAGAN

![mgm](https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/mgm.png)

arxiv：https://arxiv.org/pdf/1705.07215.pdf

经过实验发现这种训练方法收敛更快并且能产生更稳定的结果。生成器G的结构类似于SRResNet

arxiv：https://arxiv.org/pdf/1609.04802.pdf

website:https://make.girls.moe/

《Create Anime Characters with A.I. !》

https://makegirlsmoe.github.io/assets/pdf/technical_report.pdf

用了Getchu

[intro](http://www.sohu.com/a/168828469_633698)

[zhihu](https://zhuanlan.zhihu.com/p/28488946)

[zhihu](https://zhuanlan.zhihu.com/p/28527956)


之后做了区块链生成小姐姐  https://crypko.ai/

![crypko](https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/crypko.png)


### BigGAN

BigGan生成效果非常惊艳。


ANIME305用BigGAN达到不错的效果

![ANIME](https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/fake_steps_12600.jpg)

https://github.com/ANIME305/Anime-GAN-tensorflow#open-sourced-dataset

gwern 用BigGan生成的256x256的效果

![ANIME](https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/samples7.jpg)


### StyleGan

Gwern 用StyleGan 跑出如下效果

![ANIME](https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/thiswaifudoesnotexist-100samples.jpg)


从真人照片到动漫头像，之前有twin—gan做了尝试，然后很多手机应用上做了一些，包括国内美图，但是多样化很差，国外google，facebook，做的emoji效果。效果不够好，

emoji：DTN-- Unsupervised Cross-domain Image Generation https://arxiv.org/abs/1611.02200

## 动漫的图像上色


| Image Type | Paper | Source | Code/Project Link  |
| --- | --- | --- |--- |
| Line art | [User-Guided Deep Anime Line Art Colorization with Conditional Adversarial Networks](https://arxiv.org/pdf/1808.03240.pdf) | ACM MM 2018 | [[code]](https://github.com/orashi/AlacGAN) |
| Line art | Style2paints V3 : [Two-stage Sketch Colorization](http://www.cse.cuhk.edu.hk/~ttwong/papers/colorize/colorize.pdf) | SIGGRAPH Asia 2018 | [[Project]](https://www.cse.cuhk.edu.hk/~ttwong/papers/colorize/colorize.html) [[Code]](https://github.com/lllyasviel/style2paints#style2paints-v3) [[Demo]](http://s2p.moe/) <br/><br/> [[PyTorch Reimplementation]](https://github.com/Pengxiao-Wang/Style2Paints_V3) |
| Line art | Paints Chainer | Online Demo | [[Demo]](https://paintschainer.preferred.tech/) [[code]](https://github.com/pfnet/PaintsChainer) |
| Line art | PaintsTensorFlow | Github Repo | [[Code]](https://github.com/rapidrabbit76/PaintsTensorFlow) |
| Manga | MangaCraft | Online Demo | [[Demo]](https://github.com/lllyasviel/MangaCraft) |
|Line art|[Comicolorization: Semi-automatic Manga Colorization](https://arxiv.org/pdf/1706.06759.pdf) |1706.06759  (DwangoMediaVillage) | [[code]](https://github.com/DwangoMediaVillage/Comicolorization)|

### Paints Chainer 

P站在“pixiv Sketch”上线过一项黑科技新功能——自动上色。早在P站此次发布自动上色前，已经有不少工具具有“自动描线”等功能。而今年1月份，日本就已经测试用AI来提供自动上色服务的“Paints Chainer”。而P站的自动上色服务也是基于前者的技术框架，升级改造后的成果。它们的背后是也是人工智能观察60万张上色插图，学习人类上色方法，经过一连串运算得出的成果。


### style2paints 

Style2paints会根据用户的颜色提示和选择的图片风格完成对图片的上色。目前共以下迭代了4个版本。

- Style2paints V1
	（信息暂未公开）
	Style Transfer for Anime Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN
	https://arxiv.org/pdf/1706.03319.pdf
	使用的是Unet
- Style2paints V2
	于2017年12月发布，使用3×3 像素点的高精度提示和风格迁移给线稿或草图上色。已下线。
	作者在Reddit上回答说，和上一版相比，style2paints 2.0大部分训练都是纯粹无监督，甚至无条件的。
	也就是说，在这个模型的训练过程中，除了对抗规则之外没有添加其他的人工定义规则，没有规则来强迫生成器神经网络照着线稿画画，而是靠神经网络自己发现，如果遵照线稿，会更容易骗过鉴别器。
	pix2pix、CycleGAN等同类模型为了确保收敛，会对学习对象添加l1 loss，鉴别器接收到的数据是成对的[input, training data]和[input, fake output]。而style2paints 2.0模型的学习目标和经典DCGAN完全相同，没有添加其他规则，鉴别器收到的也不是成对的输出。
	作者说，让这样一个模型收敛其实是很难的，何况神经网络这么深。

	https://zhuanlan.zhihu.com/p/32461125


- Style2paints V3/PaintsTransfer-Euclid

	Style2paints V3较V2版本拥有更高的精准度，是V4版本的雏形。同样已下线，但能在github的开发页面下载源码以部署到本地。

- Style2paints V4/PaintsTransfer

	于2018年11月发布，并做出以下更新：

	1) 取消了V3版本中的颜色稳定器
	2) 引入了光渲染模式和固有色模式
	3) 加入提示保存和上传功能


	填充颜色>>添加颜色渐变>>添加阴影

- Style2paints和Mangacraft的关系
	Style2paints和Mangacraft由一秒一喵/lllyasviel 开发完成的上色软件。但S2P是线稿纯上色软件，MC是黑白漫画上色软件，后台运作方式完全相同


### PI-REC
 PI-REC: Progressive Image Reconstruction Network With Edge and Color Domain.  https://arxiv.org/abs/1903.10146

https://github.com/youyuge34/PI-REC


《Anime Sketch Coloring with Swish-Gated Residual U-Net》
pradeeplam/Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet。

### 3.7.3 动漫的超分辨率/Image scaling 

### waifu2x
Github: nagadomi/waifu2x · GitHub
http://bigjpg.com/


Anime4k
https://github.com/bloc97/Anime4K

Single-Image Super-Resolution for anime/fan-art using Deep Convolutional Neural Networks.
使用卷积神经网络(Convolutional Neural Network, CNN)针对漫画风格的图片进行放大. 
效果还是相当不错的, 下面是官方的Demo图:
https://raw.githubusercontent.com/nagadomi/waifu2x/master/images/slide.png


其他：
paGAN：用单幅照片实时生成超逼真动画人物头像

　　最新引起很大反响的“换脸”技术来自华裔教授黎颢的团队，他们开发了一种新的机器学习技术paGAN，能够以每秒1000帧的速度对对人脸进行跟踪，用单幅照片实时生成超逼真动画人像，论文已被SIGGRAPH 2018接收。具体技术细节请看新智元昨天的头条报道。

　　Pinscreen拍摄了《洛杉矶时报》记者David Pierson的一张照片作为输入（左），并制作了他的3D头像（右）。 这个生成的3D人脸通过黎颢的动作（中）生成表情。这个视频是6个月前制作的，Pinscreen团队称其内部早就超越了上述结果。

　https://tech.sina.com.cn/csj/2018-08-08/doc-ihhkuskt7977099.shtml

### 3.7.4 线稿提取

Simplifying Rough Sketches using Deep Learning，作者为 Ashish Sinha。
利用LSGAN 和CRF进行线稿简化
https://github.com/bobbens/sketch_simplification/

http://hi.cs.waseda.ac.jp/~esimo/en/research/sketch/

data： getchu head https://github.com/ANIME305/Anime-GAN-tensorflow

二次元线稿 Anime-Girl-lineart-Generator
keevs https://www.deviantart.com/keevs/art/Anime-Girl-lineart-Generator-88708558

GAN变二次元 UGATIT
https://github.com/znxlwm/UGATIT-pytorch

GAN秒变肖像画！清华刘永进提出APDrawingGAN
https://baijiahao.baidu.com/s?id=1636212645611494666&wfr=spider&for=pc

http://dy.163.com/v2/article/detail/EHPRSNRT05313FBM.html

GAN生成油画效果： AI Portraits Ars https://aiportraits.com/



其他：


https://crypko.ai/#/

https://paintschainer.preferred.tech/index_zh.html

http://mangacraft.net/

https://github.com/lllyasviel/MangaCraft

https://cs.nju.edu.cn/rl/WebCaricature.htm


玩点别的：
半色调效果

https://halftonepro.com/

半色调效果在设计中经常遇到，然而ps自带的彩色半调功能单薄，大部分人也不会代码，幸好有了这个网站，常见各种半调效果都能在线完成了。

在线制作故障风动效

https://getmosh.io


中国传统颜色

http://zhongguose.com

各种中国传统颜色一览，美得一塌糊涂。


在线制作像素风GIF图

http://www.piskelapp.com/

功能强大，界面直观。可通过网页翻译插件翻译成中文后使用。


在线制作立体像素风

http://gallery.echartsjs.com/editor.html?c=xS1l7vPPwW

无需借助3D软件，可上传自定义图片，制作立体像素风，实时预览，参数可控性强，虽然比不上专业的3D软件，但制作写简单的效果还是很快的。


在线各类格式转换

https://cn.office-converter.com/

特别强大，几乎可以转换我们常用的各种格式。


光丝图谱

http://weavesilk.com/

很多人都玩过的在线画炫光网站，可以设置颜色点数等等，有时需要画点魔法等细节丰富的光，可以用这个试试，然后滤镜叠加一下，duang~ 完美。


奇幻风格合成滤镜

http://ostagram.ru/

来自俄罗斯的Ostagram，用一种基于DeepDream算法生成绘画作品，它可以学习绘画作品的画风，然后把另一张照片的画风替换成你所想要的画风，这种合成结果和那些图像处理软件的滤镜效果有着本质的区别，它看起来非常自然：

在线黑白照片AI填色
https://colourise.sg/

http://demos.algorithmia.com/colorize-photos/

