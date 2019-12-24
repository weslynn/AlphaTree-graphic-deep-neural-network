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



非日式漫画的漫画数据库：

https://cvit.iiit.ac.in/research/projects/cvit-projects/cartoonfaces

webCaricature https://cs.nju.edu.cn/rl/WebCaricature.htm 该数据集包含了252个名人的6042幅漫画图像以及5974幅人脸图像

## 动漫人脸检测

### LBP
目前常用的动漫人脸检测方法为LBP,主要是因为识别出来的人脸，准确率较高，但是检出率相对较低。如果希望拥有较高检出率，可以使用一些DNN的检测方法。

https://github.com/nagadomi/lbpcascade_animeface

### AnimeHeadDetector
用yolo v3

https://github.com/grapeot/AnimeHeadDetector


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

用的Manga109的数据，需要邮件申请。想做一个和 300 faces in-the-wild challenge一样的ibug模型，模型不开源，开源了部分识别出的人脸标注点数据，需要配合Manga109数据使用。

<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/2dpic/mangaface.jpg" width="400">

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



other: 

### LearningToPaint

 https://github.com/hzwer/ICCV2019-LearningToPaint


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


### 3.7.4 人脸转二次元

### paGAN

paGAN：用单幅照片实时生成超逼真动画人物头像

　　最新引起很大反响的“换脸”技术来自华裔教授黎颢的团队，他们开发了一种新的机器学习技术paGAN，能够以每秒1000帧的速度对对人脸进行跟踪，用单幅照片实时生成超逼真动画人像，论文已被SIGGRAPH 2018接收。具体技术细节请看新智元昨天的头条报道。

　　Pinscreen拍摄了《洛杉矶时报》记者David Pierson的一张照片作为输入（左），并制作了他的3D头像（右）。 这个生成的3D人脸通过黎颢的动作（中）生成表情。这个视频是6个月前制作的，Pinscreen团队称其内部早就超越了上述结果。

　https://tech.sina.com.cn/csj/2018-08-08/doc-ihhkuskt7977099.shtml

### APDrawingGAN

GAN秒变肖像画！清华刘永进提出APDrawingGAN
https://baijiahao.baidu.com/s?id=1636212645611494666&wfr=spider&for=pc

http://dy.163.com/v2/article/detail/EHPRSNRT05313FBM.html

### CycleGAN

Landmark Assisted CycleGAN for Cartoon Face Generation

https://arxiv.org/pdf/1907.01424.pdf


### U-GAT-IT

https://github.com/znxlwm/UGATIT-pytorch




### 3.7.4 线稿提取

参考：

https://github.com/MarkMoHR/Awesome-Sketch-Synthesis

https://blog.csdn.net/qq_33000225/article/details/90720833

- [1. Datasets](#1-datasets)
- [2. Sketch-Synthesis Approaches](#2-sketch-synthesis-approaches)
  - [1) Category-to-sketch](#1-category-to-sketch)
  - [2) Photo-to-sketch](#2-photo-to-sketch)
  - [3) Text/Attribute-to-sketch](#3-textattribute-to-sketch)
  - [4) 3D shape-to-sketch](#4-3d-shape-to-sketch)
  - [5) Sketch(pixelwise)-to-sketch(vector)](#5-sketchpixelwise-to-sketchvector)
  - [6) Art-to-sketch](#6-art-to-sketch)


---

## 1. Datasets
Here `Vector strokes` means having *svg* data. `With photos` means having the photo-sketch paired data.

<table>
  <tr>
    <td><strong>Level</strong></td>
    <td><strong>Dataset</strong></td>
    <td><strong>Source</strong></td>
    <td><strong>Vector strokes</strong></td>
    <td><strong>With photos</strong></td>
    <td><strong>Notes</strong></td>
  </tr>
  <tr>
    <td rowspan="7"><strong>Instance-level</strong></td>
    <td> <a href="http://kanjivg.tagaini.net/">KanjiVG</a> </td> 
    <td> </td> 
    <td> :heavy_check_mark: </td> 
    <td> :x: </td> 
    <td> Chinese characters </td>
  </tr>
  <tr>
    <td> <a href="http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/">TU-Berlin</a> </td> 
    <td> SIGGRAPH 2012 </td> 
    <td> :heavy_check_mark: </td> 
    <td> :x: </td> 
    <td> Multi-category hand sketches </td>
  </tr>
  <tr>
    <td> <a href="http://sketchy.eye.gatech.edu/">Sketchy</a> </td> 
    <td> SIGGRAPH 2016 </td> 
    <td> :heavy_check_mark: </td> 
    <td> :heavy_check_mark: </td> 
    <td> Multi-category photo-sketch paired </td>
  </tr>
  <tr>
    <td> <a href="https://quickdraw.withgoogle.com/data">QuickDraw</a> </td> 
    <td> ICLR 2018 </td> 
    <td> :heavy_check_mark: </td> 
    <td> :x: </td> 
    <td> Multi-category hand sketches </td>
  </tr>
  <tr>
    <td> <a href="http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html">QMUL-Shoe-Chair-V2</a> </td> 
    <td> CVPR 2016 </td> 
    <td> :heavy_check_mark: </td> 
    <td> :heavy_check_mark: </td> 
    <td> Only two categories </td>
  </tr>
  <tr>
    <td> <a href="https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset">Sketch Perceptual Grouping (SPG)</a> </td> 
    <td> ECCV 2018 </td> 
    <td> :heavy_check_mark: </td> 
    <td> :x: </td> 
    <td> With part-level semantic segmentation information </td>
  </tr>
  <tr>
    <td> <a href="https://facex.idvxlab.com/">FaceX</a> </td> 
    <td> AAAI 2019 </td> 
    <td> :heavy_check_mark: </td> 
    <td> :x: </td> 
    <td> Labeled facial sketches </td>  
  </tr>
  <tr>
    <td rowspan="4"><strong>Scene-level</strong></td>
    <td> <a href="http://www.cs.cmu.edu/~mengtial/proj/sketch/">Photo-Sketching</a> </td> 
    <td> WACV 2019 </td> 
    <td> :heavy_check_mark: </td> 
    <td> :heavy_check_mark: </td> 
    <td> ScenePhoto-sketch paired </td>
  </tr>
  <tr>
    <td> <a href="https://sketchyscene.github.io/SketchyScene/">SketchyScene</a> </td> 
    <td> ECCV 2018 </td> 
    <td> :x: </td> 
    <td> :heavy_check_mark: </td> 
    <td> With semantic/instance segmentation information </td>  
  </tr>
  <tr>
    <td> <a href="http://projects.csail.mit.edu/cmplaces/">CMPlaces</a> </td> 
    <td> TPAMI 2018 </td> 
    <td> :x: </td> 
    <td> :heavy_check_mark: </td> 
    <td> Cross-modal scene dataset </td>  
  </tr>
  <tr>
    <td> <a href="http://sweb.cityu.edu.hk/hongbofu/doc/context_based_sketch_classification_Expressive2018.pdf">Context-Skecth</a> </td> 
    <td> Expressive 2018 </td> 
    <td> :x: </td> 
    <td> :heavy_check_mark: </td> 
    <td> Context-based scene sketches for co-classification </td>  
  </tr>
  
</table>



## 2. Sketch-Synthesis Approaches

### 1) Category-to-sketch   类别-to-草图


<table>
  <tr>
    <td><strong>Level</strong></td>
    <td><strong>Paper</strong></td>
    <td><strong>Source</strong></td>
    <td><strong>Code/Project Link</strong></td>
  </tr>
  <tr>
    <td rowspan="4"><strong>Instance-level</strong></td>
    <td> <a href="https://openreview.net/pdf?id=Hy6GHpkCW">A Neural Representation of Sketch Drawings (sketch-rnn)</a> </td> 
    <td> ICLR 2018 </td> 
    <td> 
      <a href="https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn">[Code]</a> 
      <a href="https://magenta.tensorflow.org/sketch-rnn-demo">[Project]</a> 
      <a href="https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html">[Demo]</a> 
    </td>
  </tr>
  <tr>
    <td> <a href="https://arxiv.org/pdf/1709.04121.pdf">Sketch-pix2seq: a Model to Generate Sketches of Multiple Categories</a> </td> 
    <td>  </td> 
    <td> 
      <a href="https://github.com/MarkMoHR/sketch-pix2seq">[Code]</a> 
    </td>
  </tr>
  <tr>
    <td> <a href="https://idvxlab.com/papers/2019AAAI_Sketcher_Cao.pdf">AI-Sketcher : A Deep Generative Model for Producing High-Quality Sketches</a> </td> 
    <td> AAAI 2019 </td> 
    <td> <a href="https://facex.idvxlab.com/">[Project]</a> </td>
  </tr>
  <tr>
    <td> <a href="https://arxiv.org/pdf/1901.03427.pdf">Stroke-based sketched symbol reconstruction and segmentation (stroke-rnn)</a> </td> 
    <td> </td> 
    <td> </td>
  </tr>
  
</table>

---

### 2) Photo-to-sketch  照片-to-草图


- vector image generation

<table>
  <tr>
    <td><strong>Level</strong></td>
    <td><strong>Paper</strong></td>
    <td><strong>Source</strong></td>
    <td><strong>Code/Project Link</strong></td>
  </tr>
  <tr>
    <td rowspan="1"><strong>Facial</strong></td>
    <td> <a href="https://dl.acm.org/citation.cfm?id=2461964">Style and abstraction in portrait sketching</a> </td> 
    <td> TOG 2013 </td> 
    <td>
    </td>
  </tr>
  <tr>
    <td rowspan="3"><strong>Instance-level</strong></td>
    <td> <a href="https://link.springer.com/content/pdf/10.1007%2Fs11263-016-0963-9.pdf">Free-Hand Sketch Synthesis with Deformable Stroke Models</a> </td> 
    <td> IJCV 2017 </td> 
    <td>
      <a href="https://panly099.github.io/skSyn.html">[Project]</a> 
      <a href="https://github.com/panly099/sketchSynthesis">[code]</a> 
    </td>
  </tr>
  <tr>
    <td> <a href="http://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Learning_to_Sketch_CVPR_2018_paper.pdf">Learning to Sketch with Shortcut Cycle Consistency</a> </td> 
    <td> CVPR 2018 </td> 
    <td> <a href="https://github.com/seindlut/deep_p2s">[Code1]</a> <a href="https://github.com/MarkMoHR/sketch-photo2seq">[Code2]</a> </td>
  </tr>
  <tr>
    <td> <a href="http://openaccess.thecvf.com/content_cvpr_2018/papers/Muhammad_Learning_Deep_Sketch_CVPR_2018_paper.pdf">Learning Deep Sketch Abstraction</a> </td> 
    <td> CVPR 2018 </td> 
    <td>  </td>
  </tr>
</table>


- pixelwise image generation 


<table>
  <tr>
    <td><strong>Level</strong></td>
    <td><strong>Paper</strong></td>
    <td><strong>Source</strong></td>
    <td><strong>Code/Project Link</strong></td>
  </tr>
  <tr>
    <td rowspan="2"><strong>Instance-level</strong></td>
    <td> <a href="http://openaccess.thecvf.com/content_ECCV_2018/papers/Kaiyue_Pang_Deep_Factorised_Inverse-Sketching_ECCV_2018_paper.pdf">Deep Factorised Inverse-Sketching</a> </td> 
    <td> ECCV 2018 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-27/issue-6/063006/Making-better-use-of-edges-for-sketch-generation/10.1117/1.JEI.27.6.063006.short?SSO=1">Making better use of edges for sketch generation</a> </td> 
    <td> JEI 2018 </td> 
    <td> </td>
  </tr>
  <tr>
    <td rowspan="1"><strong>Scene-level</strong></td>
    <td> <a href="https://arxiv.org/pdf/1901.00542.pdf">Photo-Sketching: Inferring Contour Drawings from Images</a> </td> 
    <td> WACV 2019 </td> 
    <td>
      <a href="https://github.com/mtli/PhotoSketch">[Code]</a> 
      <a href="http://www.cs.cmu.edu/~mengtial/proj/sketch/">[Project]</a> 
    </td>
  </tr>
</table>

---

### 3) Text/Attribute-to-sketch  文本-to-草图 



| Level | Paper | Source | Code/Project Link |
| --- | --- | --- | --- |
| **Scene-level** | [Sketchforme: Composing Sketched Scenes from Text Descriptions for Interactive Applications](https://arxiv.org/pdf/1904.04399.pdf) | UIST 2019 |  |
| **Facial** | [Text2Sketch: Learning Face Sketch from Facial Attribute Text](https://ieeexplore.ieee.org/abstract/document/8451236) | ICIP 2018 |  |

---

### 4) 3D shape-to-sketch 3D形状-to-草图

| Paper | Source | Code/Project Link |
| --- | --- | --- |
| [DeepShapeSketch : Generating hand drawing sketches from 3D objects](https://shizhezhou.github.io/projects/DeepFreeHandSke2019/deepFreehandSke2019.pdf) | IJCNN 2019 |  |

---


### 5) Sketch(pixelwise)-to-sketch(vector) 草图(像素图)-to-草图(矢量)


This means translating a pixelwise sketch into a sequential sketch imitating human's drawing order. The appearance of the sequential sketch is exactly the **same** as the pixelwise one.


| Paper | Source | Code/Project Link |
| --- | --- | --- |
| [Animated Construction of Line Drawings](http://sweb.cityu.edu.hk/hongbofu/projects/animatedConstructionOfLineDrawings_SiggA11/animatedConstructionOfLineDrawings_SiggA11.pdf) | SIGGRAPH ASIA 2011 | [[Project]](http://sweb.cityu.edu.hk/hongbofu/projects/animatedConstructionOfLineDrawings_SiggA11/) [[code]](http://sweb.cityu.edu.hk/hongbofu/projects/animatedConstructionOfLineDrawings_SiggA11/Viewer_src.zip) [[Demo]](http://sweb.cityu.edu.hk/hongbofu/projects/animatedConstructionOfLineDrawings_SiggA11/Viewer.zip) |

---

### 6) Art-to-sketch  艺术画作-to-草图

Here we list sketch synthesis based on other image types, like Manga and line art.

- Hand drawn line art (a.k.a. Sketch Simplification)


| Paper | Source | Code/Project Link |
| --- | --- | --- |
| [Closure-aware Sketch Simplification](http://www.cse.cuhk.edu.hk/~ttwong/papers/sketch/sketch.pdf) | SIGGRAPH ASIA 2015 | [[Project]](https://www.cse.cuhk.edu.hk/~ttwong/papers/sketch/sketch.html) |
| [Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup](https://esslab.jp/publications/SimoSerraSIGGRAPH2016.pdf) | SIGGRAPH 2016 | [[Code]](https://github.com/bobbens/sketch_simplification) [[Project]](https://esslab.jp/~ess/en/research/sketch/) |
| [Mastering Sketching: Adversarial Augmentation for Structured Prediction](https://esslab.jp/~ess/publications/SimoSerraTOG2018.pdf) | SIGGRAPH 2018 | [[Code]](https://github.com/bobbens/sketch_simplification)  [[Project]](https://esslab.jp/~ess/en/research/sketch_master/) |
| [Real-Time Data-Driven Interactive Rough Sketch Inking](https://esslab.jp/~ess/publications/SimoSerraSIGGRAPH2018.pdf) | SIGGRAPH 2018 | [[Code]](https://github.com/bobbens/line_thinning) [[Project]](https://esslab.jp/~ess/en/research/inking/) |
| [StrokeAggregator: Consolidating Raw Sketches into Artist-Intended Curve Drawings](https://www.cs.ubc.ca/labs/imager/tr/2018/StrokeAggregator/StrokeAggregator_authorversion.pdf) | SIGGRAPH 2018 | [[Project]](https://www.cs.ubc.ca/labs/imager/tr/2018/StrokeAggregator/) |
| [Perceptual-aware Sketch Simplification Based on Integrated VGG Layers](https://ieeexplore.ieee.org/abstract/document/8771128/) | TVCG 2019 |  |

- Manga (Comics)

| Paper | Source | Code/Project Link |
| --- | --- | --- |
| [Deep extraction of manga structural lines](https://dl.acm.org/citation.cfm?id=3073675) | SIGGRAPH 2017 | [[Code]](https://github.com/ljsabc/MangaLineExtraction) |


Simplifying Rough Sketches using Deep Learning，作者为 Ashish Sinha。



利用LSGAN 和CRF进行线稿简化
http://hi.cs.waseda.ac.jp/~esimo/en/research/sketch/


data： getchu head https://github.com/ANIME305/Anime-GAN-tensorflow


### Learning to Simplify


   "Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup"
   Edgar Simo-Serra*, Satoshi Iizuka*, Kazuma Sasaki, Hiroshi Ishikawa (* equal contribution)
   ACM Transactions on Graphics (SIGGRAPH), 2016


   "Mastering Sketching: Adversarial Augmentation for Structured Prediction"
   Edgar Simo-Serra*, Satoshi Iizuka*, Hiroshi Ishikawa (* equal contribution)
   ACM Transactions on Graphics (TOG), 2018

https://github.com/bobbens/sketch_simplification/




二次元线稿 Anime-Girl-lineart-Generator
keevs https://www.deviantart.com/keevs/art/Anime-Girl-lineart-Generator-88708558












GAN生成油画效果： AI Portraits Ars https://aiportraits.com/


## CV & NLP

### StoryGAN

https://arxiv.org/pdf/1812.02784v2.pdf

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

