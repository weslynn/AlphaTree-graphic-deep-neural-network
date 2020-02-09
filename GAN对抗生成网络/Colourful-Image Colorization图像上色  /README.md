# GANs Applications in CV



## 3.5 图像上色  Colourful Image Colorization
在计算机图形学领域中，一直存在两种为图片上色的方向：数据驱动的自动上色和用户交互的引导上色。

- [Automatic Image Colorization](#1-automatic-image-colorization)
- [User Guided Image Colorization](#2-user-guided-image-colorization)
  - [Based on color strokes](#21-based-on-color-strokes)
  - [Based on reference color image](#22-based-on-reference-color-image)
  - [Based on color palette](#23-based-on-color-palette)
  - [Based on language(text)](#24-based-on-language-or-text)
- [Video Colorization](#3-video-colorization)
  - [Automatically](#31-automatically)
  - [Based on reference](#32-based-on-reference)



### 1. Automatic Image Colorization


| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Learning Large-Scale Automatic Image Colorization](http://openaccess.thecvf.com/content_iccv_2015/papers/Deshpande_Learning_Large-Scale_Automatic_ICCV_2015_paper.pdf) | ICCV 2015 | [[project]](http://vision.cs.illinois.edu/projects/lscolor/) [[code]](https://github.com/aditya12agd5/iccv15_lscolorization) |
| [Deep Colorization](http://openaccess.thecvf.com/content_iccv_2015/papers/Cheng_Deep_Colorization_ICCV_2015_paper.pdf) | ICCV 2015 |  |
| [Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668.pdf) | ECCV 2016 | [[project]](http://people.cs.uchicago.edu/~larsson/colorization/) [[code]](https://github.com/gustavla/autocolorize) |
| [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf) | ECCV 2016 | [[project]](http://richzhang.github.io/colorization/) [[code]](https://github.com/richzhang/colorization) |
| [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf) | SIGGRAPH 2016 | [[project]](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/) [[code]](https://github.com/satoshiiizuka/siggraph2016_colorization) |
| [Unsupervised Diverse Colorization via Generative Adversarial Networks](https://arxiv.org/pdf/1702.06674.pdf) | ECML-PKDD 2017 | [[code]](https://github.com/ccyyatnet/COLORGAN) |
| [Learning Diverse Image Colorization](http://openaccess.thecvf.com/content_cvpr_2017/papers/Deshpande_Learning_Diverse_Image_CVPR_2017_paper.pdf) | CVPR 2017 | [[code]](https://github.com/aditya12agd5/divcolor) |
| [Structural Consistency and Controllability for Diverse Colorization](http://openaccess.thecvf.com/content_ECCV_2018/papers/Safa_Messaoud_Structural_Consistency_and_ECCV_2018_paper.pdf) | ECCV 2018 |  |
| [Pixelated Semantic Colorization](https://arxiv.org/abs/1901.10889) | 1901.10889 |  |
| [Coloring With Limited Data: Few-Shot Colorization via Memory Augmented Networks](http://davian.korea.ac.kr/filemanager/wl/?id=BPD0GpKupqUgHTxRMpaaLbCDrNoEjVfu) | CVPR 2019 |  |



#### Colourful Image Colourization 
2016 ECCV 里加州大学伯克利分校的一篇文章介绍的方法。这个方法与之前方法的不同之处在于，它把照片上色看成是一个分类问题——预测三百多种颜色在图片每一个像素点上的概率分布。这种方法tackle了这个任务本身的不确定性，例如，当你看到一个黑白的苹果时，你可能会觉得它是红色的，但如果这个苹果是青色的，其实也并没有多少违和感。大家也可以到作者的网站网站来试用他们的demo。 
https://richzhang.github.io/colorization/

#### Colornet

Github 地址：https://github.com/pavelgonchar/colornet

###  2. User Guided Image Colorization

这种方法是由 Levin 等人在 2004 年开创的，用户通过彩色画笔在灰度图像中进行引导性上色，随后优化算法会生成符合用户逻辑的上色结果。这种方法可以保留人工上色的部分性质，因而经常会有绝佳的表现，但往往需要密集的用户交互次数（有时超过五十次）。随着技术进步，现在的交互次数慢慢减少。

#### 2.1 Based on color strokes

| Image Type | Paper | Source | Code/Project Link  |
| --- | --- | --- |--- |
| Manga | [Manga colorization](https://dl.acm.org/citation.cfm?id=1142017) | SIGGRAPH 2006 |  |
| Line art / Sketch | [Outline Colorization through Tandem Adversarial Networks](https://arxiv.org/abs/1704.08834) | 1704.08834 | [[Demo]](http://color.kvfrans.com/) [[code]](https://github.com/kvfrans/deepcolor) |
| Line art / Sketch | [Auto-painter: Cartoon Image Generation from Sketch by Using Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1705.01908.pdf) | 1705.01908 | [[code]](https://github.com/irfanICMLL/Auto_painter) |
| Natural Gray-Scale | [Real-Time User-Guided Image Colorization with Learned Deep Priors](https://arxiv.org/abs/1705.02999) | SIGGRAPH 2017 | [[project]](https://richzhang.github.io/ideepcolor/) [[code1]](https://github.com/junyanz/interactive-deep-colorization) [[code2]](https://github.com/richzhang/colorization-pytorch) |
| Sketch | [Scribbler: Controlling Deep Image Synthesis with Sketch and Color](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sangkloy_Scribbler_Controlling_Deep_CVPR_2017_paper.pdf) | CVPR 2017 |  |
| Natural Gray-Scale | [Interactive Deep Colorization Using Simultaneous Global and Local Inputs](https://ieeexplore.ieee.org/abstract/document/8683686) (also palette based) | ICASSP 2019 |  |


#### Real-Time User-Guided Image Colorization with Learned Deep Priors

UC Berkeley  SIGGRAPH 2017 
论文链接：https://arxiv.org/abs/1705.02999
Demo 和代码链接：https://richzhang.github.io/ideepcolor/


#### 2.2 Based on reference color image

| Image Type | Paper | Source | Code/Project Link  |
| --- | --- | --- |--- |
| Line art | Style2paints V1 : [Style Transfer for Anime Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN](https://arxiv.org/abs/1706.03319) | ACPR 2017 | [[Code]](https://github.com/lllyasviel/style2paints#style2paints-v1) |
| Manga | [Comicolorization: Semi-Automatic Manga Colorization](https://arxiv.org/pdf/1706.06759.pdf) (also palette based) | SIGGRAPH Asia 2017 | [[code]](https://github.com/DwangoMediaVillage/Comicolorization) |
| Sketch | [TextureGAN: Controlling Deep Image Synthesis with Texture Patches](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_TextureGAN_Controlling_Deep_CVPR_2018_paper.pdf) | CVPR 2018 | [[code]](https://github.com/janesjanes/Pytorch-TextureGAN) |
| Natural Gray-Scale | [Deep Exemplar-based Colorization](https://arxiv.org/pdf/1807.06587.pdf) | SIGGRAPH 2018 | [[code]](https://github.com/msracver/Deep-Exemplar-based-Colorization) |
| Natural Gray-Scale | [Example-Based Colourization Via Dense Encoding Pyramids](http://www.shengfenghe.com/uploads/1/5/1/3/15132160/cgf_13659_rev_ev.pdf) (also palette based) | Pacific Graphics 2018 | [[code]](https://github.com/chufengxiao/Example-based-Colorization-via-Dense-Encoding-pyramids) |
| Natural Gray-Scale | [A Superpixel-based Variational Model for Image Colorization](https://ieeexplore.ieee.org/abstract/document/8676327) | TVCG 2019 |  |
| Natural Gray-Scale | [Automatic Example-based Image Colourisation using Location-Aware Cross-Scale Matching](https://ieeexplore.ieee.org/abstract/document/8699109) | TIP 2019 |  |

#### 2.3 Based on color palette

| Image Type | Paper | Source | Code/Project Link  |
| --- | --- | --- |--- |
| Natural Image | [Palette-based Photo Recoloring](https://gfx.cs.princeton.edu/pubs/Chang_2015_PPR/chang2015-palette_small.pdf) | SIGGRAPH 2015 | [[project]](https://gfx.cs.princeton.edu/pubs/Chang_2015_PPR/index.php) |
| Manga | [Comicolorization: Semi-Automatic Manga Colorization](https://arxiv.org/pdf/1706.06759.pdf) (also reference based) | SIGGRAPH Asia 2017 | [[code]](https://github.com/DwangoMediaVillage/Comicolorization) |
| Natural Gray-Scale | [Coloring with Words: Guiding Image Colorization Through Text-based Palette Generation](https://arxiv.org/pdf/1804.04128.pdf) (also text based) | ECCV 2018 | [[code]](https://github.com/awesome-davian/Text2Colors/) |
| Natural Gray-Scale | [Example-Based Colourization Via Dense Encoding Pyramids](http://www.shengfenghe.com/uploads/1/5/1/3/15132160/cgf_13659_rev_ev.pdf) (also reference based) | Pacific Graphics 2018 | [[code]](https://github.com/chufengxiao/Example-based-Colorization-via-Dense-Encoding-pyramids) |
| Natural Gray-Scale | [Interactive Deep Colorization Using Simultaneous Global and Local Inputs](https://ieeexplore.ieee.org/abstract/document/8683686) (also strokes based) | ICASSP 2019 |  |

#### 2.4 Based on language or text

| Image Type | Paper | Source | Code/Project Link  |
| --- | --- | --- |--- |
| Natural Gray-Scale / Sketch | [Language-Based Image Editing with Recurrent Attentive Models](https://arxiv.org/pdf/1711.06288.pdf) | CVPR 2018 | [[code]](https://github.com/Jianbo-Lab/LBIE) |
| Natural Gray-Scale | [Coloring with Words: Guiding Image Colorization Through Text-based Palette Generation](https://arxiv.org/pdf/1804.04128.pdf) (also palette based) | ECCV 2018 | [[code]](https://github.com/awesome-davian/Text2Colors/) |
| Scene Sketch | [LUCSS: Language-based User-customized Colorization of Scene Sketches](https://arxiv.org/pdf/1808.10544.pdf) | 1808.10544 | [[code]](https://github.com/SketchyScene/LUCSS) |


### 3. Video Colorization

#### 3.1 Automatically

| Paper | Source | Code/Project Link  |
| --- | --- |--- |
| [Fully Automatic Video Colorization with Self-Regularization and Diversity](https://cqf.io/papers/Fully_Automatic_Video_Colorization_CVPR2019.pdf) | CVPR 2019 |  |


#### 3.2 Based on reference

| Paper | Source | Code/Project Link  |
| --- | --- |--- |
| [Switchable Temporal Propagation Network](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sifei_Liu_Switchable_Temporal_Propagation_ECCV_2018_paper.pdf) | ECCV 2018 |  |
| [Tracking Emerges by Colorizing Videos](http://openaccess.thecvf.com/content_ECCV_2018/papers/Carl_Vondrick_Self-supervised_Tracking_by_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/wbaek/tracking_via_colorization) |
| [Deep Exemplar-based Video Colorization]() | CVPR 2019 |  |



DeOldify: Colorizing and Restoring Old Images and Videos with Deep Learning

老照片上色, 人脸处理，港星老照片：“你我当年”

