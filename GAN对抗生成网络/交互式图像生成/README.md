# GANs Applications in CV

## 3.3 交互式图像生成
## iGAN
基于DCGAN，Adobe公司构建了一套图像编辑操作，能使得经过这些操作以后，图像依旧在“真实图像流形”上，因此编辑后的图像更接近真实图像。

具体来说，iGAN的流程包括以下几个步骤：

1 将原始图像投影到低维的隐向量空间

2 将隐向量作为输入，利用GAN重构图像

3 利用画笔工具对重构的图像进行修改（颜色、形状等）

4 将等量的结构、色彩等修改应用到原始图像上。


值得一提的是，作者提出G需为保距映射的限制，这使得整个过程的大部分操作可以转换为求解优化问题，整个修改过程近乎实时。

Theano 版本：https://github.com/junyanz/iGAN


[24] Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman and Alexei A. Efros. “Generative Visual Manipulation on the Natural Image Manifold”, ECCV 2016.

## GANpaint

GAN dissection

MIT、香港中文大学、IBM等学校/机构的David Bau、朱俊彦、Joshua B.Tenenbaum、周博磊

http://gandissect.res.ibm.com/ganpaint.html?project=churchoutdoor&layer=layer4
 


### GauGAN（SPADE） Nvidia


你画一幅涂鸦，用颜色区分每一块对应着什么物体，它就能照着你的大作，合成以假乱真的真实世界效果图。
通过语义布局进行图像的生成 Segmentation mask，算法是源于Pix2PixHD，生成自然的图像。

数据来源是成对的，通过自然场景的图像进行分割，就可以得到分割图像的布局，组成了对应的图像对。
但是区别在于，之前的Pix2PixHD，场景都很规律，如室内，街景，可以使用BN，但是后来发现Pix2PixHD在COCO这些无限制的数据集训练结果很差。如果整张天空或者整张的草地，则计算通过BN后，结果很相似，这样合成会出现问题。于是修改了BN，这种方法称为空间自适应归一化合成法SPADE。将原来的label信息代入到原来BN公式中的γ和β

Semantic Image Synthesis with Spatially-Adaptive Normalization--CVPR 2019。

这篇论文的一作，照例还是实习生。另外几位作者来自英伟达和MIT，CycleGAN的创造者朱俊彦是四作。

在基于语义合成图像这个领域里，这可是目前效果最强的方法。

![gaugan](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/ganpic/gaugan.jpg)


paper：https://arxiv.org/abs/1903.07291

GitHub：https://github.com/NVlabs/SPADE

项目地点：https://nvlabs.github.io/SPADE/

https://nvlabs.github.io/SPADE/demo.html
https://nvlabs.github.io/SPADE/

https://36kr.com/p/5187136

demo:

https://nvlabs.github.io/SPADE/demo.html
http://nvidia-research-mingyuliu.com/gaugan

https://www.nvidia.com/en-us/research/ai-playground/

---------------------------

## AutoDraw

AutoDraw能将机器学习与你信手涂鸦创建的图形配对，帮助你绘制完整而比较漂亮的图形。
