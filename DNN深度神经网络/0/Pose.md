
## Pose 姿态识别

<img src="https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/blob/master/map/Pose.png?raw=true">



### motion transfer

也有叫video retargeting、video reenactment、human animation等，这些定义不完全等价，是一个比较综合的生成式任务。

早期的研究需要输入一段目标人物的视频，用来学习人物特异的外观encode-decoder，在合成的时候再利用decoder基于目标人体pose来输出整段视频，有点像deepfake的做法（https://github.com/deepfakes/faceswap）

早期代表性工作是
1) Everybody Dance Now
直接使用关键点condition 进行驱动.

https://github.com/carolineec/EverybodyDanceNow

Dance Dance Generation: Motion Transfer for Internet Videos


2) vid2vid


19年同年相关作品还有：
![image](https://user-images.githubusercontent.com/22336020/192982150-b34cbdd9-e6d8-4b3c-b87a-aa7a33e2cec2.png)

https://motionretargeting2d.github.io/

https://github.com/LONG-9621/Human_motion-Imatation

后面有一些follow-up基于人体prior进一步改进encode和decode的效果（如TransMoMo）。这类工作由于需要输入目标人物的整段视频（而且需要pose覆盖相对充分），且对于每个目标都需要重新训练，限制了这一类方法的应用场景。<img src="https://pic2.zhimg.com/50/v2-e070142b70beaa1ebcdba6f9756621fa_720w.jpg?source=1940ef5c" data-rawwidth="1818" data-rawheight="846" data-size="normal" data-caption="" data-default-watermark-src="https://picx.zhimg.com/50/v2-e43389609d33c565e27fad403db6dc66_720w.jpg?source=1940ef5c" class="origin_image zh-lightbox-thumb" width="1818" data-original="https://pic1.zhimg.com/v2-e070142b70beaa1ebcdba6f9756621fa_r.jpg?source=1940ef5c"/>


Z. Yang*, W. Zhu*, W. Wu*, C. Qian, Q. Zhou, B. Zhou, C. C. Loy. "TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. (* indicates equal contribution.)

https://arxiv.org/pdf/2003.14401.pdf
​
arxiv.org/pdf/2003.14401.pdf

![image](https://user-images.githubusercontent.com/22336020/192982213-2e92aea9-775e-4e1f-985c-47f987b7ad36.png)

https://github.com/yzhq97/transmomo.pytorch

知乎 https://zhuanlan.zhihu.com/p/145135732


后期的工作开始做基于单图输入的motion transfer，是一种single-shot的setting。由于无法对单个用户直接训练一个模型，这块一般有两个思路（近期关注得比较少，欢迎补充其他思路）：由于单图的外观representation不好学，为了降低合成难度，会基于输入的pose和目标的pose去估计一个warp参数，然后将目标人物的图像warp到目标pose，最后可能再加个网络来refine一下合成的结果。至于如何warp，不同文章用了不同的方式，比如2D flow、3D flow等。

代表性工作有MonkeyNet、Liquid Warping GAN等等。<img src="https://picx.zhimg.com/50/v2-d0e166409ccd752dd8056ad335885dcc_720w.jpg?source=1940ef5c" data-rawwidth="1826" data-rawheight="690" data-size="normal" data-caption="" data-default-watermark-src="https://pic4.zhimg.com/50/v2-cc99ae0f453448da7a3efddc351c3fac_720w.jpg?source=1940ef5c" class="origin_image zh-lightbox-thumb" width="1826" data-original="https://pic3.zhimg.com/v2-d0e166409ccd752dd8056ad335885dcc_r.jpg?source=1940ef5c"/>

2. NVIDIA沿着vid2vid也做了一个general的few-shot vid2vid工作。和上一类方法不同，他们基于输入的单图去预测视频decoder的一些网络参数，从而将外观特异的信息用到了合成网络中。这也是他们之前一个叫SPADE的工作的延续。<img src="https://pic1.zhimg.com/50/v2-6f791e66e119270f0a4da611612d9013_720w.jpg?source=1940ef5c" data-rawwidth="1318" data-rawheight="1100" data-size="normal" data-caption="" data-default-watermark-src="https://pic1.zhimg.com/50/v2-663a0ee45bc80c67ead6bfbda82cae1b_720w.jpg?source=1940ef5c" class="origin_image zh-lightbox-thumb" width="1318" data-original="https://pic1.zhimg.com/v2-6f791e66e119270f0a4da611612d9013_r.jpg?source=1940ef5c"/>此外，这个领域一般都会用到很多human的prior来强化网络的学习，最简单的是human pose（人体关键点）、human parsing（部件分割），复杂的有dense pose、SMPL、3D pose等。为了让脸部效果更好，很多工作还会单独对脸部进行合成提高质量。

https://nvlabs.github.io/few-shot-vid2vid/

![image](https://user-images.githubusercontent.com/22336020/192982451-f569ce6f-386f-450e-82e5-35390753c37a.png)




2. First Order Motion Model for Image Animation (FOMM)
-  [github](https://github.com/AliaksandrSiarohin/first-order-model)
- 使用自监督预测移动点+泰勒级数表示稀疏运动预测稠密运动
















部分参考：
https://www.zhihu.com/question/481811366 下系列回答
