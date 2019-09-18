一些paper

# Video Understanding

Video Understanding 一般分为三类：
1 Classification/ Activity Classification
	从相应数据集来看，比如sport1m，hmdb51，activity-net等，大都属于运动/行为类型的类别，所以，video classification和action recognition有很多的overlap(当然 action recognition 从人的动作监督信息可以分为 skeleton based 和non-skeleton based)

	研究人员大多从两方面去展开: 
	1.c3d，3d卷积同时学习每帧图像上的行为信息和短时间序列上的行为信息。
	2.two stream，帧stream+光流stream+ensemble的方法，方别学习帧图像上的行为信息和短时序列上的行为信息。
	cvpr17有文章把二者结合也做了文章(其实也没比TSN好)


    更深一步是video的piexl semantic classfication, 比如video semantic segmentation,也叫作video scene pharsing,(视频场景解析)。 常用的数据集有camVid 和cityscape，就是把图片的语义分割拓展到了video.，这个可以用到自动驾驶领域，所以Facebook，腾讯都有做这个任务，iccv17上有一系列他们的工作，貌似idea有一些比较像的地方。估计是英雄所见略同吧。还有一个任务是video object segmentation, cvpr16上提了一个数据集DAVIS。貌似还办了个比赛 DAVIS: Densely Annotated VIdeo Segmentation。

2 Activity detection

	Video Segmentation
	不论是比较传统的光流还是基于anchor的proposal network或者是其他的一些方法，由于长视频本身的复杂性，在时间序列上将视频分割成更加单一的clip再进行下一步的处理都是极有必要的。

	Activity Detection
	在segmentation的基础上，最为常见的一个应用是Temporal Activity Detection，即在时间序列上检测事件的起始时间，主要用于untrimmed video data，数据集比如activity-net有相应的标记信息，即事件的开始和结束的时间点信息。Activity在时间序列上的检测就像Object在图像的空间维度上的检测，也是一个非常重要的任务。

	推荐：CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos


	更深一步就是 Event detection，比如trecvid 每年举办的比赛里的multimedia event detection(MED)，在几十万各种各样的视频中，包含有指定(复杂)事件的一些视频，需要用算法找出最可能包含这些事件的视频。这其中根据训练数据的数量分为100x，10x，0x，前两者属于non-zero-shot learning，所以一般会使用提取特征+训练分类器的方法去做，做工作的地方会主要在特征提取的环节。后者0x是没有训练数据的，需要从事件的语义信息中去找事件组成子元素(concept怎么翻译 )，然后通过易得的其他数据中训练这些子元素，再去视频中找他们。
	这个任务很难，特别是一些很复杂却又不好和相近类别分开的事件(比如med里有个事件是”做木制手艺活”)


3 Video Caption
	Video Caption是Image Caption的升级版，针对一段视频生成一句（段）描述性语句，不仅要求视频在时间序列上的合理分割，还要求对每段clip的视觉内容做caption之后合理整合，难度更大。
	从早期的cnn+rnn的结构，即帧图像特征提取+建立文字序列结构到现在各种新方法的探索，现在已经能做的很不错了。当然仍有很大的空间。类似的topic还有很多，比如video2doc(一段文字描述，如果没记错的话，有这么一篇文章)，video2vec，加上seq2seq，又可以转到新的表达形式。

	推荐：Hierarchical Recurrent Neural Encoder for Video Representation with Application to Captioning

	Video Question Answering


4 Video QA

Video QA是Image QA （也就是常说的VQA）的升级版，它可以看成是一种检索，输入的Question就是检索的关键词和限定条件，但是同时又对Video Analysis有很高的要求。

如

IJCAI 17的Video Question Answering via Hierarchical Spatio-Temporal Attention Networks，这篇是比较典型的时空注意力机制的应用，由于视频的时空特性，多层注意力机制的强大表现力是可预见的。
SIGIR 17的Video Question Answering via Attribute-Augmented Attention Network Learning，这篇文章引入了Attribute概念，加强了frame-level的Video Representation，也同样采用了Temporal Attention的方法；
而MM 17的Video Question Answering via Gradually Refined Attention over Appearance and Motion这篇文章则通过结合Appearance和Motion两个通道的不同Attention来加强问题和视频表达间的联系，再通过RNN cell的变体AMU来对问题进行处理。
现有的Video QA基本都还逃不开spatio-temporal model。


已读： Video Question Answering via Gradually Refined Attention over Appearance and Motion

摘自[zhihu](https://www.zhihu.com/question/64021205)

此外 图像 标签 层级相关 之前李佳 在2010年发过一篇文章  Building and Using a Semantivisual Image Hierarchy 

3D： 
主题：Image Processing，Pose Estimation
2018 CVPR Best Paper

Total Capture: A 3D Deformation Model for Tracking Faces, Hands, and Bodies

https://arxiv.org/pdf/1801.01615

同时进行脸、身体、手姿态估计，并融合到三维模型Frankenstein。之后使用Frankenstein和数据集训练有头发和衣服的Adam模型。

利用SMPL(A Skinned Multi-Person Linear Model)获得Body model，使用PCA model获得Face model，使用Artist rigged hand mesh获得Hand model,使用“bending matrix”将不同模型融合在一起。

使用OpenPose获得Pose

分别为关键点的误差、迭代最近点（Iterative Closest Point, ICP）的误差、连接约束误差、3D点云的噪声误差组成。

【迭代最近点】迭代最近点算法（ICP）是一种点云匹配算法。其思想是：通过旋转、平移使得两个点集之间的距离最小。ICP算法由Besl等人于1992年提出，文献可以参考：A Method for Registration of 3D Shapes，另外还可以参考：Least-Squares Fitting of Two 3-D Point Sets。前者使用的是四元数方法来求解旋转矩阵，而后者则是通过对协方差矩阵求解SVD来得到最终的旋转矩阵。
————————————————
[李艳宾](https://blog.csdn.net/linghugoolge/article/details/87942340)


美颜：
很多拍照软件，会帮忙调整人的五官，可能参考了siggraph2008这一篇：
Data-Driven Enhancement of Facial Attractiveness


图像检索：
综述 ： SIFT Meets CNN: A Decade Survey of Instance  Retrieval 2015 [pdf](https://arxiv.org/abs/1608.01807)
看了一下有翻译版，顺便链接上： [上](https://mp.weixin.qq.com/s?__biz=MzUyMjE2MTE0Mw==&mid=2247486346&idx=1&sn=139309de32ae1b72fc3ce5e81fd7811a&chksm=f9d15512cea6dc0449ba0ba223ab5b8fb790b8892d7dc755766475e7877b5dcf88e2cfe74863&scene=21#wechat_redirect) [下](https://www.imooc.com/article/33964)


videosearch - Large-scale video retrieval using image queries [link](https://ieeexplore.ieee.org/document/7851077/citations#citations)
https://github.com/andrefaraujo/videosearch


分布式训练：
A Hitchhiker's Guid on Distributed Training of Deep Neural Networks

2D->3D  景深变换

3D Ken Burns Effect from a Single Image
https://arxiv.org/abs/1909.05483