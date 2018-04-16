
* MNIST手写数字图像库, Yann LeCun, Corinna Cortes and Chris Burges，这个库有共70,000张图片，每张图片的大小是28*28，共包含10类从数字0到9的手写字符图像，在图像检索里一般是直接使用它的灰度像素作为特征，特征维度为784维。
* CIFAR-10 and CIFAR-100 datasets，这个数据库包含10类图像，每类6k，图像分辨率是32*32。另外还有一个CIFAR-100。如果嫌CIFAR-100小，还有一个更大的Tiny Images 
Dataset，上面CIFAR-10和CIFAR-100都是从这个库里筛选出来的，这个库有80M图片。

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.CIFAR-10(用作10类图像分类)此数据集包含了60,000张32x32的RGB图像,总共有10类图像,大约6000张图像/类,50,000张做训练,10,000张做测试!此数据集有三个版本的数据可供下载: Python版本(163MB), MATLAB版本(175MB), 二值版本(162MB)!CIFAR-100(用作100类图像分类)这个数据集和CIFAR-10相比,它具有100个类,大约600张/


* Caltech101和Caltech256，从后面的数据可以看出它们分别有多少类了。虽然这两个库用于做图像分类用得很多，不过也非常适合做CBIR，前面给的两个数据库由于图像大小尺寸较小，在检索可视化的时候显示的效果不是很好。所以我比较推荐用Caltech256和Caltech101，Caltech256有接近30k的图片，用这个发发论文完全是没什么问题的。如果要做几百万的实际应用，那得另寻数据库。
	* 
INRIA Holidays，也是一个在做CBIR时用的很多的数据库，图像检索的论文里很多都会用这个数据库。该数据集是Herve Jegou研究所经常度假时拍的图片（风景为主），一共1491张图，500张query（一张图一个group）和对应着991张相关图像，已提取了128维的SIFT点4455091个，visual dictionaries来自Flickr60K，链接。
	* 
Oxford Buildings Dataset，5k Dataset images，有5062张图片，是牛津大学VGG小组公布的，在基于词汇树做检索的论文里面，这个数据库出现的频率极高，下载链接。
	* 
Oxford Paris，The Paris Dataset，oxford的VGG组从Flickr搜集了6412张巴黎旅游图片，包括Eiffel Tower等。
	* 
201Books and CTurin180，The CTurin180 and 201Books Data Sets，2011.5，Telecom Italia提供于Compact Descriptors for Visual Search，该数据集包括：Nokia E7拍摄的201本书的封面图片（多视角拍摄，各6张），共1.3GB； Turin市180个建筑的视频图像，拍摄的camera有Galaxy S、iPhone 3、Canon A410、Canon S5 IS，共2.7GB。
	* 
Stanford Mobile Visual Search，Stanford Mobile Visual Search Dataset，2011.2，stanford提供，包括8种场景，如CD封面、油画等，每组相关图片都是采自不同相机（手机），所有场景共500张图，链接；随后又发布了一个patch数据集，Compact Descriptors for Visual Search Patches Dataset，校对了相同patch。
	* 
UKBench，UKBench database，2006.7，Henrik Stewénius在他CVPR06文章中提供的数据集，图像都为640*480，每个group有4张图，文件接近2GB，提供visual words，链接。
	* 
MIR-FLICKR，MIR-FLICKR-1M，2010，1M张Flickr上的图片，也提供25K子集下载，链接。
此外，还有COREL，NUS-WIDE等。一般做图像检索验证算法，前面给出的四个数据库应该是足够了的。
	* 
ImageCLEFmed医学图像数据库，见Online Multiple Kernel Similarity Learning for Visual Search。这个Project page里，有5个图像库，分别是Indoor、Caltech256、Corel (5000)、ImageCLEF (Med)、Oxford Buildings，在主页上不仅可以下到图像库，而且作者还提供了已经提取好的特征。



VOC2007 与 VOC2012   此数据集可以用于图像分类,目标检测,图像分割!!!数据集下载镜像网站: http://pjreddie.com/projects/pascal-voc-dataset-mirror/VOC2012: Train/Validation Data(1.9GB),Test Data(1.8GB),主页: http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOC2007: Train/Validation Data(439MB),Test Data(431MB),主页: http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/M 

80 million tiny images dataset这个数据集包含了79,302,017张32x32的RGB图像,下载时包含了5个文件,网站上也提供了示例代码教你如何加载这些数据!1. Image binary (227GB)2. Metadata binary (57GB)3. Gist binary (114GB)4. Index data (7MB)5. Matlab Tiny Images toolbox (150kB)Caltech_101(用作101类图像分类)这个数据集包含了101类的图像,每类大约有40~800张图像,大部分是50张/类,在2003年由lifeifei收集,每张图像的大小大约是300x200.数据集下载: 101_ObjectCategories.tar.gz(131MB)Caltech_256(用作256类图像分类)此数据集和Caltech_101相似,包含了30,607张图像,数据集下载: 256_ObjectCategroies.tar(1.2GB)ImagenetIMAGENET Large Scale Visual Recognition Challenge(ILSVRC)




图像数据
综合图像
Visual Genome 图像数据
Visual7w 图像数据
COCO 图像数据
SUFR 图像数据
ILSVRC 2014 训练数据（ImageNet的一部分）
PASCAL Visual Object Classes 2012 图像数据
PASCAL Visual Object Classes 2011 图像数据
PASCAL Visual Object Classes 2010 图像数据
80 Million Tiny Image 图像数据【数据太大仅有介绍】
ImageNet【数据太大仅有介绍】
Google Open Images【数据太大仅有介绍】
Imagenet 小尺寸图像数据集
Yahoo Flickr 照片和视频数据集

场景图像
Street Scences 图像数据
Places2 场景图像数据
UCF GoogleStreet View 图像数据
SUN 场景图像数据
The Celebrity inPlaces 图像数据

Web标签图像
HARRISON 社交标签图像
NUS-WIDE 标签图像
Visual Synset 标签图像
Animals WithAttributes 标签图像

人形轮廓图像
MPII Human Shape人体轮廓数据
Biwi Kinect Head Pose 头部姿势数据
上半身人像数据 INRIA Person 数据集



特定一类事物图像
著名的猫图像标注数据
Caltech-UCSDBirds200 鸟类图像数据
Stanford Car 汽车图像数据
Cars 汽车图像数据
MIT Cars 汽车图像数据
Stanford Cars 汽车图像数据
Food-101 美食图像数据
17_Category_Flower 图像数据
102_Category_Flower 图像数据
UCI Folio Leaf 图像数据
Labeled Fishes in the Wild 鱼类图像
美国 Yelp 点评网站酒店照片
CMU-Oxford Sculpture 塑像雕像图像
Oxford-IIIT Pet 宠物图像数据
Nature Conservancy Fisheries Monitoring 过度捕捞监控图像数据【Kaggle数据】
Stanford Dogs Dataset 数据集
辛普森一家卡通形象图像【Kaggle竞赛】
Fashion-MNIST 时尚服饰图像数据



材质纹理图像
CURET 纹理材质图像数据
ETHZ Synthesizability 纹理图像数据
KTH-TIPS 纹理材质图像数据
Describable Textures 纹理图像数据



物体分类图像
COIL-20 图像数据
COIL-100 图像数据
Caltech-101 图像数据
Caltech-256 图像数据
CIFAR-10 图像数据
CIFAR-100 图像数据
STL-10 图像数据
LabelMe_12_50k图像数据
NORB v1.0 图像数据
NEC Toy Animal 图像数据
iCubWorld 图像分类数据
Multi-class 图像分类数据
GRAZ 图像分类数据




指纹识别
NIST FIGS 指纹识别数据
NIST Supplemental Fingerprint Card Data (SFCD) 指纹识别数据
NIST Plain and Rolled Images from Paired Fingerprint Cards in 500 pixels per inch 指纹识别数据
NIST Plain and Rolled Images from Paired Fingerprint Cards 1000 pixels per inch 指纹识别数据

其它图像数据
Visual Question Answering V1.0 图像数据
Visual Question Answering V2.0 图像数据


视频数据
综合视频
DAVIS_Densely Annotated Video Segmentation 数据
YouTube-8M 视频数据集【数据太大仅有介绍】
YouTube 网站视频备份【数据太大仅有介绍】


目标检测视频
UCSD Pedestrian 行人视频数据
Caltech Pedestrian 行人视频数据
ETH 行人视频数据
INRIA 行人视频数据
TudBrussels 行人视频数据
Daimler 行人视频数据
ALOV++ 物体追踪视频数据

密集人群视频
Crowd Counting 高密度人群图像
Crowd Segmentation 高密度人群视频数据
Tracking in High Density Crowds 高密度人群视频

其它视频
Fire Detection 视频数据



音频数据
综合音频
Google Audioset 音频数据【数据太大仅有介绍】

语音识别
Sinhala TTS 英语语音识别
TIMIT 美式英语语音识别数据
LibriSpeech ASR corpus 语音数据
Room Impulse Response and Noise 语音数据
ALFFA 非洲语音数据
THUYG-20 维吾尔语语音数据
AMI Corpus 语音识别



处理后的科研和竞赛数据
NIPS 2003 属性选择竞赛数据
台湾大学林智仁教授处理为 LibSVM 格式的分类建模数据
Large-scale 分类建模数据
几个UCI 中 large-scale 分类建模数据
Social Computing Data Repository 社交网络数据
猫和狗分类识别竞赛数据【Kaggle竞赛】
DSTL 卫星图像识别竞赛数据【Kaggle竞赛】
根据手机应用软件使用行为预测用户性别年龄竞赛数据【Kaggle竞赛】
人脸关键点标定竞赛数据【Kaggle竞赛】
Kaggle竞赛数据合辑（部分竞赛数据）
UCI多分类组合出的二分类数据集
UCI经典二分类数据集
场景图像分类竞赛数据【ChallengerAI 竞赛】
人体骨骼关键点检测竞赛数据【ChallengerAI 竞赛】
图像中文表述竞赛数据【ChallengerAI 竞赛】
英文同声传译竞赛数据【ChallengerAI 竞赛】
中英文本翻译竞赛数据【ChallengerAI 竞赛】
虚拟股票趋势预测【ChallengerAI 竞赛数据】
机器视觉推理实验数据
BigMM 2015 竞赛验证数据集
KONECT 网络图结构和网络科学数据合辑
