
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
UKBench，UKBench database，2006.7，Henrik Stewénius在他CVPR06文章中提供的数据集，图像都为640 ×480，每个group有4张图，文件接近2GB，提供visual words，链接。
	* 
MIR-FLICKR，MIR-FLICKR-1M，2010，1M张Flickr上的图片，也提供25K子集下载，链接。
此外，还有COREL，NUS-WIDE等。一般做图像检索验证算法，前面给出的四个数据库应该是足够了的。
	* 
ImageCLEFmed医学图像数据库，见Online Multiple Kernel Similarity Learning for Visual Search。这个Project page里，有5个图像库，分别是Indoor、Caltech256、Corel (5000)、ImageCLEF (Med)、Oxford Buildings，在主页上不仅可以下到图像库，而且作者还提供了已经提取好的特征。



VOC2007 与 VOC2012   此数据集可以用于图像分类,目标检测,图像分割!!!数据集下载镜像网站: http://pjreddie.com/projects/pascal-voc-dataset-mirror/VOC2012: Train/Validation Data(1.9GB),Test Data(1.8GB),主页: http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOC2007: Train/Validation Data(439MB),Test Data(431MB),主页: http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/M 

80 million tiny images dataset这个数据集包含了79,302,017张32x32的RGB图像,下载时包含了5个文件,网站上也提供了示例代码教你如何加载这些数据!1. Image binary (227GB)2. Metadata binary (57GB)3. Gist binary (114GB)4. Index data (7MB)5. Matlab Tiny Images toolbox (150kB)Caltech_101(用作101类图像分类)这个数据集包含了101类的图像,每类大约有40~800张图像,大部分是50张/类,在2003年由lifeifei收集,每张图像的大小大约是300x200.数据集下载: 101_ObjectCategories.tar.gz(131MB)Caltech_256(用作256类图像分类)此数据集和Caltech_101相似,包含了30,607张图像,数据集下载: 256_ObjectCategroies.tar(1.2GB)ImagenetIMAGENET Large Scale Visual Recognition Challenge(ILSVRC)



YouTube-8M：标注为 4716 个不同类别的七百万个 YouTube 视频

YouTube-Bounding Boxes：含有 5 百万个边界框的 21 万个 YouTube 视频

Speech Commands Dataset：数千个人说的简短控制词汇

AudioSet：2 百万个 10 秒长的 YouTube 视频，标注为了 527 个不同的声音事件

AVA：5.7 万个短视频中标注了一共 32 万个动作标签

Open Images：标记为 6000 个分类的 9 百万张带有创意共享许可的图像

Open Images with Bounding Boxes：600 个不同类别的图像中带有 120 万个边界框


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
Deep Fashion http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html
Deep Fashion2 https://arxiv.org/pdf/1901.07973.pdf https://sites.google.com/view/cvcreative/home?authuser=0


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


接下来我们将介绍语义分割领域最近最受欢迎的大规模数据集。所有列出的数据集均包含像素级别或点级别的标签。这个列表将根据数据内在属性分为3个部分：2维的或平面的RGB数据集，2.5维或带有深度信息的RGB（RGB-D）数据集，以及纯体数据或3维数据集。表1给出了这些数据集的概览，收录了所有本文涉及的数据集并提供了一些有用信息如他们的被构建的目的、类数、数据格式以及训练集、验证集、测试集划分情况。


1 二维数据集
      自始至终，最关注的是二维图像。因此，二维数据集在所有类型中是最丰富的，这考虑到所有的包含二维表示如灰度或RGB图像的数据集。
      PASCAL视觉物体分类数据集（PASCAL-VOC）[27] (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) : 包括一个标注了的图像数据集和五个不同的竞赛：分类、检测、分割、动作分类、人物布局。分割的竞赛很有趣：他的目标是为测试集里的每幅图像的每个像素预测其所属的物体类别。有21个类，包括轮子、房子、动物以及其他的：飞机、自行车、船、公共汽车、轿车、摩托车、火车、瓶子、椅子、餐桌、盆栽、沙发、显示器（或电视）、鸟、猫、狗、马、绵羊、人。如果某像素不属于任何类，那么背景也会考虑作为其标签。该数据集被分为两个子集：训练集1464张图像以及验证集1449张图像。测试集在竞赛中是私密的。争议的说，这个数据集是目前最受欢迎的语义分割数据集，因此很多相关领域卓越的工作将其方法提交到该数据集的评估服务器上，在其测试集上测试其方法的性能。方法可以只用该数据集训练，也可以借助其他的信息。另外，其方法排行榜是公开的而且可以在线查询。

      PASCAL 上下文数据集（PASCAL Context） [28] （http://www.cs.stanford.edu/∼roozbeh/pascal-context/）：对于PASCAL-VOC 2010识别竞赛的扩展，包含了对所有训练图像的像素级别的标注。共有540个类，包括原有的20个类及由PASCAL VOC分割数据集得来的图片背景，分为三大类，分别是物体、材料以及混合物。虽然种类繁多，但是只有59个常见类是较有意义的。由于其类别服从一个幂律分布，其中有很多类对于整个数据集来说是非常稀疏的。就这点而言，包含这59类的子集常被选作真实类别来对该数据集进行研究，其他类别一律重标为背景。

      PASCAL 部分数据集（PASCAL Part）[29] （http://www.stat.ucla.edu/∼xianjie.chen/pascal part dataset/pascal part.html）：对于PASCAL-VOC 2010识别竞赛的扩展，超越了这次竞赛的任务要求而为图像中的每个物体的部分提供了一个像素级别的分割标注（或者当物体没有连续的部分的时候，至少是提供了一个轮廓的标注）。原来的PASCAL-VOC中的类被保留，但被细分了，如自行车被细分为后轮、链轮、前轮、手把、前灯、鞍座等。本数据集包含了PASCAL VOC的所有训练图像、验证图像以及9637张测试图像的标签。

      语义边界数据集（SBD）[30] （http://home.bharathh.info/home/sbd）：是PASCAL数据集的扩展，提供VOC中未标注图像的语义分割标注。提供PASCAL VOC 2011 数据集中11355张数据集的标注，这些标注除了有每个物体的边界信息外，还有类别级别及实例级别的信息。由于这些图像是从完整的PASCAL VOC竞赛中得到的，而不仅仅是其中的分割数据集，故训练集与验证集的划分是不同的。实际上，SBD有着其独特的训练集与验证集的划分方式，即训练集8498张，验证集2857张。由于其训练数据的增多，深度学习实践中常常用SBD数据集来取代PASCAL VOC数据集。

      微软常见物体环境数据集（Microsoft COCO） [31]：(http://mscoco.org/) 是另一个大规模的图像识别、分割、标注数据集。它可以用于多种竞赛，与本领域最相关的是检测部分，因为其一部分是致力于解决分割问题的。该竞赛包含了超过80个类别，提供了超过82783张训练图片，40504张验证图片，以及超过80000张测试图片。特别地，其测试集分为4个不同的子集各20000张：test-dev是用于额外的验证及调试，test-standard是默认的测试数据，用来与其他最优的方法进行对比，test-challenge是竞赛专用，提交到评估服务器上得出评估结果，test-reserve用于避免竞赛过程中的过拟合现象（当一个方法有嫌疑提交过多次或者有嫌疑使用测试数据训练时，其在该部分子集上的测试结果将会被拿来作比较）。由于其规模巨大，目前已非常常用，对领域发展很重要。实际上，该竞赛的结果每年都会在ECCV的研讨会上与ImageNet数据集的结果一起公布。

      图像与注释合成数据集（SYNTHIA）[32] （http://synthia-dataset.net/）是一个大规模的虚拟城市的真实感渲染图数据集，带有语义分割信息，是为了在自动驾驶或城市场景规划等研究领域中的场景理解而提出的。提供了11个类别物体（分别为空、天空、建筑、道路、人行道、栅栏、植被、杆、车、信号标志、行人、骑自行车的人）细粒度的像素级别的标注。包含从渲染的视频流中提取出的13407张训练图像，该数据集也以其多变性而著称，包括场景（城镇、城市、高速公路等）、物体、季节、天气等。

      城市风光数据集 [33] （https://www.cityscapes-dataset.com/）是一个大规模的关注于城市街道场景理解的数据集，提供了8种30个类别的语义级别、实例级别以及密集像素标注（包括平坦表面、人、车辆、建筑、物体、自然、天空、空）。该数据集包括约5000张精细标注的图片，20000张粗略标注的图片。数据是从50个城市中持续数月采集而来，涵盖不同的时间以及好的天气情况。开始起以视频形式存储，因此该数据集按照以下特点手动选出视频的帧：大量的动态物体，变化的场景布局以及变化的背景。

      CamVid数据集 [55,34] （http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/）是一个道路、驾驶场景理解数据集，开始是五个视频序列，来自一个安装在汽车仪表盘上的960x720分辨率的摄相机。这些序列中采样出了701个帧（其中4个序列在1fps处，1个序列在15fps处），这些静态图被手工标注上32个类别：空、建筑、墙、树、植被、栅栏、人行道、停车场、柱或杆、锥形交通标志、桥、标志、各种文本、信号灯、天空、……（还有很多）。值得注意的是，Sturgess等人[35]将数据集按照367-100-233的比例分为训练集、验证集、测试集，这种分法使用了部分类标签：建筑、树、天空、车辆、信号、道路、行人、栅栏、杆、人行道、骑行者。

      KITTI [56] 是用于移动机器人及自动驾驶研究的最受欢迎的数据集之一，包含了由多种形式的传感器得出的数小时的交通场景数据，包括高分辨率RGB、灰度立体摄像机以及三维激光扫描器。尽管很受欢迎，该数据集本身并没有包含真实语义分割标注，但是，众多的研究者手工地为该数据集的部分数据添加标注以满足其问题的需求。Alvarez等人[36,37]为道路检测竞赛中的323张图片生成了真实标注，包含三个类别：道路、垂直面和天空。Zhang等人[39]标注了252张图片，其中140张训练、112张测试，其选自追踪竞赛中的RGB和Velodyne扫描数据，共十个类。Ros等人[38]在视觉测距数据集中标注了170个训练图片和46个测试图片，共11个类。

      YouTube物体数据集 [57] 是从YouTube上采集的视频数据集，包含有PASCAL VOC中的10个类。该数据集不包含像素级别的标注，但是Jain等人[42]手动的标注了其126个序列的子集。其在这些序列中每10个帧选取一张图片生成器语义标签，总共10167张标注的帧，每帧480x360的分辨率。

      Adobe肖像分割数据集 [26] （http://xiaoyongshen.me/webpage portrait/index.html） 包含从Flickr中收集的800x600的肖像照片，主要是来自手机前置摄像头。该数据集包含1500张训练图片和300张预留的测试图片，这些图片均完全被二值化标注为人或背景。图片被半自动化的标注：首先在每幅图片上运行一个人脸检测器，将图片变为600x800的分辨率，然后，使用Photoshop快速选择工具将人脸手工标注。这个数据集意义重大，因为其专门适用于人脸前景的分割问题。

      上下文语料数据集（MINC）[43] 是用于对块进行分类以及对整个场景进行分割的数据集。该数据集提供了23个类的分割标注（文中有详细的各个类别的名称），包含7061张标注了的分割图片作为训练集，5000张的测试集和2500张的验证集。这些图片均来自OpenSurfaces数据集[58]，同时使用其他来源如Flickr或Houzz进行增强。因此，该数据集中的图像的分辨率是变化的，平均来看，图片的分辨率一般是800x500或500x800。

      密集标注的视频分割数据集（DAVIS）[44,45]（http://davischallenge.org/index.html）：该竞赛的目标是视频中的物体的分割，这个数据集由50个高清晰度的序列组成，选出4219帧用于训练，2023张用于验证。序列中的帧的分辨率是变化的，但是均被降采样为480p的。给出了四个不同类别的像素级别的标注，分别是人、动物、车辆、物体。该数据集的另一个特点是每个序列均有至少一个目标前景物体。另外，该数据集特意地较少不同的大动作物体的数量。对于那些确实有多个前景物体的场景，该数据集为每个物体提供了单独的真实标注，以此来支持实例分割。

      斯坦福背景数据集[40] （http://dags.stanford.edu/data/iccv09Data.tar.gz）包含了从现有公开数据集中采集的户外场景图片，包括LabelMe, MSRC, PASCAL VOC 和Geometric Context。该数据集有715张图片（320x240分辨率），至少包含一个前景物体，且有图像的水平位置信息。该数据集被以像素级别标注（水平位置、像素语义分类、像素几何分类以及图像区域），用来评估场景语义理解方法。

      SiftFlow [41]：包含2688张完全标注的图像，是LabelMe数据集[59]的子集。多数图像基于8种不同的户外场景，包括街道、高山、田地、沙滩、建筑等。图像是256x256的，分别属于33个语义类别。未标注的或者标为其他语义类别的像素被认为是空。

 
2 2.5维数据集
      随着廉价的扫描器的到来，带有深度信息的数据集开始出现并被广泛使用。本章，我们回顾最知名的2.5维数据集，其中包含了深度信息。
      NYUDv2数据集[46]（http://cs.nyu.edu/∼silberman/projects/indoor scene seg sup.html）包含1449张由微软Kinect设备捕获的室内的RGB-D图像。其给出密集的像素级别的标注（类别级别和实力级别的均有），训练集795张与测试集654张均有40个室内物体的类[60]，该数据集由于其刻画室内场景而格外重要，使得它可以用于某种家庭机器人的训练任务。但是，它相对于其他数据集规模较小，限制了其在深度网络中的应用。

      SUN3D数据集[47]（http://sun3d.cs.princeton.edu/）：与NYUDv2数据集相似，该数据集包含了一个大规模的RGB-D视频数据集，包含8个标注了的序列。每一帧均包含场景中物体的语义分割信息以及摄像机位态信息。该数据集还在扩充中，将会包含415个序列，在41座建筑中的254个空间中获取。另外，某些地方将会在一天中的多个时段被重复拍摄。

      SUNRGBD数据集[48]（http://rgbd.cs.princeton.edu/）由四个RGB-D传感器得来，包含10000张RGB-D图像，尺寸与PASCAL VOC一致。该数据集包含了NYU depth v2 [46], Berkeley B3DO [61], 以及SUN3D [47]数据集中的图像，整个数据集均为密集标注，包括多边形、带方向的边界框以及三维空间，适合于场景理解任务。

      物体分割数据集（OSD）[62]（http://www.acin.tuwien.ac.at/?id=289）该数据集用来处理未知物体的分割问题，甚至是在部分遮挡的情况下进行处理。该数据集有111个实例，提供了深度信息与颜色信息，每张图均进行了像素级别的标注，以此来评估物体分割方法。但是，该数据集并没有区分各个类，使其退化为一个二值化的数据集，包含物体与非物体两个类。

      RGB-D物体数据集[49] （http://rgbd-dataset.cs.washington.edu/）该数据集由视频序列构成，有300个常见的室内物体，分为51个类，使用WordNet hypernym-hyponym关系进行分类。该数据集使用Kinect型三维摄像机进行摄制，640x480RGB图像，深度信息30赫兹。对每一帧，数据集提供了RGB-D及深度信息，这其中包含了物体、位置及像素级别的标注。另外，每个物体放在旋转的桌面上以得出360度的视频序列。对于验证过程，其提供了22个标注的自然室内场景的包含物体的视频序列。

 
3 三维数据集
      纯粹的三维数据集是稀缺的，通常可以提供CAD网格或者其他的体元表示如点云等。为分割问题获取三维数据集是困难的，因此很少有深度学习方法可以处理这种数据。也因此，三维数据集目前还不是很受欢迎。尽管如此，我们还是介绍目前出现的相关数据集来解决现有的问题。
      ShapeNet部分数据集[50]（http://cs.stanford.edu/ericyi/project page/part annotation/）是ShapeNet[63]数据集的子集，关注于细粒度的三维物体分割。包含取自元数据及16个类的31693个网格，每个形状类被标注为二到五个部分，整个数据集共有50个物体部分，也就是说，物体的每个部分比如飞机的机翼、机身、机尾、发动机等都被标注了。真实标注按照被网格分割开的点呈现。

      斯坦福2D-3D-S数据集[51]（http://buildingparser.stanford.edu）是一个多模态、大规模室内空间数据集，是斯坦福三维语义分析工作[64]的扩展。提供了多个模态：二维RGB，2.5维添加深度信息的图片、三维网格和点云，均提供分割标注信息。该数据集有70496张高分辨率的RGB图像（1080x1080分辨率），以及其对应的深度图谱、表面法线、网格以及点云，军事带有像素级别及点级别的语义标注信息。这些数据取自6个室内区域，分别来自三个不同的教育与办公建筑。共有271个房间，大约7亿个点，被标以13个类。

      三维网格分割基准数据集[52]（http://segeval.cs.princeton.edu/）该基准数据集有380个网格，被分为19个类。每个网格手动的被分割为不同的功能区域，主要目标是提供对于人们如何分配网格功能的一个概率分布。

      悉尼城市物体数据集[53]（http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml）该数据集包含多个常见的城市道路物体，由Velodyne HDK-64E LIDAR扫描得到，共有631个独立扫描的点云，由车辆、行人、标志、树木等类别组成。有趣的是，除了正常的扫描之外，还提供了全景360度的扫描标注。

      大规模点云分类基准数据集[54]（http://www.semantic3d.net/）该基准数据集提供手工标注的三维点云，面向自然与城市场景。该数据集在形成点云时保留了细节与密度等信息，训练集和测试集各包含15个大规模的点云，其规模达到超过十亿个标注点的级别。

 

 [参考](https://blog.csdn.net/mieleizhi0522/article/details/82902359)