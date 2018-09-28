# 人脸相关数据库

有年龄数据的有： Adience  CACD2000  IMDB-WIKI 


名人数据的：CelebA MSRA-CFW CASIA-WebFace IMDB-WIKI CACD2000 MsCelebV1
 
人脸姿态： FERET CAS-PEAL UMIST CMU-PIE

<img src="https://github.com/weslynn/graphic-deep-neural-network/blob/master/otherpic/databasepic/face.png">

## 小数据集 0-10000


* ORL Database of Faces  (AT&T Dataset)由剑桥大学AT&T实验室创建,包含400张面部图像,(http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
ORL数据集是剑桥大学AT&T实验室收集的一个人脸数据集。包含了从1992.4到1994.4该实验室的成员。该数据集中图像分为40个不同的主题，每个主题包含10幅图像。对于其中的某些主题，图像是在不同的时间拍摄的。在关照，面部表情（张开眼睛，闭合眼睛，笑，非笑），面部细节（眼镜）等方面都变现出了差异性。所有图像都是以黑色均匀背景，并且从正面向上方向拍摄。

其中图片都是PGM格式，图像大小为92×102，包含256个灰色通道。

* Yale Face Database 15位志愿者的165张图片,包含光照,表情和姿态的变化. http://vision.ucsd.edu/content/yale-face-database

* Yale Face Database B  包含了10个人的5,850幅多姿态,多光照的图像.其中的姿态和光照变化的图像都是在严格控制的条件下采集的,主要用于光照和姿态问题的建模与分析.由于采集人数较少,该数据库的进一步应用受到了比较大的限制. http://vision.ucsd.edu/content/extended-yale-face-database-b-b

* MIT人脸数据库
MIT-CBCL Face Recognition Database， 10 subjects. (http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html)
MIT-CBCL Face Databases 2,429 faces, 4,548 non-faces (http://cbcl.mit.edu/software-datasets/FaceData2.html)

* CMU-MIT

 CMU-MIT是由卡内基梅隆大学和麻省理工学院一起收集的数据集，所有图片都是黑白的gif格式。里面包含511个闭合的人脸图像，其中130个是正面的人脸图像。Github下载链接为https://github.com/watersink/CMU-MIT

* GENKI

 GENKI数据集是由加利福尼亚大学的机器概念实验室收集。该数据集包含GENKI-R2009a,GENKI-4K,GENKI-SZSL三个部分。GENKI-R2009a包含11159个图像，GENKI-4K包含4000个图像，分为“笑”和“不笑”两种，每个图片的人脸的尺度大小，姿势，光照变化，头的转动等都不一样，专门用于做笑脸识别。GENKI-SZSL包含3500个图像，这些图像包括广泛的背景，光照条件，地理位置，个人身份和种族等，下载链接为http://mplab.ucsd.edu，如果进不去可以，同样可以去下面的github下载，链接https://github.com/watersink/GENKI

* FDDB（Face Detection Data Set and Benchmark）

  FDDB数据集主要用于约束人脸检测研究，该数据集选取野外环境中拍摄的2845个图像，从中选择5171个人脸图像。是一个被广泛使用的权威的人脸检测平台。下载链接为http://vis-www.cs.umass.edu/fddb/

* AFW（Annotated Faces in the Wild）

  AFW数据集是使用Flickr（雅虎旗下图片分享网站）图像建立的人脸图像库，包含205个图像，其中有473个标记的人脸。对于每一个人脸都包含一个长方形边界框，6个地标和相关的姿势角度。数据库虽然不大，额外的好处是作者给出了其2012 CVPR的论文和程序以及训练好的模型。下载链接为http://www.ics.uci.edu/~xzhu/face/

* MALF (Multi-Attribute Labelled Faces)

 MALF是为了细粒度的评估野外环境中人脸检测模型而设计的数据库。数据主要来源于Internet，包含5250个图像，11931个人脸。每一幅图像包含正方形边界框，俯仰、蜷缩等姿势等。该数据集忽略了小于20×20的人脸，大约838个人脸，占该数据集的7%。同时，该数据集还提供了性别，是否带眼镜，是否遮挡，是否是夸张的表情等信息。需要申请才可以得到官方的下载链接，链接为http://www.cbsr.ia.ac.cn/faceevaluation/

* MUCT Data Sets

  MUCT人脸数据库由3755个人脸图像组成，每个人脸图像有76个点的landmark，图片为jpg格式，地标文件包含csv,rda,shape三种格式。该图像库在种族、关照、年龄等方面表现出更大的多样性。下载链接为http://www.milbo.org/muct/

* IMM Data Sets

  IMM人脸数据库包括了240张人脸图片和240个asf格式文件，共40个人（7女33男），每人6张人脸图片，每张人脸图片被标记了58个特征点。所有人都未戴眼镜,下载链接为http://www2.imm.dtu.dk/~aam/datasets/datasets.html

## 中型数据集  10000-100000

* LFW：Labeled Faces in the Wild LFW（Labeled Faces in the Wild）

 LFW是一个用于研究无约束的人脸识别的标准数据库。该数据集包含了从网络收集的5k+人脸，13000张图像，每张图像都以被拍摄的人名命名。其中，有1680个人有两个或两个以上不同的照片。下载链接为http://vis-www.cs.umass.edu/lfw/index.html#download 

* FERET, FERET Database,14,051张多姿态,光照的灰度人脸图像 https://www.nist.gov/programs-projects/face-recognition-technology-feret
FERET, FERET Color Database https://www.nist.gov/itl/iad/image-group/color-feret-database

* CAS-PEAL 1040个人的30k+张人脸图像，主要包含姿态、表情、光照变化 (http://www.jdl.ac.cn/peal/index.html)

* CMU-PIE CMU Face Pose, Illumination, and expression_r(PIE) Database (http://www.ri.cmu.edu/projects/project_418.html)
由美国卡耐基梅隆大学创建,包含68位志愿者的41,368张多姿态,光照和表情的面部图像.其中的姿态和光照变化图像也是在严格控制的条件下采集的,

* Adience 包含2k+个人的26k+张人脸图像人脸性别，人脸年龄段(8组) 
该数据集来源为Flickr相册，由用户使用iPhone5或者其它智能手机设备拍摄，同时具有相应的公众许可。该数据集主要用于进行年龄和性别的未经过滤的面孔估计。同时，里面还进行了相应的landmark的标注。是做性别年龄估计和人脸对齐的一个数据集。图片包含2284个类别和26580张图片。下载链接为http://www.openu.ac.il/home/hassner/Adience/data.html#agegender

* FaceScrub. 非限制场景（100,100张，530人）. http://vintage.winklerbros.net/facescrub.html

* Pubfig 200个人的58k+人脸图像 http://www.cs.columbia.edu/CAVE/databases/pubfig/

* IJB-A (IARPA JanusBenchmark A)

  IJB-A是一个用于人脸检测和识别的数据库，包含24327个图像和49759个人脸。需要邮箱申请相应帐号才可以下载，下载链接为http://www.nist.gov/itl/iad/ig/ijba_request.cfm

亚洲数据集

* PF01 由韩国浦项科技大学创建,包含103人的1,751张不同光照,姿态,表情的面部图像,志愿者以韩国人为主

* KFDB人脸数据库 ，包含了1,000人,共52,000幅多姿态,多光照,多表情的面部图像,其中姿态和光照变化的图像，是在严格控制的条件下采集的.志愿者以韩国人为主.

## 大型数据集 

目前人脸识别领域常用的人脸数据库主要有:

* WebFaces,Caltech， 10k+人，约500K张图片，非限制场景， http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/#Download

* CelebA，Multimedia Laboratory The Chinese University of Hong Kong 汤晓鸥，10K 名人，202K 脸部图像，每个图像40余标注属性 http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
该数据集为香港中文大学汤晓鸥老师组开源的数据集，主要包含了5个关键点，40个属性值等，包含了202599张图片，图片都是高清的名人图片，可以用于人脸检测，5点训练，人脸头部姿势的训练等。
5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young


* MSRA-CFW,MSRA,202792 张, 1583人 Data Set of Celebrity Faces on the Web http://research.microsoft.com/en-us/projects/msra-cfw/CASIA 

* MsCelebV1 　MSR IRC是目前世界上规模最大、水平最高的图像识别赛事之一，由MSRA（微软亚洲研究院）图像分析、大数据挖掘研究组组长张磊发起，ms_celeb_1m就是这个比赛的数据集 10M images for 100K celebrities 

https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/MSCeleb-1M-a.pdf

https://www.msceleb.org/

https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/

* CASIA-WebFace,李子青 Center for Biometrics and Security Research， 500k图片，10k个人 http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
该数据集为中科院自动化所，李子青老师组开源的数据集，包含了10575个人，一共494414张图片，其中有3个人和lfw中的一样。该数据集主要用于人脸识别。图像都是著名电影中crop而出的，每个图片的大小都是250×250，每个类下面都有3张以上的图片，非常适合做人脸识别的训练。需要邮箱申请 


* MegaFace，华盛顿大学百万人脸MegaFace数据集 
MegaFace资料集包含一百万张图片，代表690000个独特的人。所有数据都是华盛顿大学从Flickr（雅虎旗下图片分享网站）组织收集的。这是第一个在一百万规模级别的面部识别算法测试基准。 现有脸部识别系统仍难以准确识别超过百万的数据量。为了比较现有公开脸部识别算法的准确度，华盛顿大学在去年年底开展了一个名为“MegaFace Challenge”的公开竞赛。这个项目旨在研究当数据库规模提升数个量级时，现有的脸部识别系统能否维持可靠的准确率。需要邮箱申请才可以下载，下载链接为http://megaface.cs.washington.edu/dataset/download.html

* IMDB-WIKI 20k+个名人的460k+张图片 和维基百科62k+张图片, 总共： 523k+张图片，名人年龄、性别 https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

* CACD2000 2k名人160k张人脸图片 http://bcsiriuschen.github.io/CARC/

* AFLW（Annotated Facial Landmarks in the Wild）

  AFLW人脸数据库是一个包括多姿态、多视角的大规模人脸数据库，而且每个人脸都被标注了21个特征点。此数据库信息量非常大，包括了各种姿态、表情、光照、种族等因素影响的图片。AFLW人脸数据库大约包括25000万已手工标注的人脸图片，其中59%为女性，41%为男性，大部分的图片都是彩色，只有少部分是灰色图片。该数据库非常适合用于人脸识别、人脸检测、人脸对齐等方面的研究，具有很高的研究价值。图像如下图所示，需要申请帐号才可以下载，下载链接为http://lrs.icg.tugraz.at/research/aflw/


* WIDER FACE
   WIDER FACE是香港中文大学的一个提供更广泛人脸数据的人脸检测基准数据集，由YangShuo， Luo Ping ，Loy ，Chen Change ，Tang Xiaoou收集。它包含32203个图像和393703个人脸图像，在尺度，姿势，闭塞，表达，装扮，关照等方面表现出了大的变化。WIDER FACE是基于61个事件类别组织的，对于每一个事件类别，选取其中的40%作为训练集，10%用于交叉验证（cross validation），50%作为测试集。和PASCAL VOC数据集一样，该数据集也采用相同的指标。和MALF和Caltech数据集一样，对于测试图像并没有提供相应的背景边界框。图像如下图所示，下载链接为http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

* 300W

  300W数据集是由AFLW，AFW，Helen，IBUG，LFPW，LFW等数据集组成的数据库。需要邮箱申请才可以下载，下载链接为http://ibug.doc.ic.ac.uk/resources/300-W/

* VGG Face dataset
该数据集包含了2622个不同的人，官网提供了每个图片的URL，下载链接：http://www.robots.ox.ac.uk/~vgg/data/vgg_face/

VGGFace2 Dataset 
3.31 million images of 9131 subjects (identities), with an average of 362.6 images for each subject
该数据集包含331万个9131个受试者的图像，每个受试者平均有362.6个图像。图像从谷歌图像搜索下载，并在姿势，年龄，照明，种族和职业（例如演员，运动员，政治家）有很大的变化。收集数据集时考虑了三个目标：（i）同时拥有大量身份以及每个身份的大量图像; （ii）涵盖各种姿势，年龄和种族; （iii）尽量减少标签噪音
http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ https://arxiv.org/pdf/1710.08092.pdf



## 3D数据库


UMDFace

MTFL(TCDCN所用)

[300W-3D]: The fitted 3D Morphable Model (3DMM) parameters of 300W samples.

[300W-3D-Face]: The fitted 3D mesh, which is needed if you do not have Basel Face Model (BFM)


[300W-LP]: The synthesized large-pose face images from 300W. 300W standardises multiple alignment
databases with 68 landmarks, including AFW, LFPW, HELEN, IBUG and XM2VTS.

[AFLW2000-3D]: The fitted 3D faces of the first 2000 AFLW samples, which can be used for 3D face alignment evaluation.



## 视频数据库

YouTube Face   1,595个人 3,425段视频非限制场景 
该数据集主要用于非约束条件下的视频中人脸识别，姿势判定等。该数据集包含1595个不同人的3425个视频，平均每个人的类别包含了2.15个视频，每个类别最少包含48帧，最多包含6070帧，平均包含181.3帧。
下载链接：
http://www.cslab.openu.ac.il/agas/，
http://www.cslab.openu.ac.il/download/，

如果没有效果，可以尝试filezilla下载，
server:agas.openu.ac.il

Path: /v/data9/cslab/wolftau/

filezilla模式设置为"Transfer mode"




##　AR人脸数据

* AR人脸数据库
由西班牙巴塞罗那计算机视觉中心建立,包含116人的3,288幅图像.采集环境中的摄像机参数,光照环境,摄像机距离等都是严格控制的.
(http://cobweb.ecn.purdue.edu/~aleix/aleix_face_DB.html)
* BANCA人脸数据库
该数据库是欧洲BANCA计划的一部分,包含了208人,每人12幅不同时间段的面部图像.

* MPI人脸数据库
该人脸数据库包含了200人的头部3维结构数据和1,400幅多姿态的人脸图像.

* XM2VTS人脸数据库
包含了295人在4个不同时间段的图像和语音视频片断.在每个时间段,每人被记录了2个头部
旋转的视频片断和6个语音视频片断.此外,其中的293人的3维模型也可得到.

## 其他
闭眼数据集 Closed Eyes In The Wild (CEW)
http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html

## 人群密度估计数据库

* UCSD 

该数据集分为，UCSD Pedestrain ,people annotation，people counting三个部分，下载链接为：http://visal.cs.cityu.edu.hk/downloads/

 

* PETS

该数据集包含S0，S1，S2，S3四个子集，S0为训练数据，S1为行人计数和密度估计，S2为行人跟踪，S3为流分析和事件识别，下载链接为：http://www.cvg.reading.ac.uk/PETS2009/a.html





* Mall dataset

下载链接为：http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html



* ShanghaiTech_Crowd_Counting_Dataset:

该数据集为上海科技大学研究生张营营，在其2016cvpr中所使用的数据集，数据集分为A,B两部分，每一部分都分好了train和test，下载链接为:https://pan.baidu.com/s/1gfyNBTh



* UCF_CC_50：
数据来源于FLICKR，使用数据集，发表文章 需要引用 
Haroon Idrees, Imran Saleemi, Cody Seibert, Mubarak Shah, Multi-Source Multi-Scale Counting in Extremely Dense Crowd Images, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

http://crcv.ucf.edu/data/crowd_counting.php



## 人头检测数据库

HollywoodHeads dataset

该数据集为从视频中截取的图片，包含224740张jpeg格式图片，还有xml格式的标注，和VOC的标注方式一样。下载链接为:http://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip






## 表情识别常用数据库　　

CK+，JAFFE，KDEF和Pain expressions form PICS）建立了一个面部表情数据库含有七个基本情绪状态和2062个不平衡样本。


1： The Japanese Female FacialExpression (JAFFE) Database
http://www.kasrl.org/jaffe.html
这个数据库比较小，而且是一个比较老的数据库了，早在1998年就发布了。该数据库是由10位日本女性在实验环境下根据指示做出各种表情，再由照相机拍摄获取的人脸表情图像。整个数据库一共有213张图像，10个人，全部都是女性，每个人做出7种表情，这7种表情分别是： sad, happy, angry, disgust,surprise, fear, neutral. 每个人为一组，每一组都含有7种表情，每种表情大概有3,4张样图。这样每组大概20张样图，目前在这个数据库上的识别率已经很高了，不管是person independent 或者是person dependent。识别率都很高。这个数据库可以用来熟悉人脸表情识别的一些基础知识，包括特征提取，分类等。

2： The Extended Cohn-Kanade Dataset(CK+)
http://www.pitt.edu/~emotion/ck-spread.htm

这个数据库是在 Cohn-Kanade Dataset 的基础上扩展来的，发布于2010年。这个数据库比起JAFFE 要大的多。而且也可以免费获取，包含表情的label和Action Units 的label。 
这个数据库包括123个subjects, 593 个 image sequence，每个image sequence的最后一张 Frame 都有action units 的label，而在这593个image sequence中，有327个sequence 有 emotion的 label。这个数据库是人脸表情识别中比较流行的一个数据库，很多文章都会用到这个数据做测试。具体介绍可以参考如下文献： 
P.Lucey, J. F. Cohn, T.Kanade, J. Saragih, Z. Ambadar, and I. Matthews, “TheExtended Cohn-KanadeDataset (CK+)_ A complete dataset for action unit andemotion-specifiedexpression,” inComputer Vision andPattern RecognitionWorkshops (CVPRW), 2010 IEEE Computer Society Conference on,2010, pp. 94-101.

3: Pain expressions
599张图 http://pics.psych.stir.ac.uk/zips/pain.zip

4： GEMEP-FERA 2011
http://gemep-db.sspnet.eu/

这个数据是2011年，IEEE 的 Automatic Face & GestureRecognition and Workshops (FG 2011), 2011 IEEE International Conference on 上提供的一个数据库，如果要获取这个数据库，需要签署一个assignment，而且只有学术界可以免费使用。 
这个数据库拥有的表情图很多，但是subjects 很少。具体介绍可以参考如下文献： 
M.F. Valstar, M. Mehu, B.Jiang, M. Pantic, and K. Scherer, “Meta-Analysis ofthe First FacialExpression Recognition Challenge,”Systems,Man, andCybernetics, Part B: Cybernetics, IEEE Transactions on, vol. 42,pp. 966-979,2012.

5: GENKI-4K
http://mplab.ucsd.edu/wordpress/?page_id=398

这个数据库专门用于做笑脸识别的，整个数据库一共有4000张图片，分为“笑”和“不笑”两种，图片中的人脸并不是posed，而是spontaneous的，每个图片的人脸的尺度大小也不一样，而且还有姿势，光照的变化，以及头的转动，相对于posed facialexpression, 这个数据库的难度要更大。 
详细信息可以参考如下文献： 
WhitehillJ, Littlewort G, Fasel I, et al. Toward practical smile detection[J]. PatternAnalysis and Machine Intelligence, IEEE Transactions on, 2009, 31(11):2106-2111.


6： AFEW 

AFEW_4_0_EmotiW_2014
http://cs.anu.edu.au/few/emotiw2014.html

这个数据库用作ACM 2014 ICMI TheSecond Emotion Recognition In The Wild Challenge and Workshop。去年已经进行了第一届的竞赛，这个数据库提供原始的video clips, 都截取自一些电影，这些clips 都有明显的表情，这个数据库与前面的数据库的不同之处在于，这些表情图像是 in the wild, not inthe lab. 所以一个比较困难的地方在于人脸的检测与提取。详细信息可以参考如下文献： 
A.Dhall, R. Goecke, J. Joshi,M. Wagner, and T. Gedeon, “Emotion RecognitionIn The Wild Challenge2013,” inProceedings of the 15thACM on Internationalconference on multimodal interaction, 2013, pp.509-516.



AFEW6.0
  该数据集中每个视频被标记为一种情绪，一共七种情绪：anger，disgust，fear，happiness，sad，surprise和neural，该数据集共有1750个短视频，其中训练集774个，验证集383个，测试集593个。


7：The UNBC-McMaster shoulder painexpression archive database
http://www.pitt.edu/~emotion/um-spread.htm

这个数据库用于做pain的表情识别，目前发布的数据库包含25个subject，200个video sequences，每个video sequence的长度不一，从几十帧图到几百帧图，每张图都有66 个facial landmarks，pain的intensity (0-15)，以及facial action units的编号，每个video sequence也有一个整体pain的 OPI。 
详细信息可以参考如下文献： 
Lucy,P., Cohn, J. F., Prkachin, K. M., Solomon, P., & Matthrews, I. (2011).Painful data: The UNBC-McMaster Shoulder Pain Expression Archive Database. IEEEInternational Conference on Automatic Face and Gesture Recognition (FG2011).


8：Bimodal Face and Body GestureDatabase （FABO）
http://www.eecs.qmul.ac.uk/~hatice/fabo.html
这个数据库与其它数据库相比，多了gesture的信息，目前利用multimodality 做情感计算的尝试取得很多进展，利用语音信息，人脸表情，body language等做emotion analysis正在受到越来越多的关注，这个数据库就是从facial expression与body gesture两个方面考虑人的情感，不过这个数据库的ground truth label 很繁琐，具体的信息可以参考：
GunesH, Piccardi M. A bimodal face and body gesture database for automatic analysisof human nonverbal affective behavior[C]//Pattern Recognition, 2006. ICPR 2006.18th International Conference on. IEEE, 2006, 1: 1148-1153.

表情识别比赛

　　1）The Third Emotion Recognition in the Wild Challenge 
　　这是ACM International Conference on Multimodal Interaction (ICMI 2015)举办的一个表情识别的竞赛，每年都举办，感兴趣的可以参加一下。 
　　https://cs.anu.edu.au/few/emotiw2015.html 
　　https://sites.google.com/site/emotiw2016/ 


    2 EmotiW 2016


参考：
https://blog.csdn.net/qq_14845119/article/details/51913171











other info：

* Annotated Database (Hand, Meat, LV Cardiac, IMM face) (http://www2.imm.dtu.dk/~aam/)

* BioID Face Database (http://www.bioid.com/downloads/facedb/index.php)
* Caltech Computational Vision Group Archive (Cars, Motorcycles, Airplanes, Faces, Leaves, Background) (http://www.vision.caltech.edu/html-files/archive.html)
* Carnegie Mellon Image Database (motion, stereo, face, car, ...) (http://vasc.ri.cmu.edu/idb/)

* CMU Cohn-Kanade AU-Coded Facial Expression Database (http://www.ri.cmu.edu/projects/project_421.html
* CMU Face Detection Databases (http://www.ri.cmu.edu/projects/project_419.html)
* CMU Face Expression Database (http://amp.ece.cmu.edu/projects/FaceAuthentication/download.htm)
* 
* CMU VASC Image Database (motion, road sequences, stereo, CIL’s stereo data with ground truth, JISCT, face, face expressions, car) (http://www.ius.cs.cmu.edu/idb/)
* Content-based Image Retrieval Database (http://www.cs.washington.edu/research/imagedatabase/groundtruth/)
* Face Video Database of the Max Planck Institute for Biological Cybernetics (http://vdb.kyb.tuebingen.mpg.de/)

* Georgia Tech Face Database (http://www.anefian.com/face_reco.htm)
* German Fingerspelling Database (http://www.anefian.com/face_reco.htm )
* Indian Face Database (http:// www.cs.umass.edu/~vidit/IndianFaceDatabase)
* MIT-CBCL Car Database (http://cbcl.mit.edu/software-datasets/CarData.html)
* 
* MIT-CBCL Pedestrian Database (http://cbcl.mit.edu/software-datasets/PedestrianData.html)
* MIT-CBCL Street Scenes Database (http://cbcl.mit.edu/software-datasets/streetscenes/)
* NIST/Equinox Visible and Infrared Face Image Database (http://www.equinoxsensors.com/products/HID.html)
* NIST Fingerprint Data at Columbia (Link)
*  
* Rutgers Skin Texture Database (http://www.caip.rutgers.edu/rutgers_texture/)
* The Japanese Female Facial expression_r(JAFFE) Database (http://www.kasrl.org/jaffe.html
* The Ohio State University SAMPL Image Database (3D, still, motion) (http://sampl.ece.ohio-state.edu/database.htm)
* The University of Oulu Physics-Based Face Database (http://www.ee.oulu.fi/research/imag/color/pbfd.html)
* UMIST Face Database (http://images.ee.umist.ac.uk/danny/database.html)
* USF Range Image Data (with ground truth) (http://marathon.csee.usf.edu/range/DataBase.html)
* Usenix Face Database (hundreds of images, several formats) (Link)
* UCI Machine Learning Repository (http://www1.ics.uci.edu/~mlearn/MLSummary.html)
* USC-SIPI Image Database (collection of digitized images) (http://sipi.usc.edu/services/database/Database.html)
* UCD VALID Database (multimodal for still face, audio, and video) (http://ee.ucd.ie/validdb/)
* UCD Color Face Image (UCFI) Database for Face Detection (http://ee.ucd.ie/~prag/)
* UCL M2VTS Multimodal Face Database (http://www.tele.ucl.ac.be/PROJECTS/M2VTS/m2fdb.html)
* Vision Image Archive at UMass (sequences, stereo, medical, indoor, outlook, road, underwater, aerial, satellite, space and more) (http://sipi.usc.edu/database/)
* Where can I find Lenna and other images? (http://www.faqs.org/faqs/compression-faq/part1/section-30.html)
* (http://cvc.yale.edu/projects/yalefaces/yalefaces.html)
Bao Face 人脸数据
DC-IGN 论文人脸数据
300 Face in Wild 图像数据

CMU Frontal Face Images

NIST Mugshot Identification Database Faces in the Wild 人脸数据
