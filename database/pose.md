
姿势动作图像
HMDB_a large human motion database
Human Actionsand Scenes Dataset
Buffy Stickmen V3 人体轮廓识别图像数据
Human Pose Evaluator 人体轮廓识别图像数据
Buffy pose 人类姿势图像数据
VGG Human Pose Estimation 姿势图像标注数据

人类动作视频
Microsoft Research Action 人类动作视频数据
UCF50 Action Recognition 动作识别数据
UCF101 Action Recognition 动作识别数据
UT-Interaction 人类动作视频数据
UCF iPhone 运动中传感器数据
UCF YouTube 人类动作视频数据
UCF Sport 人类动作视频数据
UCF-ARG 人类动作视频数据
HMDB 人类动作视频
HOLLYWOOD2 人类行为动作视频数据
Recognition of human actions 动作视频数据
Motion Capture 动作捕捉视频数据
SBU Kinect Interaction 肢体动作视频数据


1. Weizmann 人体行为库

此数据库一共包括90段视频，这些视频分别是由9个人执行了10个不同的动作（bend, jack, jump, pjump, run, side, skip, walk, wave1,wave2）。视频的背景，视角以及摄像头都是静止的。而且该数据库提供标注好的前景轮廓视频。不过此数据库的正确率已经达到100%了，现在发文章基本没人用了呀。下载地址：http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html





2. KTH人体行为数据库
该数据库包括6类行为（walking, jogging, running, boxing, hand waving, hand clapping）,是由25个不同的人执行的，分别在四个场景下，一共有599段视频。背景相对静止，除了镜头的拉近拉远，摄像机的运动比较轻微。这个数据库是现在的benchmark，正确率需要达到95.5%以上才能够发文章。下载地址：http://www.nada.kth.se/cvap/actions/






3. INRIA XMAX多视角视频库
该数据库从五个视角获得，一共11个人执行14种行为。室内四个方向和头顶一共安装5个摄像头。另外背景和光照基本不变。下载地址：http://4drepository.inrialpes.fr/public/viewgroup/6






4. UCF Sports 数据库
该视频包括150段关于体育的视频，一共有13个动作。实验室采用留一交叉验证法。2011年cvpr有几篇都用这个数据库，正确率要达到87%才能发文章。下载地址：http://vision.eecs.ucf.edu/data.html




5. Hollywood 人体行为库
该数据库包括8类行为。这些都是电影中的片段。 下载地址：http://www.di.ens.fr/~laptev/actions/hollywood2/








6. Olympic sports dataset

该数据库有16种行为，783段视频。现在的正确率大约在75%左右。下载地址：http://vision.stanford.edu/Datasets/OlympicSports/






7. UIUC action dataset

这个数据库已经做到98%了，建议不要去做了。下载地址：http://vision.cs.uiuc.edu/projects/activity/


8. TRECVID视频库


.1    Weizmann
　　Weizmann[27]数据库包含了10个动作分别是走，跑，跳，飞跳，向一侧移动，单只手挥动，2只手挥动，单跳，2只手臂挥动起跳,每个动作有10个人执行。在这个视频集中，其背景是静止的，且前景提供了剪影信息。该数据集较为简单。
1.2    KTH
　　KTH[45]行人数据库包含了6种动作，分别为走，慢跑，跑挥手和鼓掌。每种动作由25个不同的人完成。每个人在完成这些动作时又是在4个不同的场景中完成的，4个场景分别为室外，室内，室外放大，室外且穿不同颜色的衣服。
网址链接2：http://www.nada.kth.se/cvap/actions/
1.3    PETS
　　PETS[51]，其全称为跟踪与监控性能评估会议，它的数据库是从现实生活中获取的，主要来源于直接从视频监控系统拍摄的视频，比如说超市的监控系统。从2000年以后，基本上每年都会组织召开这个会议。
1.4    YouTube
   YouTube包含11类动作，因为是现实生活中的视频数据，所以其背景比较复杂，这些种类的动作识别起来有些困难。
网址链接3：http://www.cs.ucf.edu/-liujg/YouTube\_Action\_dataset.html
1.5    UCF Sports
    UCF包含几个数据集，这里是指UCF的运动数据库,该视频数据包括了182个视频序列，共有9类动作。因为是现实生活中的视频数据，所以其背景比较复杂，这些种类的动作识别起来有些困难。
1.6    INRIA XMAS
　　INRIA XMAS数据库[53]是从5个视角拍摄的，室内的4个方向和头顶的1个方向。总共有11个人完成14种不同的动作，动作可以沿着任意方向执行。摄像机是静止的，环境的光照条件也基本不变。另外该数据集还提供有人体轮廓和体积元等信息。
1.7    Hollywood与Hollywood2
　　Hollywood电影的数据库包含有几个，其一Hollywood的视频集有8种动作，分别是接电话，下轿车，握手，拥抱，接吻，坐下，起立，站立。这些动作都是从电影中直接抽取的，由不同的演员在不同的环境下演的。其二Hollywood2在上面的基础上又增加了4个动作，骑车，吃饭，打架，跑。并且其训练集给出了电影的自动描述文本标注，另外一些是由人工标注的。因为有遮挡，移动摄像机，动态背景等因素，所以这个数据集非常有挑战。
1.8    HMDB51
    HMDB51包含51类动作，共有6849个视频，其中作者提供了70%用于训练，剩下的用于测试，320*240,。来自于YouTube、movie和其他的。
http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads 
1.9    UCF101
    UCF101包含101类动作，其中每一类由25个人做动作，每个人做4-7组，共有13320个视频，320*240.  

网址链接4：http://crcv.ucf.edu/data/UCF101/UCF101.rar

1.Weizman-包含10种动作(走路、快跑、向前跳、测试跳、弯腰、挥单手、原地跳、全身跳、单腿跳)，每个动作由10个人来掩饰，背景固定并且前景轮廓已经包含在数据库中，视角固定。

2.KTH-包含6种动作(走、跳、跑、击拳、挥手、拍手)，由25个人执行，分别在四个场景下，共599段视频，除了镜头的拉近拉远、摄像机的轻微运动外，背景相对静止。

3.UCF Sports-包含10类动作(跳水、打高尔夫、踢腿、举重、骑马、跑步、滑板、摇摆、侧摆、走路)，150个视频，从广播体育频道上收集到的，涵盖很广的场景类型和视角区域。

4.UCF50/UCF101-包含50/101类动作，6680段视频，都是网络上的视频，是真实场景下的。

5.Hollywood(2)-包含12类动作，2859个视频，从电影中截取的

6. HMDB-包含51类动作，6849个视频，由布朗大学SERRE实验室发布。

7.IXMAS Action-包含17类动作，是多角度行为数据，由8个视频角度的摄像机同时对一个行为进行拍摄。由英国Kingston大学发布.中科院自动化所发布了类似的数据集，CASIA.

8.UT-Interaction-监控场景下的数据库，识别从简单的单人行为上升到多人的交互行为。

9.MSR Action 3D/MSR Daily Activity 3D-利用Kinect传感器捕获除彩色图像以外的人体深度图像序列，利用Kinect采集的深度数据可获取较为精准的人体关节点骨架序列，这些序列为深入研究人体运动模式提供了很好的研究数据。

10.Northwestern-UCLA Multiview Action 3D-将深度、骨架和多视角数据融合在一起。

11.CUM Motion Capture-利用8个红外摄像头对41个标记点的人体进行重构，更为准确的估计出人体的骨架结构。


12.Activities of Daily Living(ADL)和First Person Social Interaction—用可穿戴设备采集的第一人称视角的行为数据库

1. The KTH Dataset(2004)
KTH数据集于2004 年的发布,是计算机视觉领域的一个里程碑。此后，许多新的数据库陆续发布。数据库包括在 4个不同场景下 25 个人完成的 6 类动作(walking, jogging, running,boxing, hand waving and hand clapping)共计 2391个视频样本，是当时拍摄的最大的人体动作数据库，它使得采用同样的输入数据对不同算法的性能作系统的评估成为可能。数据库的视频样本中包含了尺度变化、 衣着变化和光照变化，但其背景比较单一，相机也是固定的。下载地址：http://www.nada.kth.se/cvap/actions/
但是现在该数据集无法下载了(本人在这个网站中未能下载下)；发现数据堂上面有，有点贵。本人有一份，free。
 
2. The Weizmann Dataset(2005)
2005年，以色列 Weizmann institute 发布了Weizmann 数据库。数据库包含了 10个动作（bend, jack, jump, pjump, run,side, skip, walk, wave1,wave2），每个动作有 9 个不同的样本。视频的视角是固定的，背景相对简单，每一帧中只有 1 个人做动作。数据库中标定数据除了类别标记外还包括:前景的行为人剪影和用于背景抽取的背景序列。下载地址：http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html
KTH 和 Weizmann 数据库是行为识别领域引用率最高的数据库，对行为识别的研究起了较大的促进作用。当然，这两个数据库的局限性也是很明显的，由于背景比较简单，没有包含相机运动， 动作种类也较少，并且每段视频只有1个人在做单一的运动，这与真实的场景差别很大。
3. The IXMAS Dataset(2006)
该数据库为多视角数据库，该数据库从五个视角获得，室内四个方向和头顶一共安装5个摄像头，另外背景和光照基本不变。包含了11个人做14个动作，重复3次，这14个动作包括{check watch, cross arms, scratch head, sit down, get up, turnaround, walk, wave, punch, kick, point, pick up, throw (over head), throw (frombottom up)}。下载地址：http://4drepository.inrialpes.fr/public/viewgroup/6
 
4. The Hollywood Dataset(2008、2009)
Hollywood(2008年发布)、Hollywood-2数据库是由法国IRISA研究院发布的。早先发布的数据库基本上都是在受控的环境下拍摄的，所拍摄视频样本有限。2009年发布的Hollywood-2是Hollywood数据库的拓展版，包含了 12 个动作类别和 10个场景共3669个样本，所有样本均是从69部 Hollywood 电影中抽取出来的。视频样本中行为人的表情、姿态、穿着，以及相机运动、光照变化、遮挡、背景等变化很大，接近于真实场景下的情况，因而对于行为的分析识别极具挑战性。下载地址：http://www.di.ens.fr/~laptev/actions/hollywood2/
 
5. The UCF Dataset(2007-)
美国University of central Florida(UCF)自2007年以来发布的一系列数据库：1UCF sports action dataset(2008)，2UCF Youtube(2008)，3UCF50，4UCF101，引起了广泛关注。这些数据库样本来自从 BBC/ESPN的广播电视频道收集的各类运动样本、以及从互联网尤其是视频网站YouTube上下载而来的样本。其中UCF101是目前动作类别数、样本数最多的数据库之一，样本为13320段视频，类别数为101类。
下载地址：http://crcv.ucf.edu/data/
  
6. The Olympic sports dataset UCF sports action dataset(2010)
Stanford university2010年发布Olympic sports dataset UCF sports action dataset，包含了运动员的各类运动视频。视频都是从YouTube上下载的，包含有16个运动类别的50个视频，标记信息为运动类别。
下载地址：http://vision.stanford.edu/Datasets/OlympicSports/

7. The UT-interactiondataset
UT-interaction database是针对交互行为的数据库，包含有6类人人交互的动作(shaking hands, pointing, hugging,pushing, kicking and punching)总共 20 段样本,长度在 1 min 左右。
下载地址：http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html

8. The VideoWebdataset(2010)
California大学的VideoWebdatabase于 2010年发布，该数据库重点放在多人间的非语言交流的行为上(non-verbal communication),包含由最少4个至第4期视频序列中的行为识别研究进展多8个摄像机拍摄的长度为2.5min的视频。(未找到链接)


9. The HMDB51 dataset(2011)
Brown university大学发布的HMDB51于2011年发布，视频多数来源于电影，还有一部分来自公共数据库以及YouTube等网络视频库。数据库包含有6849段样本，分为51类，每类至少包含有101段样本。
下载地址：http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#dataset
 
除此之外还有：CMU MoBo DataSet(2001)、CMU MoCapDataSet(2006)、Human Eva(2009)、i3DPostMultiView(2009)

总体而言，数据库的动作类别越来越多，样本越来越多，数据库也更庞大，视频场景越来越复杂。较早的数据库，比如KTH，视频背景较简单，动作类别不多，相机固定，这使得现有的算法很容易达到饱和，不好区分算法的优劣。最近几年发布的数据库有如下几个趋势：背景嘈杂，视角不固定，甚至相机是运动的; 样本涉及到人人交互，人物交互；行为类别数较最早发布的数据库多了很多，总之是更接近于不受控的自然状态下的情景，这对于算法的鲁棒性提出了很大的挑战。

 