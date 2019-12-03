
# Recommendation 推荐系统

信息是怎样流通的？

在过去，人们的信息都是来自于一些中心化的信息，譬如报纸，书籍，广播，电视，门户网站。信息传播的渠道决定了很多东西，包含价值观的传递。内容传播秉承着“中心化分发，展示位有限、千人一面”的状态，信息传播的决策权始终握在编辑手中。优势在于，依赖人的专业知识完成了从海量内容到有限展示位置的过滤和初步筛选，具有较高的平均质量。

进入算法时代，如何针对个人或者群体，给出不一样的个性化推荐，达到更好的目的，成了大家追求的目标。源于沃尔玛的 啤酒和尿布一起摆放，就是经典的item-based 推荐算法。

最开始，算法还是沿用人们传统的思维,到2016年开始进入深度学习方法阶段：

以youtube为例：

第一阶段，基于User-Video图游历算法，2008年[1]。
在这个阶段，YouTube认为应该给用户推荐曾经观看过视频的同类视频，或者说拥有同一标签的视频。然而此时，YouTube的视频已是数千万量级，拥有标签的部分却非常小，所以如何有效的扩大视频标签，被其认为是推荐的核心问题。

解决方案的核心有两块：
一是基于用户共同观看记录构建的图结构（Video Co-View Graph）； 
二是基于此数据结构的算法，被称为吸附算法（Adsorption Algorithm）。

视频的共同观看关系构建的图，可以从两个角度观察，一是视频构成的图，一是视频-用户构成的图，“视频”图可以看成由“视频用户”图（图1）抽取出。而视频之间的边，可以是同时观看过两个视频的用户个数，或者是在同一个Session中被同时观看的次数，甚至可以将顺序也考虑于其中。

那么到底如何给视频扩大标签呢？标签可以看成是一个分类，所谓“近朱者赤，近墨者黑”，在图结构中，一个节点的信息与属性可以通过其周围的节点得到。“标签”也不例外。Adsorption
Algorithm的核心思想是，部分节点将拥有一些标签，每一次迭代，可以将标签传递给相邻的节点，如此不停迭代，直到标签稳定分布在节点中。


第二阶段，基于Video-Video图游历算法，2010年[2]。
在这个阶段，YouTube认为需要将用户观看过的视频的相似视频推荐给用户。而什么是相似视频？主要以用户行为对其进行界定，可以是：
1. 被一定量用户共同观看的视频；
2. 在同一个Session中经常被同时观看的视频；
3. 考虑顺序信息的，在同一个Session中经常被同时观看的视频。

如上这几种选择，信息的有效性逐渐更好，但数据则逐渐稀疏，YouTube更加偏好第二种方式。

第三阶段，基于搜索以及协同过滤，2014年[3]。
本文陈述了“相关视频”的优化方法，即用户在观看某一个视频时推荐的视频。但实质上是定义了一种相似或者相关视频的计算方式。而“相似对象”的定义是推荐的核心问题，有了不同的计算方法，也意味着有了新的推荐方法。
为什么要有一个新的“相关视频”计算方法呢？协同过滤是当时最好的方法，但其适用于有了一定用户观看记录的视频，但对于新视频以及长尾视频，并不能良好应用。



第四阶段，基于深度神经网络，2016年[4]。
本文呈现的推荐系统解决方案分为两个部分，一个是备选生成（Candidate Generation），其目标是初选结果，从海量数据中选择出符合其个人需求偏好的百级别数据。一个则是排序（Ranking），通过更加丰富的用户，视频乃至场景信息，对结果进行精细化排序，得到呈现给用户的备选。
在本文中，推荐系统的建模方式有了实质性不同，即将推荐系统定义为一个多分类器，其职责是确定某个用户，在某个场景与时间下，将从系统的视频中选择消费哪一个视频。

首先获取视频的Embedding描述，将视频的文本放入Embedding工具即可（例如Word2Vec，但TensorFlow自带）即可。构建用户的Embedding，则是通过训练而来。以SoftMax分类为最终优化对象，将用户观看视频的Embedding整合值，搜索记录，其它信息如年龄性别等作为特征。中间为数层ReLU。能利用除了用户行为外的其它信息，也是神经网络相对于普通MF类算法的优势。


[1]Video Suggestion and Discovery for YouTube: Taking Random
Walks Through the View Graph
[2] The YouTube Video Recommendation System
[3] Up Next: Retrieval Methods for Large Scale Related Video
Suggestion
[4] Deep Neural Networks for YouTube Recommendations


作者：王小科 [zhihu](https://www.zhihu.com/question/20829671/answer/205421638)


基于Tensorflow的推荐系统的开源框架openrec
基于pytorch的推荐系统的框架Spotlight



https://github.com/NVIDIA/DeepRecommender


1. 书籍
Recommender Systems: The Textbook (2016, Charu Aggarwal)
Recommender Systems Handbook 2nd Edition (2015, Francesco Ricci)
Recommender Systems Handbook 1st Edition (2011, Francesco Ricci)
Recommender Systems An Introduction (2011, Dietmar Jannach) slides
2. 会议
AAAI (AAAI Conference on Artificial Intelligence)
CIKM (ACM International Conference on Information and Knowledge Management)
CSCW (ACM Conference on Computer-Supported Cooperative Work & Social Computing)
ICDM (IEEE International Conference on Data Mining)
IJCAI (International Joint Conference on Artificial Intelligence)
ICLR (International Conference on Learning Representations)
ICML (International Conference on Machine Learning)
IUI (International Conference on Intelligent User Interfaces)
NIPS (Neural Information Processing Systems)
RecSys (ACM Conference on Recommender Systems)
SDM (SIAM International Conference on Data Mining)
SIGIR (ACM SIGIR Conference on Research and development in information retrieval)
SIGKDD (ACM SIGKDD International Conference on Knowledge discovery and data mining)
SIGMOD (ACM SIGMOD International Conference on Management of Data)
VLDB (International Conference on Very Large Databases)
WSDM (ACM International Conference on Web Search and Data Mining)
WWW (International World Wide Web Conferences)
3. 研究人员
George Karypis (University of Minnesota)
Joseph A. Konstan (University of Minnesota)
Philip S. Yu (University of Illinons at Chicago)
Charu Aggarwal (IBM T. J. Watson Research Center)
Martin Ester (Simon Fraser University)
Paul Resnick (University of Michigan)
Peter Brusilovsky (University of Pittsburgh)
Bamshad Mobasher (DePaul University)
Alexander Tuzhilin (New York University)
Yehuda Koren (Google)
Barry Smyth (University College Dublin)
Lior Rokach (Ben-Gurion University of the Negev)
Loren Terveen (University of Minnesota)
Chris Volinsky (AT&T Labs)
Ed H. Chi (Google AI)
Laks V.S. Lakshmanan (University of British Columbia)
Badrul Sarwar (LinkedIn)
Francesco Ricci (Free University of Bozen-Bolzano)
Robin Burke (University of Colorado, Boulder)
Brent Smith (Amazon)
Greg Linden (Amazon, Microsoft)
Hao Ma (Facebook AI)
Giovanni Semeraro (University of Bari Aldo Moro)
Dietmar Jannach (University of Klagenfurt)
4. 论文
Explainable Recommendation: A Survey and New Perspectives (2018, Yongfeng Zhang)
Deep Learning based Recommender System: A Survey and New Perspectives (2018, Shuai Zhang)
Collaborative Variational Autoencoder for Recommender Systems (2017, Xiaopeng Li)
Neural Collaborative Filtering (2017, Xiangnan He)
Deep Neural Networks for YouTube Recommendations (2016, Paul Covington)
Wide & Deep Learning for Recommender Systems (2016, Heng-Tze Cheng)
Collaborative Denoising Auto-Encoders for Top-N Recommender Systems (2016, Yao Wu)
AutoRec: Autoencoders Meet Collaborative Filtering (2015, Suvash Sedhain)
Collaborative Deep Learning for Recommender Systems (2015, Hao Wang)
Collaborative Filtering beyond the User-Item Matrix A Survey of the State of the Art and Future Challenges (2014, Yue Shi)
Deep content-based music recommendation (2013, Aaron van den Oord)
Time-aware Point-of-interest Recommendation (2013, Quan Yuan)
Location-based and Preference-Aware Recommendation Using Sparse Geo-Social Networking Data (2012, Jie Bao)
Context-Aware Recommender Systems for Learning: A Survey and Future Challenges (2012, Katrien Verbert)
Exploiting Geographical Influence for Collaborative Point-of-Interest Recommendation (2011, Mao Ye)
Recommender Systems with Social Regularization (2011, Hao Ma)
The YouTube Video Recommendation System (2010, James Davidson)
Matrix Factorization Techniques for Recommender Systems (2009, Yehuda Koren)
A Survey of Collaborative Filtering Techniques (2009, Xiaoyuan Su)
Collaborative Filtering with Temporal Dynamics (2009, Yehuda Koren)
Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (2008, Yehuda Koren)
Collaborative Filtering for Implicit Feedback Datasets (2008, Yifan Hu)
SoRec: social recommendation using probabilistic matrix factorization (2008, Hao Ma)
Flickr tag recommendation based on collective knowledge (2008, Borkur Sigurbjornsson)
Restricted Boltzmann machines for collaborative filtering (2007, Ruslan Salakhutdinov)
Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions(2005, Gediminas Adomavicius)
Evaluating collaborative filtering recommender systems (2004, Jonatan L. Herlocker)
Amazon.com Recommendations: Item-to-Item Collaborative Filtering (2003, Greg Linden)
Content-boosted collaborative filtering for improved recommendations (2002, Prem Melville)
Item-based collaborative filtering recommendation algorithms (2001, Badrul Sarwar)
Explaining collaborative filtering recommendations (2000, Jonatan L. Herlocker)
An algorithmic framework for performing collaborative filtering (1999, Jonathan L. Herlocker)
Empirical analysis of predictive algorithms for collaborative filtering (1998, John S. Breese)
Social information filtering: Algorithms for automating "word of mouth" (1995, Upendra Shardanand)
GroupLens: an open architecture for collaborative filtering of netnews (1994, Paul Resnick)
Using collaborative filtering to weave an information tapestry (1992, David Goldberg)
5. GitHub Repositories
List_of_Recommender_Systems (Software, Open Source, Academic, Benchmarking, Applications, Books)
Deep-Learning-for-Recommendation-Systems (Papers, Blogs, Worshops, Tutorials, Software)
RecommenderSystem-Paper (Papers, Tools, Frameworks)
RSPapers (Papers)
awesome-RecSys-papers (Papers)
DeepRec (Tensorflow Codes)
RecQ (Tensorflow Codes)
NeuRec (Tensorflow Codes)
Surprise (Python Library)
LightFM (Python Library)
Spotlight (Python Library)
python-recsys (Python Library)
TensorRec (Python Library)
CaseRecommender (Python Library)
recommenders (Jupyter Notebook Tutorial)
6. 有用站点
WikiCFP (Call For Papers of Conferences, Workshops and Journals - Recommender System)
Guide2Research (Top Computer Science Conferences)
PapersWithCode (Papers with Code - Recommender System)
7.新增
https://github.com/microsoft/recommenders（微软发布的推荐代码）
shenweichen/DeepCTR （这篇文章涉及到ctr预估的经典方法，也有一些实现，非常适合做推荐的朋友）
参考：
1.https://github.com/jihoo-kim/awesome-RecSys?fbclid=IwAR1m6OebmqO9mfLV1ta4OTihQc9Phw8WNS4zdr5IeT1X1OLWQvLk0Wz45f4