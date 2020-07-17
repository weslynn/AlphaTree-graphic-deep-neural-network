
### 推荐系统 (Recommendation)


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


[zhihu](https://www.zhihu.com/question/20829671/answer/205421638)


推荐系统的指标：


基于Tensorflow的推荐系统的开源框架openrec
基于pytorch的推荐系统的框架Spotlight

阿里 xdeeplearning的推荐系统
https://github.com/alibaba/x-deeplearning

随着硬件计算能力发展带动深度学习的进步，预估领域的算法也逐渐的从传统的CTR（Click Through-Rate）预估模型迁移到深度CTR预估模型。 这些模型都可以归结为Embedding&MLP的范式：首先通过embedding layer将大规模的稀疏特征投影为低维连续的embedding vector， 然后将这些向量concatate后输入到一个全连接网络中，计算其最终的预估目标。相较于传统的方法， 这样的做法利用了深度学习模型更强的model capacity自动的学习特征之间以及特征与目标的关系， 减少了传统模型需要人工经验和实验验证的特征筛选设计阶段。


### DNN
最经典的范式是Embedding&MLP(代表为Youtube DNN)。这种范式中，user ID、用户历史交互item ID等高维category特征都被映射成为低维的embedding，MLP只能接受固定长度的输入，而用户历史上交互过的item数目是不固定，Embedding&MLP范式的应对方法是把用户交互过的item embedding求均值(进行average pooling操作)，然后把这个固定长度的均值作为user representation的一部分。
 图一 

图示为电商场景下Embedding&MLP的范式图示，用户交互过的N个商品，每个商品都可以求得一个embedding，对N个embedding进行pooling后，再与其他固定长度的特征concat，作为MLP的输入。
当对很多个兴趣(历史交互item)的embedding求均值时，有些长尾兴趣可能被湮没、新求的均值兴趣可能会有不可预知的偏移、原先embedding的维度可能不足以表达分散、多样的兴趣。


### DIN && MIND
1）背景与动机
为了解决表达用户多样兴趣的问题，阿里分别提出了DIN(Deep Interest Network KDD2018)和MIND(Multi-Interest Network with Dynamic Routing)两种深度网络，针对电商场景分别在推荐的排序阶段和召回阶段建模表达用户的多样兴趣。

•	DIN引入了attention机制，通过一个兴趣激活模块(Activation Unit)，用预估目标Candidate ADs的信息去激活用户的历史点击商品，以此提取用户与当前预估目标相关的兴趣。同一个用户与不同的item进行预测时，DIN会产生不同的用户embedding，具体来说，当预测某个item时，计算出该item与用户历史交互item的“匹配度”，权重高的历史行为表明这部分兴趣和当前广告相关，权重低的则是和广告无关的”兴趣噪声“。用这个匹配度作为权重对用户历史交互item做加权平均得到用户的兴趣embedding,作为当前预估目标ADs相关的兴趣状态表达.之后用这个兴趣embedding与用户静态特征和上下文相关特征以及ad相关的特征拼接起来组成所谓label-aware的用户embedding,输入到后续的多层DNN网络，最后预测得到用户对当前目标ADs的点击概率。

•	而MIND使用另外一种思路，既然使用一个向量表达用户多样兴趣有困难，那么为什么不使用一组向量呢？具体来说，如果我们可以对用户历史行为的embedding进行聚类，聚类后的每个簇代表用户的一组兴趣，不就解决问题了么。 
•	传统序列化推荐方法只考虑了用户的last behavior，没有使用到完整的session行为序列信息，作者引入RNN-based方法直接解决这个问题

2）模型与特点
DIN在多值离散型特征的Embedding进入Pooling之前，先引入一个Attention，乘上权重。论文将Attention定义为“Local Activation Unit”，它接受两个输入：广告的Embedding，用户特定兴趣或行为的Embedding。Attention的作用、权重的物理意义是特定的广告对用户的某个或某几个兴趣产生的特定的激活。
 
MIND借鉴了Hiton的胶囊网络(Capsule Network)，提出了Multi-Interest Extractor Layer来对用户历史行为embedding进行软聚类，结构如下图所示：
  
（3）数据集
Amazon Books和TmallData两个电商数据集
（4）代码地址
https://github.com/zhougr1993/DeepInterestNetwork

（5）Baselines：
MIND与Youtube DNN等其他方法做了对比，表现都是最好的。另外还对Label-aware attetion中power操作指数的大小做了实验，结果证明“注意力越集中，效果越好”。
线上实验的话，表现也是优于YouTube DNN和item-based CF的(注意，item-based CF效果要好于YouTubeDNN，作者在这里给YouTubeDNN找了下场子，说是可能因为item-based CF经过长时间实践优化的原因)。另外，线上实验还表明，用户兴趣分得越细(用户向量数K越大)，效果越好。根据用户历史行为数动态调整兴趣数(K的值)虽然线上指标没有提升，但是模型资源消耗降低了。


### 双塔模型 DSSM
LearningDeep Structured Semantic Models for Web Search using Clickthrough Data
Sampling-bias-corrected neural modeling for large corpus item recommendations –google
1）背景与动机
在大规模的推荐系统中，利用双塔模型对user-item对的交互关系进行建模，从而学习【用户，上下文】向量和【item】向量的关联。针对大规模流数据，提出in-batch softmax损失函数与流数据频率估计方法更好的适应item的多种数据分布。
2）模型与特点
利用双塔模型构建Youtube视频推荐系统，对于用户侧的塔根据用户观看视频特征构建user embedding，对于视频侧的塔根据视频特征构建video emebdding。两个塔分别是相互独立的网络。
 

https://github.com/InsaneLife/dssm

-----------------------------------


如果考虑到用户历史行为的序列化建模，业界研究近期主要集中在Session-Based Recommendation.
基于会话的推荐，我们可以理解为从进入一个app直到退出这一过程中，根据你的行为变化所发生的推荐；也可以理解为根据你较短时间内的行为序列发生的推荐.

### 2015 GRU4REC：Session-based Recommendations with Recurrent Neural Networks
1）背景与动机
传统序列化推荐方法只考虑了用户的last behavior，没有使用到完整的session行为序列信息，作者引入RNN-based方法直接解决这个问题
2）模型与特点
 
GRU4REC是一个单纯基于GRU的session sequence->next step的序列化预测模型（不同于sequence->target类型的序列化建模），每一个step 的输入经过embedding layer给到GRU，得到next step的预测
（3）数据集
RecSys Challenge 2015：网站点击流
Youtube-like OTT video service platform Collection
（4）代码地址/评价指标
recall@20、MRR
https://github.com/Songweiping/GRU4Rec_TensorFlow
（5）Baselines：
POP：推荐训练集中最受欢迎的item；
S-POP：推荐当前session中最受欢迎的item；
Item-KNN：推荐与实际item相似的item，相似度被定义为session向量之间的余弦相似度
BPR-MF：一种矩阵分解法，新会话的特征向量为其内的item的特征向量的平均，把它作为用户特征向量。
（6）总结
GRU4Rec算是开创了RNN-based的session行为推荐方法，相比较于未来混合结构的序列模型，单一RNN的结构就比较简单。但是其在工程实践方面的优化以及loss的思考上确实很不错，开阔大家的思路

### 2018 SASRec：Self-Attentive Sequential Recommendation
1）背景与动机
传统MC方法仅考虑用户last behavior的影响，模型简单在稀疏数据场景效果更好。RNN-based的方法，能够处理长的用户行为序列，模型复杂在数据丰富并且支持复杂计算的场景更好。SASRec作为Attention-based方法，在两类方法之间做到一定的兼顾和平衡
2） 模型与特点

 
因为self-attention model没有任何关于序列的结构化信息，所以SASRec在embedding layer加入position embedding，给模型引入了结构信息
“We modify the attention by forbidding all links between Qi and Kj (j > i) ”。整体结构脱胎于Transformer，不同点在于在Self Attention Layer保证了left-to-right unidirectional architectures，进行一层Mask处理就好了
4）代码地址
https://github.com/kang205/SASRec
5）小结
Transformer在序列化推荐上的应用，对于Self Attention的改造值得学习

### 2018 DIEN: Deep Interest Evolution Network for Click-Through Rate Prediction 阿里
1）背景与动机
这篇文章是阿里DIN的升级版，提出一个观点是用户兴趣是随时间而发生变化的
2）模型与特点
 
Behavior Layer，behavior序列/category特征做embedding
Interest Extractor Layer，区别于一般的直接用behavior embedding表达兴趣(类似DIN)，这里使用GRU来提取用户的潜在兴趣表达&行为之间的依赖。
文章提到3种Attention和GRU结合的方式AIGRU/AGRU/AUGRU，AIGRU通过attention score at标量乘法直接弱化hidden state ht的影响，但input 0也能影响到GRU的hidden state；AGRU用attention score替换了GRU的update gate，也是at标量乘法直接弱化hidden state ht的影响，但是失去了分布式表达不同维度差异化的表达；AUGRU能够减弱兴趣漂移的干扰，让相关性高的兴趣能平滑的evolve，最后采用了AUGRU
•	小结
DIEN对attention和GRU的结合做了比较多的工作
https://github.com/mouna99/dien


### 2018 Caser：Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding
1）背景与动机
主要还是解决MC类方法，只考虑last behavior而没有考虑序列pattern的问题。虽然RNN-Based和Attention-Based方法一定程度上能够解决这类问题，但文章提出的基于CNN思路很不错值得学习下
2）模型与特点

 
在用户行为序列上滑窗产生样本，前L个行为作为input，接下来T个行为作为target
L个input行为经过embedding layer得到L*d的矩阵（d是latent dimension size），将其看做一张二维图像，分别用水平和垂直两个convolution kernel来捕捉用户行为序列里面的结构信息
最后和user representation concat到一起全连接，预测next T-targets
•	小结
没有直接作为典型的left-to-right unidirectional 结构来处理，而是整个作为结构化的信息交给CNN来进行特征提取。
https://github.com/graytowne/caser

E 2019 BERT4Rec：Sequential Recommendation with Bidirectional Encoder Representations from Transformer
•	1)背景与动机
•	这是阿里在电商推荐上对大热BERT的一次尝试。文章挑战了left-to-right unidirectional 结构的表达能力，以及其在实际推荐场景的合理性。Introduction中举了一个例子，用户在浏览多个口红时的顺序并没有太多的作用，作者Bidirectional model更为合适
2)  模型与特点

  
将用户行为序列看做文本序列，但BERT不能直接用双向信息做序列预测（会造成信息泄露），于是在文章中序列化建模的问题转变成了Cloze task。随机mask掉user behavior sequence中部分item，然后基于sequence中的上下文来预测被mask掉的item。 Cloze Task随机对sequence mask来构成样本的机制，使得BERT4Rec可以产生大量的样本用于training
为了training task保持一致，预测时在user behavior sequence最后加上special token [mask]，然后进行mask predict即等价于预测next item。为了使得模型更好的match predict task，特意构造出只mask sequence最后一个item这样的样本用于训练
•	小结
1.	文章实际上依靠position embedding来产生sequence的信息，bidirectional sequential更多的是基于Transformer的context
2.	对于序列化建模问题的转换，以及训练样本的处理都很值得借鉴

https://github.com/FeiSun/BERT4Rec

### 2019 DSIN：Deep Session Interest Network for Click-Through Rate Prediction 阿里
1) 背景与动机
用户在session内的行为是内聚的，而跨session的行为会出现显著差异。一般的序列化建模，并不会刻意去区分行为序列中不同session之间的差异。DSIN则是构建了一种session层面的序列模型，意图解决这个问题
2)模型与特点
DSIN模型的总体框架如下图：
   
DSIN在全连接层之前，分成了两部分，左边的那一部分，将用户特征和物品特征转换对应的向量表示，这部分主要是一个embedding层，就不再过多的描述。右边的那一部分主要是对用户行为序列进行处理，从下到上分为四层：
1）序列切分层session division layer
2）会话兴趣抽取层session interest extractor layer
3）会话间兴趣交互层session interest interacting layer
4）会话兴趣激活层session interest acti- vating layer

Session Division Layer将用户行为序列切到不同的session中(浅粉色)；
紧接着session序列中每个session做一个sum polling，再加上bias encoding输入到Transformer（浅黄色）；
Transformer输出的序列，一方面与待预测item通过激活单元做了一个attention的处理（浅紫色），一方面输入到Bi-LSTM（浅蓝色）；
经过Bi-LSTM，用户行为才得到真正序列化处理（个人认为transformer处理序列数据，仅仅是将sequence的内聚性做了进一步的加强，所以很多应用还要额外再输入position encoding来强调结构化信息），再和item做activation attention
https://arxiv.org/abs/1905.06482


https://github.com/shenweichen/DSIN







https://github.com/NVIDIA/DeepRecommender

-------------------------


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
