# GRAPH NEURAL NETWORK(GNN)图神经网络
CNN只适用于张量数据，例如二维图像或一维文本序列。然而，有许多数据及其关系是难以简单的用张量表示的，而是需要借助另一种常见的数据结构，即由顶点（vertex）和边（edge）表示的图。

图的结构一般来说是十分不规则的，可以认为是无限维的一种数据，所以它没有平移不变性。每一个节点的周围结构可能都是独一无二的，像是卷积或者池化这样对于张量数据十分自然的操作，对于图结构却是难以直接定义的，这就是图网络所研究的主要问题：如何对图进行学习？

我们需要从图里得出特征，得到图的嵌入表示（graph embedding），然后进行进行节点分类（node classification）、图分类（graph classification）、边预测（link prediction）


综述：

Deep Learning on Graphs: A Survey [pdf](https://arxiv.org/pdf/1812.04202.pdf)

Graph Neural Networks: A Review of Methods and Applications  [pdf](https://arxiv.org/pdf/1812.08434.pdf)

Representation Learning on Networks [link](http://snap.stanford.edu/proj/embeddings-www/)

A Comprehensive Survey on Graph Neural Networks (https://arxiv.org/abs/1901.00596.pdf)

A Comprehensive Survey of Graph Embedding: Problems, Techniques and Applications
(https://arxiv.org/pdf/1709.07604.pdf)

 HOW POWERFUL ARE GRAPH NEURAL NETWORKS?[link](https://link.zhihu.com/?target=https%3A//cs.stanford.edu/people/jure/pubs/gin-iclr19.pdf)

要想对图进行学习，首先需要对图的顶点数据、边数据和子图数据进行降维，这就是图嵌入（graph embedding）。


![graphembeding](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/gnnpic/graphembeding.jpg)


如图1所示：一幅图（image）所抽取的特征图（features map）里每个元素，可以理解为图（image）上的对应点的像素及周边点的像素的加权和（还需要再激活一下）。

同样可以设想：一个图（graph）所抽取的特征图（也就是特征向量）里的每个元素，也可以理解为图（graph）上对应节点的向量与周边节点的向量的加权和。



![image-graphhd](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/gnnpic/image-graphhd.jpg)


## Graph Embedding

![graphembedingpaper](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/gnnpic/graphembedingpaper.jpg)



### 1.基于因子分解的方法

例如，DeepWalk是一种基本的基于表示学习的图嵌入方法，它将SkipGram模型与随机游走相结合。但是，这类方法也有计算量大和泛化能力弱等缺点。当增加一个新的节点时，DeepWalk需要重新进行训练，因而也就不适用于动态图。


### 2.基于随机游走的方法
### 2.1. DeepWalk

DeepWalk方法受到word2vec的启发，首先选择某一特定点为起始点，做随机游走得到点的序列，然后将这个得到的序列视为句子，用word2vec来学习，得到该点的表示向量。DeepWalk通过随机游走去可以获图中点的局部上下文信息，因此学到的表示向量反映的是该点在图中的局部结构，两个点在图中共有的邻近点（或者高阶邻近点）越多，则对应的两个向量之间的距离就越短。

DeepWalk: https://github.com/phanein/deepwalk

LINE: Large-scale information network embedding

http://dl.acm.org/citation.cfm?id=2741093

https://github.com/snowkylin/line 

https://github.com/VahidooX/LINE

https://github.com/tangjianpku/LINE c++

简介：

虽然 DeepWalk 和 LINE 属于网络表示学习中的算法，与现在端到端的图神经网络有一定的区别，但目前一些图神经网络应用（如社交网络、引用网络节点分类）依然使用 DeepWalk/LINE 来作为预训练算法，无监督地为节点获得初始特征表示。另外，DeepWalk 项目中的 Random Walk 也可以被直接拿来用作图神经网络的数据采样操作。

### 2.2. node2vec

与DeepWalk相似，node2vec通过最大化随机游走得到的序列中的节点出现的概率来保持节点之间的高阶邻近性。与DeepWalk的最大区别在于，node2vec采用有偏随机游走，在广度优先（bfs）和深度优先（dfs）图搜索之间进行权衡，从而产生比DeepWalk更高质量和更多信息量的嵌入。

node2vec: Scalable Feature Learning for Networks, KDD’16

http://dl.acm.org/citation.cfm?id=2939672.2939754

https://arxiv.org/abs/1607.00653

https://github.com/aditya-grover/node2vec

https://github.com/apple2373/node2vec

https://github.com/eliorc/node2vec

https://github.com/xgfs/node2vec-c

### 2.3. Hierarchical representation learning for networks (HARP)

DeepWalk和node2vec随机初始化节点嵌入以训练模型。由于它们的目标函数是非凸的，这种初始化很可能陷入局部最优。HARP引入了一种策略，通过更好的权重初始化来改进解决方案并避免局部最优。为此，HARP通过使用图形粗化聚合层次结构上一层中的节点来创建节点的层次结构。然后，它生成最粗糙的图的嵌入，并用所学到的嵌入初始化精炼图的节点嵌入（层次结构中的一个）。它通过层次结构传播这种嵌入，以获得原始图形的嵌入。因此，可以将HARP与基于随机行走的方法（如DeepWalk和node2vec）结合使用，以获得更好的优化函数解。

### 2.4. Walklets

DeepWalk和node2vec通过随机游走生成的序列，隐式地保持节点之间的高阶邻近性，由于其随机性，这些随机游走会得到不同距离的连接节点。另一方面，基于因子分解的方法，如GF和HOPE，通过在目标函数中对节点进行建模，明确地保留了节点之间的距离。Walklets将显式建模与随机游走的思想结合起来。该模型通过跳过图中的某些节点来修改DeepWalk中使用的随机游走策略。这是针对多个尺度的跳跃长度执行的，类似于在GraRep中分解，并且随机行走获得的一组点的序列用于训练类似于DeepWalk的模型。


![DeepWalkmore](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/gnnpic/DeepWalkmore.jpg)






### 3.基于深度学习的方法

从结构来分析，可以分为
- 图自编码器（ Graph Autoencoders）
- 图卷积网络（Graph Convolution Networks，GCN）
- 图注意力网络（Graph Attention Networks)
- 图生成网络（ Graph Generative Networks） 
- 图时空网络（Graph Spatial-temporal Networks）


![notDeepwalk](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/gnnpic/notDeepwalk.jpg)

### 3.1. Graph Auto-Encoders


#### Structural deep network embedding (SDNE)

SDNE建议使用深度自动编码器来保持一阶和二阶网络邻近度。它通过联合优化这两个近似值来实现这一点。该方法利用高度非线性函数来获得嵌入。模型由两部分组成：无监督和监督。前者包括一个自动编码器，目的是寻找一个可以重构其邻域的节点的嵌入。后者基于拉普拉斯特征映射，当相似顶点在嵌入空间中彼此映射得很远时，该特征映射会受到惩罚。

[KDD 2016](http://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)

https://github.com/xiaohan2012/sdne-keras

#### Deep neural networks for learning graph representations (DNGR)

DNGR结合了随机游走和深度自动编码器。该模型由3部分组成：随机游走、正点互信息（PPMI）计算和叠加去噪自编码器。在输入图上使用随机游走模型生成概率共现矩阵，类似于HOPE中的相似矩阵。将该矩阵转化为PPMI矩阵，输入到叠加去噪自动编码器中得到嵌入。输入PPMI矩阵保证了自动编码器模型能够捕获更高阶的近似度。此外，使用叠加去噪自动编码器有助于模型在图中存在噪声时的鲁棒性，以及捕获任务（如链路预测和节点分类）所需的底层结构。


#### Variational graph auto-encoders (VGAE)

VGAE采用了图形卷积网络（GCN）编码器和内积译码器。输入是邻接矩阵，它们依赖于GCN来学习节点之间的高阶依赖关系。他们的经验表明，与非概率自编码器相比，使用变分自编码器可以提高性能。

https://arxiv.org/abs/1611.07308

https://github.com/tkipf/gae


https://zhuanlan.zhihu.com/p/62629465

基于深度神经网络的方法，即SDNE和DNGR，以每个节点的全局邻域（一行DNGR的PPMI和SDNE的邻接矩阵）作为输入。对于大型稀疏图来说，这可能是一种计算代价很高且不适用的方法。





### 3.2. Graph convolutional networks (GCN)


Bruna等人(2013)首次提出了对GCNs的突出研究，该研究基于频谱图理论开发了一种图形卷积的变体。
由于频谱方法通常同时处理整个图，且难以并行或缩放到大型图，因此基于空间的图卷积网络近年来发展迅速。这些方法通过聚集相邻节点的信息，直接在图域中进行卷积。结合采样策略，可以在一批节点中进行计算，而不是整个图，具有提高效率的潜力。

使用图结构和节点内容信息作为输入, GCN的输出使用以下不同的机制, 可以关注于不同的图分析任务.

- 节点级输出与节点回归和分类任务相关. 图卷积模块直接给出节点潜在的表达, 多层感知机或者softmax层被用作最终的GCN层.
- 边级输出与边分类和连接预测任务相关. 为了预测边的标记和连接强度, 一个附加函数将会把来自图卷积模块的两个节点的潜在表达作为输入.
- 图级输出与图分类任务有关. 为了获取一个图级的紧凑表达, 池化模块被用来粗化图到子图, 或用来加和/平均节点表达.

![gcn](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/gnnpic/gcn.png)


GCN的概念首次提出于ICLR2017（成文于2016年）

Thomas Kipf, Graph Convolutional Networks (2016)

http://tkipf.github.io/graph-convolutional-networks/

图卷积网络（GCN）通过在图上定义卷积算子来解决这个问题。该模型迭代地聚合了节点的邻域嵌入，并使用在前一次迭代中获得的嵌入及其嵌入的函数来获得新的嵌入。仅局部邻域的聚合嵌入使其具有可扩展性，并且多次迭代允许学习嵌入一个节点来描述全局邻域。最近几篇论文提出了利用图上的卷积来获得半监督嵌入的方法，这种方法可以通过为每个节点定义唯一的标签来获得无监督嵌入。这些方法在卷积滤波器的构造上各不相同，卷积滤波器可大致分为空间滤波器和谱滤波器。空间滤波器直接作用于原始图和邻接矩阵，而谱滤波器作用于拉普拉斯图的谱。

TensorFlow: https://github.com/tkipf/gcn

PyTorch: https://github.com/tkipf/pygcn

GCN 论文作者提供的源码，该源码提供了大量关于稀疏矩阵的代码。例如如何构建稀疏的变换矩阵（这部分代码被其他许多项目复用）、如何将稀疏 CSR 矩阵变换为 TensorFlow/PyTorch 的稀疏 Tensor，以及如何构建兼容稀疏和非稀疏的全连接层等，几乎是图神经网络必读的源码之一了。


- 快速图卷积网络 FastGCN 

FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling



https://arxiv.org/abs/1801.10247

https://github.com/matenure/FastGCN

https://openreview.net/forum?id=rytstxWAW

FastGCN 作者提供的源码，基于采样的方式构建 mini-match 来训练 GCN，解决了 GCN 不能处理大规模数据的问题。


### 3.3.Graph Attention Networks(GAT) 图注意力网络 

论文地址：https://arxiv.org/abs/1710.10903

GAT 论文作者提供的源码。源码中关于 mask 的实现、以及稀疏版 GAT 的实现值得借鉴。

Github：https://github.com/PetarV-/GAT




#### DeepInf 

Mini-batch版 图注意力网络 

https://github.com/xptree/DeepInf

简介：

DeepInf 论文其实是 GAT 的一个应用，但其基于 Random Walk 采样子图构建 mini-batch 的方法解决了 GAT 在大规模网络上应用的问题。




其他：

Graph Embedding Techniques, Applications, and Performance: A Survey : https://github.com/palash1992/GEM


## 库

### DGL

### PyTorch Geometric:PyG

https://github.com/rusty1s/pytorch_geometric

https://rusty1s.github.io/pytorch_geometric/build/html/index.html

https://arxiv.org/pdf/1903.02428.pdf

[知乎](https://zhuanlan.zhihu.com/p/58987454)


###  Graph Nets
DeepMind 开源的图神经网络框架 Graph Nets

链接：

https://github.com/deepmind/graph_nets

简介：

基于 TensorFlow 和 Sonnet。上面的项目更侧重于节点特征的计算，而 graph_nets 同时包含节点和边的计算，可用于一些高级任务，如最短路径、物理场景模拟等。


### Euler

工业级分布式图神经网络框架 Euler

链接：

https://github.com/alibaba/euler

简介：

Euler 是阿里巴巴开源的大规模分布式的图学习框架，配合 TensorFlow 或者阿里开源的 XDL 等深度学习工具，它支持用户在数十亿点数百亿边的复杂异构图上进行模型训练。


参考 http://www.qianjia.com/html/2019-03/19_329563.html

## 应用


1.https://github.com/jihoo-kim/awesome-RecSys?fbclid=IwAR1m6OebmqO9mfLV1ta4OTihQc9Phw8WNS4zdr5IeT1X1OLWQvLk0Wz45f4

### 知识图谱（knowledge graph, KG）
 KG里面节点是entity，边是一些特定的semantic relation，是一个图的结构。早期就有很多在KG上学graph embedding然后做link prediction之类的工作（代表作TransE等），有GNN之后自然可以用来学KG上的embedding，做knowledge base completion。

Modeling Relational Data with Graph Convolutional Networks https://link.springer.com/chapter/10.1007/978-3-319-93417-4_38

One-Shot Relational Learning for Knowledge Graphs https://www.aclweb.org/anthology/D18-1223


 然后利用KG可以帮助很多别的task，比如zero-shot learning，question answering等等。

Zero-Shot Recognition via Semantic Embeddings and Knowledge Graphs http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Zero-Shot_Recognition_via_CVPR_2018_paper.html

Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text https://www.aclweb.org/anthology/D18-1455


### 文本分析

直接从文本中构建entity的graph的，去学entity之间的relation、coreference等关系。这个时候的GNN其实就是fully-connected graph上的self-attention，思想是一样的。拿建出来的graph可以再去做别的task比如generation。

Multi-Task Identification of Entities, Relations, and Coreferencefor Scientific Knowledge Graph Construction http://ssli.ee.washington.edu/~luanyi/YiLuan_files/EMNLP2018_YL_sciIE.pdf




### 句法结构（dependency structure）

虽然一般是树结构，但是还是可以用图的模型来处理，或者有的时候加一些边就变成了图。这种linguistic knowledge在很多地方大家会认为有帮助，比如关系抽取，从以前的Tree LSTM[10]到Graph LSTM[11]，也可以用GCN[12]；比如semantic role labeling[13][14]（后面这篇是去年EMNLP best paper，我上面说了self-attention 就是fully-connected graph的GNN）；还有machine translation[15]。


### 社交网络（social network）

GCN以及后续很多paper的实验setting用的是杨植麟学长去CMU的第一个工作[16]，我们本科是在一个实验室，所以做GNN的初衷很多是从social network这些地方来的。另外也可以去看Jure的工作。对NLP而言，分析social media的时候考虑network structure一般都是很有帮助的，所以我相信一定有很多用GNN的工作，我就偷个懒不去找引用了，最近看这些比较少。利用user-item graph也可以做推荐系统[17]。


[zhihu](https://www.zhihu.com/question/330103469/answer/731276217)

### 医药

癌症药物发现 Veselkov等人,Nature,2019





## Databases:

### 引文网络（Cora、PubMed、Citeseer）
引文网络，顾名思义就是由论文和他们的关系构成的网络，这些关系包括例如引用关系、共同的作者等，具有天然的图结构，数据集的任务一般是论文的分类和连接的预测，比较流行的数据集有三个，分别是Cora、PubMed、Citeseer，它们的组成情况如图1所示，Nodes也就是数据集的论文数量，features是每篇论文的特征，数据集中有一个包含多个单词的词汇表，去除了出现频率小于10的词，但是不进行编码，论文的属性是由一串二进制码构成，只用0和1表示该论文有无这个词汇。

样本特征，标签，邻接矩阵

文件构成
以cora数据集为例，数据集包含两个文件，cora.cites和cora.content，cora.cites文件中的数据如下：

<ID of cited paper> <ID of citing paper>

即原论文和引用的论文，刚好构成了一条天然的边，cora.content文件的数据如下：

<paper id> <word attributes> + <class label>

有论文id、上面说到的二进制码和论文对应的类别组成。

该数据集共2708个样本点，每个样本点都是一篇科学论文，所有样本点被分为8个类别，类别分别是1）基于案例；2）遗传算法；3）神经网络；4）概率方法；5）强化学习；6）规则学习；7）理论

每篇论文都由一个1433维的词向量表示，所以，每个样本点具有1433个特征。词向量的每个元素都对应一个词，且该元素只有0或1两个取值。取0表示该元素对应的词不在论文中，取1表示在论文中。所有的词来源于一个具有1433个词的字典。

每篇论文都至少引用了一篇其他论文，或者被其他论文引用，也就是样本点之间存在联系，没有任何一个样本点与其他样本点完全没联系。如果将样本点看做图中的点，则这是一个连通的图，不存在孤立点。

文件格式

下载的压缩包中有三个文件，分别是cora.cites，cora.content，README。

README是对数据集的介绍；cora.content是所有论文的独自的信息；cora.cites是论文之间的引用记录。

cora.content共有2708行，每一行代表一个样本点，即一篇论文。如下所示，每一行由三部分组成，分别是论文的编号，如31336；论文的词向量，一个有1433位的二进制；论文的类别，如Neural_Networks。

	31336	0	0.....	0	0	0	0	0	0	0	0	0	0	0	0	Neural_Networks
	1061127	0	0.....	0	0	0	0	0	0	0	0	0	0	0	0	Rule_Learning
	1106406	0	0.....	0	0	0	0	0	0	0	0	0	0	0	Reinforcement_Learning

cora.cites共5429行， 每一行有两个论文编号，表示第一个编号的论文先写，第二个编号的论文引用第一个编号的论文。如下所示：

	35	1033
	35	103482
	35	103515

如果将论文看做图中的点，那么这5429行便是点之间的5429条边。

Cora：https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/cora_raw.zip

https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

Pubmed：https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/pubmed.zip

Citeseer：https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/citeseer.zip

### 社交网络（BlogCatalog、Reddit、Epinions）


BlogCatalog数据集是一个社会关系网络，图是由博主和他（她）的社会关系（比如好友）组成，labels是博主的兴趣爱好。Reddit数据集是由来自Reddit论坛的帖子组成，如果两个帖子被同一人评论，那么在构图的时候，就认为这两个帖子是相关联的，labels就是每个帖子对应的社区分类。Epinions是一个从一个在线商品评论网站收集的多图数据集，里面包含了多种关系，比如评论者对于另一个评论者的态度（信任/不信任），以及评论者对商品的评级。

文件构成
BlogCatalog数据集的结点数为10312，边条数为333983，label维度为39，数据集包含两个文件：

Nodes.csv：以字典的形式存储用户的信息，但是只包含节点id。

Edges.csv：存储博主的社交网络（好友等），以此来构图。

Epinions数据集包含文件如下：

Ratings_data.txt：包含用户对于一件物品的评级，文件中每一行的结构为user_id

item_id rating_value。

Trust_data.txt：存储了用户对其他用户的信任状态，存储方式为source_user_id

target_user_id trust_statement_value，其中信任状态只有信任和不信任（1、0）。



BlogCatalog：http://socialcomputing.asu.edu/datasets/BlogCatalog

Reddit：https://github.com/linanqiu/reddit-dataset

Epinions：http://www.trustlet.org/downloaded_epinions.html


### 生物化学结构（PPI、NCI-1、NCI-109、MUTAG、QM9、Tox21）

PPI是蛋白质互作网络，数据集中共有24张图，其中20张作为训练，2张作为验证，2张作为测试，每张图对应不同的人体组织，实例如图3，该数据是为了从系统的角度研究疾病分子机制、发现新药靶点等等。


平均每张图有2372个结点，每个结点特征长度为50，其中包含位置基因集，基序集和免疫学特征。基因本体集作为labels（总共121个），labels不是one-hot编码。

NCI-1、NCI-109和MUTAG是关于化学分子和化合物的数据集，原子代表结点，化学键代表边。NCI-1和NCI-109数据集分别包含4100和4127个化合物，labels是判断化合物是否有阻碍癌细胞增长得性质。MUTAG数据集包含188个硝基化合物，labels是判断化合物是芳香族还是杂芳族。

 QM9数据集包括了13万有机分子的构成,空间信息及其对应的属性. 它被广泛应用于各类数据驱动的分子属性预测方法的实验和对比。

Toxicology in the 21st Century 简称tox21，任务是使用化学结构数据预测化合物对生物化学途径的干扰，研究、开发、评估和翻译创新的测试方法，以更好地预测物质如何影响人类和环境。数据集有12707张图，12个labels。

文件构成
PPI数据集的构成：

train/test/valid_graph.json：保存了训练、验证、测试的图结构数据。

train/test/valid_feats.npy ：保存结点的特征，以numpy.ndarry的形式存储，shape为[n, v]，n是结点的个数，v是特征的长度。

train/test/valid_labels.npy：保存结点的label，也是以numpy.ndarry的形式存储，形为nxh，h为label的长度。

train/test/valid/_ graph_id.npy ：表示这个结点属于哪张图，形式为numpy.ndarry，例如[1, 1, 2, 1...20].。

NCI-1、NCI-109和MUTAG数据集的文件构成如下：（用DS代替数据集名称）

n表示结点数，m表示边的个数，N表示图的个数

DS_A.txt (m lines)：图的邻接矩阵，每一行的结构为(row, col)，即一条边。

DS_graph_indicator.txt (n lines)：表明结点属于哪一个图的文件。

DS_graph_labels.txt (N lines)：图的labels。

DS_node_labels.txt (n lines)：结点的labels。

DS_edge_labels.txt (m lines)：边labels。

DS_edge_attributes.txt (m lines)：边特征。

DS_node_attributes.txt (n lines)：结点的特征。

DS_graph_attributes.txt (N lines)：图的特征，可以理解为全局变量。

QM9的文件结构如下：

QM9_nano.npz：该文件需要用numpy读取,其中包含三个字段：

'ID' 分子的id，如:qm9:000001；

'Atom' 分子的原子构成，为一个由原子序数的列表构成,如[6,1,1,1,1]表示该分子由一个碳(C)原子和4个氢(H)原子构成.；

'Distance' 分子中原子的距离矩阵,以上面[6,1,1,1,1]分子为例,它的距离矩阵即为一个5x5的矩阵,其中行列的顺序和上述列表一致,即矩阵的第N行/列对应的是列表的第N个原子信息.

'U0' 分子的能量属性(温度为0K时),也是我们需要预测的值（分类的种类为13）

Tox21文件夹中包含13个文件，其中12个文件夹就是化合物的分类


PPI：http://snap.stanford.edu/graphsage/ppi.zip

NCI-1：https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip

NCI-109：https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/NCI109.zip

MUTAG：https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip

QM9：https://github.com/geekinglcq/QM9nano4USTC

Tox21：https://tripod.nih.gov/tox21/challenge/data.jsp




