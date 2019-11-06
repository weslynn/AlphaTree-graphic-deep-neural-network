# GRAPH NEURAL NETWORK(GNN)图神经网络
CNN只适用于张量数据，例如二维图像或一维文本序列。然而，有许多数据及其关系是难以简单的用张量表示的，而是需要借助另一种常见的数据结构，即由顶点（vertex）和边（edge）表示的图。

图的结构一般来说是十分不规则的，可以认为是无限维的一种数据，所以它没有平移不变性。每一个节点的周围结构可能都是独一无二的，像是卷积或者池化这样对于张量数据十分自然的操作，对于图结构却是难以直接定义的，这就是图网络所研究的主要问题：如何对图进行学习？

我们需要从图里得出特征，得到图的嵌入表示（graph embedding），然后进行进行节点分类（node classification）、图分类（graph classification）、边预测（link prediction）


综述：

Deep Learning on Graphs: A Survey [pdf](https://arxiv.org/pdf/1812.04202.pdf)

Graph Neural Networks: A Review of Methods and Applications  [pdf](https://arxiv.org/pdf/1812.08434.pdf)

Representation Learning on Networks [link](http://snap.stanford.edu/proj/embeddings-www/)



A Comprehensive Survey of Graph Embedding: Problems, Techniques and Applications
(https://arxiv.org/pdf/1709.07604.pdf)

 HOW POWERFUL ARE GRAPH NEURAL NETWORKS?[link](https://link.zhihu.com/?target=https%3A//cs.stanford.edu/people/jure/pubs/gin-iclr19.pdf)

要想对图进行学习，首先需要对图的顶点数据、边数据和子图数据进行降维，这就是图嵌入（graph embedding）。


![graphembeding](https://github.com/weslynn/graphic-deep-neural-network/blob/master/gnnpic/graphembeding.jpg)


如图1所示：一幅图（image）所抽取的特征图（features map）里每个元素，可以理解为图（image）上的对应点的像素及周边点的像素的加权和（还需要再激活一下）。

同样可以设想：一个图（graph）所抽取的特征图（也就是特征向量）里的每个元素，也可以理解为图（graph）上对应节点的向量与周边节点的向量的加权和。



![image-graphhd](https://github.com/weslynn/graphic-deep-neural-network/blob/master/gnnpic/image-graphhd.jpg)


## Graph Embedding

![graphembedingpaper](https://github.com/weslynn/graphic-deep-neural-network/blob/master/gnnpic/graphembedingpaper.jpg)



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


![DeepWalkmore](https://github.com/weslynn/graphic-deep-neural-network/blob/master/gnnpic/DeepWalkmore.jpg)






### 3.基于深度学习的方法

从结构来分析，可以分为
- 图自编码器（ Graph Autoencoders）
- 图卷积网络（Graph Convolution Networks，GCN）
- 图注意力网络（Graph Attention Networks)
- 图生成网络（ Graph Generative Networks） 
- 图时空网络（Graph Spatial-temporal Networks）


![notDeepwalk](https://github.com/weslynn/graphic-deep-neural-network/blob/master/gnnpic/notDeepwalk.jpg)

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

GCN的概念首次提出于ICLR2017（成文于2016年）
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


