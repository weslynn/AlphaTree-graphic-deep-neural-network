
# NLP 自然语言处理


NLP 是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。自然语言处理是一门融语言学、计算机科学、数学于一体的学科。NLP 由两个主要的技术领域构成：自然语言理解和自然语言生成。

自然语言理解方向，主要目标是帮助机器更好理解人的语言，包括基础的词法、句法等语义理解，以及需求、篇章、情感层面的高层理解。

自然语言理解按照功能分：
- 分类（Classification）:

	·文本分类、情感分析

	·应用：文档分类、信息过滤、标签生成…

- 匹配（Matching）：

	·语义相似计算、同义词/近义词

	·应用：检索、排序、过滤、聚类、基础特征

- 结构化预测（Structured Prediction）：

	·NER、PoS Tagging、Semantic Role Labeling、Parsing

	·应用：分词、语法分析、信息提取

- 转换/翻译（Transform/Translate):

	·机器翻译、语音识别、摘要

	·应用：机器翻译、语音助手


自然语言生成方向，主要目标是帮助机器生成人能够理解的语言，比如文本生成、自动文摘等。


![history](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/nlppic/history.png)

历史：自然语言处理随着计算机的出现而出现，最早是做规则的系统：rules-Based，后面做统计的系统language modeling
，现在做神经网络的系统。

初期NLP方向主要为RNN 循环神经网络

循环神经网络是一类用于处理序列数据的神经网络。就像卷积网络是专门处理网格化数据XX(如一个图像)的神经网络，循环神经网络是专门用于处理序列x(1),...,x(τ)x(1),...,x(τ)的神经网络。正如卷积网络可以很容易地扩展到具有很大宽度和高度的图像，以及处理大小可变的图像，循环网络可以扩展到更长的序列，且大多数循环网络可以处理可变长度的序列。

后来主要为Transformer。

神经网络自然语言处理的技术体系分为如下几个部分：


## Word Embedding 词的编码 2014
	词编码的目的是用多维向量来表征词的语义。
### One-Hot(bag of word)

### fast-text 

    https://github.com/facebookresearch/fastText


### word2vec
	- CBOW（(Continuous Bag-of-Words），用周围的词预测当前的词
	- Skip-gram，用当前的词预测周围的词。
	通过大规模的学习训练，就可以得到每个词稳定的多维向量，作为它的语义表示。

![word2vec](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/nlppic/word2vec.jpg)


句子中每个单词以Onehot形式作为输入，然后乘以学好的Word Embedding矩阵Q，就直接取出单词对应的Word Embedding了


## 句子的编码
	一般通过RNN（循环神经网络）或者CNN（卷积神经网络）来做。
	- RNN从左到右对句子进行建模，每个词对应一个隐状态，该引状态代表了从句首到当前词的语义信息，句尾的状态就代表了全句的信息。
	- CNN从理论上分别进行词嵌入+位置嵌入+卷积，加上一个向量表示，对应句子的语义。

基于这样的表征，我们就可以做编码、解码机制。比如说我们可以用图上的红点，它代表全句的语义信息，来进行解码，可以从一种语言翻译成另一种语言，凡是从一个序列串变成另外一个序列串都可以通过编码、解码机制来运行。

![sentence](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/nlppic/sentence.JPG)


## 注意力模型Attention Model

	https://arxiv.org/abs/1706.03762

	它综合考量了在当前状态下对应的编码的每一个隐状态，加权平均，来体现当前的动态输入。这类技术引入之后，神经网络机器翻译就得到了飞速的发展。
	后面又引入了Transformer。Transformer引入了自编码，一个词跟周围的词建立相似，引入多头，可以引入多种特征表达，所以编码效果或者编码的信息更加丰富。

### Transformer
	Transformer是个叠加的“自注意力机制（Self Attention）”构成的深度网络，是目前NLP里最强的特征提取器

	http://nlp.seas.harvard.edu/2018/04/03/attention.html

	https://jalammar.github.io/illustrated-transformer/

	Transformer利用self-attention和position embedding克服了RNN中的长距离依赖、无法并行计算的缺点，也解决了CNN中远距离特征捕获难的问题，并在机器翻译领域大有取代RNN的之势。然而，在语言建模中，Transformer 目前使用固定长度的上下文来实现，即将一个长文本序列截成多个包含数百个字符的长度固定的句段，然后单独处理每个句段。
	这引入了两个关键限制：

	1 算法无法对超过固定长度的依赖关系建模
	2 句段通常不遵循句子边界，从而造成上下文碎片化，导致优化效率低下。在长程依赖性不成其为问题的短序列中，这种做法尤其令人烦扰
	为了解决这些限制，我们推出了 Transformer-XL，
 
### Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context


## 预训练模型 pretrain model

预训练模型引起了很多人的关注。
多义词问题：

多义词是自然语言中经常出现的现象，也是语言灵活性和高效性的一种体现。多义词对Word Embedding来说有什么负面影响？
多义词Bank，有两个常用含义，但是Word Embedding在对bank这个单词进行编码的时候，是区分不开这两个含义的，因为它们尽管上下文环境中出现的单词不同，但是在用语言模型训练的时候，不论什么上下文的句子经过word2vec，都是预测相同的单词bank，而同一个单词占的是同一行的参数空间，这导致两种不同的上下文信息都会编码到相同的word embedding空间里去。所以word embedding无法区分多义词的不同语义，这就是它的一个比较严重的问题。

### ELMo

ELMo可以根据上下文体现它唯一的表征。

ELMo “Embedding from Language Models” NACCL2018 最佳论文

“Deep contextualized word representation”

从左到右对句子编码，也可以从右到左对句子编码，每一层对应的节点并起来，就形成了当前这个词在上下文的语义表示。用的时候就用这个语义加上词本身的词嵌入，来做后续的任务，性能便得到相应的提高。

在此之前的Word Embedding本质上是个静态的方式，所谓静态指的是训练好之后每个单词的表达就固定住了，以后使用的时候，不论新句子上下文单词是什么，这个单词的Word Embedding不会跟着上下文场景的变化而改变，

ELMO的本质思想是：我事先用语言模型学好一个单词的Word Embedding，此时多义词无法区分，不过这没关系。在我实际使用Word Embedding的时候，单词已经具备了特定的上下文了，这个时候我可以根据上下文单词的语义去调整单词的Word Embedding表示，这样经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以ELMO本身是个根据当前上下文对Word Embedding动态调整的思路。



ELMO采用了典型的两阶段过程，第一个阶段是利用语言模型进行预训练；第二个阶段是在做下游任务时，从预训练网络中提取对应单词的网络各层的Word Embedding作为新特征补充到下游任务中。



ELMO有什么值得改进的缺点呢？首先，一个非常明显的缺点在特征抽取器选择方面，ELMO使用了LSTM而不是新贵Transformer，Transformer是谷歌在17年做机器翻译任务的“Attention is all you need”的论文中提出的，引起了相当大的反响，很多研究已经证明了Transformer提取特征的能力是要远强于LSTM的。如果ELMO采取Transformer作为特征提取器，那么估计Bert的反响远不如现在的这种火爆场面。另外一点，ELMO采取双向拼接这种融合特征的能力可能比Bert一体化的融合特征方式弱，

https://blog.csdn.net/malefactor/article/details/83961886


### GPT “Generative Pre-Training”
如果把ELMO这种预训练方法和图像领域的预训练方法对比，发现两者模式看上去还是有很大差异的。除了以ELMO为代表的这种基于特征融合的预训练方法外，NLP里还有一种典型做法，这种做法和图像领域的方式就是看上去一致的了，一般将这种方法称为“基于Fine-tuning的模式”，而GPT就是这一模式的典型开创者。

GPT也采用两阶段过程，第一个阶段是利用语言模型进行预训练，第二阶段通过Fine-tuning的模式解决下游任务。

![gpt](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/nlppic/gpt.png)

### BERT
Bert采用和GPT完全相同的两阶段模型，首先是语言模型预训练；其次是使用Fine-Tuning模式解决下游任务。和GPT的最主要不同在于在预训练阶段采用了类似ELMO的双向语言模型，当然另外一点是语言模型的数据规模要比GPT大。

它用左边、右边的信息来预测最外部的词的信息，同时它也可以判断下一句是真的下一句还是伪造的下一句，用两种方式对句子每一个词进行编码，得到的训练结果就表征了这个词在上下文中的语义表示。基于这样的语义表示，就可以判断两个句子的关系，比如说是不是附属关系，判断一个句子的分类（例如Q&A中，判断回答对应的边界是不是对应提问），以及对输入的每一个词做一个标注，结果就得到一个词性标注。

 https://arxiv.org/pdf/1810.04805.pdf

 https://jalammar.github.io/illustrated-bert/



### GPT2  

Language Models are Unsupervised Multitask Learners
数据集 WebText

GPT-2 是 GPT 的升级版本，其最大的区别在于模型规模更大，训练数据更多，GPT 是12层的 Transformer，BERTLARGE 是24层的 Transformer，GPT-2 则为48层单向 Transformer，共有15亿个参数。

GPT-2 共有四个型号，如下图所示。「小号」的 GPT-2 模型堆叠了 12 层，「中号」24 层，「大号」36 层，还有一个「特大号」堆叠了整整 48 层。

在预训练阶段，GPT-2 采用了多任务的方式，每个任务都要保证其损失函数能收敛，不同的任务共享主体 Transformer 参数，该方案借鉴微软的 MT-DNN，这样能进一步的提升模型的泛化能力，因此即使在无监督 – 没有微调的情况下依旧有非常不错的表现。

这就是GPT-2的主要改进点，总结一下，多任务预训练+超大数据集+超大规模模型，

![gpt2](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/nlppic/gpt2.png)

![vs](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/nlppic/vs.png)
https://blog.csdn.net/weixin_38937984/article/details/101759331


中文预训练 GPT-2 项目，它开源了预训练结果与 Colab Demo 演示，只需要单击三次，我们就能生成定制的中文故事。

该 15 亿参数量的 GPT-2 中文预训练模型在 15GB 的纯文本上进行训练，一共迭代了 10 万步。这 15GB 的纯文本主要选自 THUCNews 与 nlp_chinese_corpus，它们会做一系列的数据清理。

THUCNews：http://thuctc.thunlp.org/#中文文本分类数据集THUCNews

nlp_chinese_corpus：https://github.com/brightmart/nlp_chinese_corpus

项目地址：https://github.com/imcaspar/gpt2-ml

Colab 演示地址：https://colab.research.google.com/github/imcaspar/gpt2-ml/blob/master/pretrained_model_demo.ipynb

项目作者开放的预训练模型是在 TPU Pod v3-256 上复现的 15 亿参数 GPT2，这也是 GitHub 上第一个支持大规模 TPU 训练的中文 GPT-2 项目。

本项目的训练脚本：https://github.com/imcaspar/gpt2-ml/tree/master/train

XLNET，以及UNILM、MASS、MT-DNN、XLM，都是基于这种思路的扩充，解决相应的任务各有所长。其中微软研究院的UNILM可同时训练得到类似BERT和GPT的模型，而微软MASS采用encoder-decoder训练在机器翻译上效果比较好。还有MT-DNN强调用多任务学习预训练模型，而XLM学习多语言BERT模型，在跨语言迁移学习方面应用效果显著。

![pretrain](https://github.com/weslynn/graphic-deep-neural-network/blob/master/pic/nlppic/pretrain.png)


现在由于这种预训练模型大行其道，人们在思考，自然语言处理是不是应该改换一种新的模态。过去我们都说用基于知识的方法来充实当前的输入，但是过去都没有做到特别好，而这种新的预训练模型给我们带来一个新的启发：

我们可以针对大规模的语料，提前训练好一个模型，这个模型既代表了语言的结构信息，也有可能代表了所在领域甚至常识的信息，只不过我们看不懂。加上我们未来的预定的任务，这个任务只有很小的训练样本，把通过大训练样本得到的预训练模型，做到小训练样本上，效果就得到了非常好的提升。

[自然语言处理的未来之路](https://www.leiphone.com/news/201907/djMxwOkOO5u4sf6O.html)



### T5 Google

T5（Text-to-Text Transfer Transformer）

《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》中，谷歌提出预训练模型 T5，参数量达到了 110 亿，再次刷新 Glue 榜单，成为全新的 NLP SOTA 预训练模型。

论文链接：https://arxiv.org/abs/1910.10683
Github 链接：https://github.com/google-research/text-to-text-transfer-transformer


标签系统

WSABIE: Scaling Up To Large Vocabulary Image Annotation (http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)



Datasets:
google发表新的数据集PAWS 和PAWS-X
该语料库包含六种不同语言的 PAWS 示例翻译，包含：法语、西班牙语、德语、汉语、日语和韩语。详情可通过这里查看（https://github.com/google-research-datasets/paws/tree/master/pawsx）
数据集下载地址：

https://github.com/google-research-datasets/paws

原文链接：

https://ai.googleblog.com/2019/10/releasing-paws-and-paws-x-two-new.htm

「Colossal Clean Crawled Corpus」（或简称 C4 语料库）


movielens 

https://grouplens.org/datasets/movielens/