随深度学习技术的发展，使用**循环神经网络（Recurrent Neural Network，RNN）**建立的各种序列模型，使语音识别、机器翻译及自然语言理解等应用成为可能。

### 表示与类型

自然语言、音频等数据都是前后相互关联的数据，比如理解一句话要通过一整句而不是其中的几个词，分析一段音频要通过连续的片段而不是其中的几帧。前面所学的DNN以及CNN处理的都是前后毫无关联的一个个单独数据，对于这些序列数据则需要采用RNN进行处理。

![序列](https://ws1.sinaimg.cn/large/82e16446gy1fop0tdxehdj21250i343a.jpg)

用循环神经网络处理时间序列时，首先要对时间序列进行标记化。对一个序列数据$x$，用符号$x^{\langle t\rangle}$来表示这个序列数据中的第$t$个元素。这个元素的类型因数据类型而异，对一段音频，它可能其中的几帧，对一句话，它可能是一个或几个单词，如下图所示。

![Harry Potter](https://ws1.sinaimg.cn/large/82e16446gy1fomosg9szjj210205ft8u.jpg)

第$i$个序列数据的第$t$个元素用符号$x^{(i)\langle t\rangle}$表示，其标签用符号$y^{(i)\langle t\rangle}$表示。

序列中的每个元素有相应的标签，一般需要先建立一个包含序列中所有类型的元素的**字典（Dictionary）**。例如对上图中的句子，建立一个含有10000个单词的列向量词典，单词顺序以A~Z排列，然后根据单词在列向量中的位置，用one—hot向量来表示该单词的标签，部分表示如下图：

![字典](https://ws1.sinaimg.cn/large/82e16446gy1foo6h6dbpdj20vz0ft0uj.jpg)

标记化完成后，将训练数据输入网络中。一种循环神经网络的结构如下图：
![RNN结构](https://ws1.sinaimg.cn/large/82e16446gy1foocscndavj20vy0c0aaw.jpg)

左边的网络可简单表示为右图的结构，其中元素$x^{\langle t\rangle}$输入对应时间步（TIme Step）的隐藏层的同时，该隐藏层也会接收上一时间步的隐藏层激活$a^{\langle t-1\rangle}$，其中$a^{\langle 0\rangle}$一般直接初始化为零向量。一个时间步输出一个对应的预测结果${\hat y}^{\langle t\rangle}$，输入、激活、输出有对应的参数$W\_{ax}$、$W\_{aa}$、$W\_{y}$。

以上结构的前向传播过程，有：$$a^{\langle 0\rangle} = \vec 0$$ $$a^{\langle t\rangle} = g\_1(W\_{aa} a^{\langle t-1\rangle} + W\_{ax} x^{\langle t\rangle} + b\_a)$$ $${\hat y}^{\langle t\rangle} = g\_2(W\_{y} a^{\langle t\rangle} + b\_y)$$

其中$b\_a$、$b\_y$是两个偏差参数，激活函数$g\_1$通常选择tanh，有时也用ReLU，$g\_2$的选择取决于需要的输出类型，可选sigmoid或Softmax。

具体计算中以上的式子还可以进一步简化，以方便运算。将$W\_{ax}$和$W\_{aa}$堆叠成一个矩阵$W\_a$，$a^{\langle t-1\rangle}$和$x^{\langle t\rangle}$也堆叠成一个矩阵，有：$$ W\_a = [W\_{ax}, W\_{aa}] $$ $$a^{\langle t\rangle} = g\_1(W\_{a}[a^{\langle t-1\rangle},x^{\langle t\rangle}] + b\_a)$$

反向传播的过程类似于深度神经网络，如下图所示：
![RNN反向传播](https://ws1.sinaimg.cn/large/82e16446gy1fopmw4dzzqj21ba0jatch.jpg)

这种结构的一个缺陷是，某一时刻的预测结果仅使用了该时刻之前输入的序列信息。根据所需的输入及输出数量，循环神经网络可分为“一对一”、“多对一”、“多对多”等结构：

![类型](https://ws1.sinaimg.cn/large/82e16446gy1foozkxloctj211q0h73zq.jpg)

这些网络结构可在不同的领域中得到应用。

### RNN应用：语言模型

**语言模型（Language Model）**是根据语言客观事实而进行的语言抽象数学建模。例如对一个语音识别系统，输入的一段语音可能表示下面两句话：
![English](https://ws1.sinaimg.cn/large/82e16446gy1fop15b67t7j20fe04gdfs.jpg)

其中的“pair”和“pear”读音相近，但是在日常表达及语法上显然这段语音是第二句的可能性要大，要使这个语音识别系统能够准确判断出第二句话为正确结果，就需要语言模型的支持。这个语言模型能够分别计算出语音表示以上两句话的概率，以此为依据做出判断。

建立语言模型所采用的训练集是一个大型的**语料库（Corpus）**。建立过程中，如之前所述，需要先建立一个字典，之后将语料库中每个词表示为对应的one-hot向量。此外需要额外定义一个标记EOS（End Of Sentence）表示一个句子的结尾，也可以将其中的标点符号加入字典后也用one=hot向量表示。对于语料库中部分（英语）人名、地名等特殊的不包含在字典中的词汇，可在词典中加入再用一个UNK（Unique Token）标记来表示。

将标志化后的训练集输入网络中的训练过程，如下例所示：

![语言模型](https://ws1.sinaimg.cn/large/82e16446gy1fop5r1t9stj20sj0e2gm9.jpg)

第一个时间步中输入的$a^{\langle 0\rangle}$和$x^{\langle 1\rangle}$都是零向量，${\hat y}^{\langle 1\rangle}$是通过softmax预测出的字典中每一个词作为第一个词出现的概率；第二个时间步中输入的$x^{\langle 2\rangle}$是下面的训练数据中第一个单词“cats”的标签$y^{\langle 1\rangle}$和上一层的激活$a^{\langle 1\rangle}$,输出的$y^{\langle 2\rangle}$表示的是单词“cats”后面出现字典中的其他词，特别是“average”的条件概率。后面的时间步与第二步类似，到最后就可以得到整个句子出现的概率。

这样，损失函数将表示为：$$ \mathcal{L}({\hat y}^{\langle t\rangle},y^{\langle t\rangle})=-\sum\_t y^{\langle t\rangle}\_i log\ {\hat y}^{\langle t\rangle}$$

成本函数表示为：$$ \mathcal{J} = \sum_t \mathcal{L}^{\langle t\rangle}({\hat y}^{\langle t\rangle},y^{\langle t\rangle})$$

训练好一个这个语言模型后，可通过**采样（Sample）**新的序列，来了解这个模型中都学习到了一些什么。从模型中采样出新序列的过程如下：
![采样新序列](https://ws1.sinaimg.cn/large/82e16446gy1fop90xh6rvj20ti0aht94.jpg)

第一个时间步中输入的$a^{\langle 0\rangle}$和$x^{\langle 1\rangle}$还是零向量，依据softmax预测出的字典中每一个词作为第一个词出现的概率，选取一个词${\hat y}^{\langle 1\rangle}$作为第二个时间步的输入。后面与此类似，模型将自动生成一些句子，从这些句子中可发现模型通过语料库学习到的知识。

以上是基于词汇构建的语言模型，也就是所用的字典中包含的是一个个单词。实际应用中，还可以构建基于字符的语言模型，不过这种方法的结果中将得到过多过长的序列，计算成本也较大，在当今的NLP领域也用得较少。

### GRU与LSTM

如下图中的句子时，后面的动词用“was”还是“were”取决于前面的名词“cat”是单数还是复数。
![grammer](https://ws1.sinaimg.cn/large/82e16446gy1fopdh23y3rj20q004nq32.jpg)
一般的循环神经网络不擅于捕捉这种序列中存在的长期依赖关系，其中的原因是，一般的循环神经网络也会出现类似于深度神经网络中的梯度消失问题，而使后面输入的序列难以受到早先输入序列的影响。梯度爆炸的问题也会出现，不过可以采用**梯度修剪（Gradient Clipping）**应对，相比之下梯度消失问题更难以解决。

#### GRU

**GRU（Gated Recurrent Units, 门控循环单元）**网络改进了循环神经网络的隐藏层，从而使梯度消失的问题得到改善。GRU的结构如下图：
![GRU](https://ws1.sinaimg.cn/large/82e16446gy1foqshsehx4j20pr0c6mxg.jpg)

其中的$c$代表**记忆细胞（Memory Cell）**，用它来“记住”类似前面例子中“cat”的单复数形式，且这里的记忆细胞$c^{\langle t\rangle}$直接等于输出的激活$a^{\langle t\rangle}$；$\tilde{c}$代表下一个$c$的候选值；$\Gamma_u$代表**更新门（Update Gate）**，用它来控制记忆细胞的更新与否。上述结构的具体表达式有：$$\tilde{c}^{\langle t \rangle} = \tanh(W_c[c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b\_c)$$ $$\Gamma_u = \sigma(W_u[c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b\_u)$$ $$c^{\langle t \rangle} = \Gamma_u \times \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) \times c^{\langle t-1 \rangle}$$ $$a^{\langle t\rangle} = c^{\langle t\rangle}$$
$\tilde{c}$的计算中以tanh作为激活函数，使用simgoid作为激活函数得到的$\Gamma_u$值将在0到1的范围内。当$\Gamma_u=1$时，输出的$c$值被更新为$\tilde{c}$，否者保持为输入的$c$值。

上面所述的是简化后的GRU，完整的GRU结构如下：
![GRU-FULL](https://ws1.sinaimg.cn/large/82e16446gy1forb3vh4ujj20r80ddaah.jpg)
其中**相关门（Relevance Gate）**$\Gamma_r$表示上一个$c$值与下一个$c$的候选值的相关性。与简化的GRU相比，表达式发生如下变化：$$\Gamma_r = \sigma(W_r[c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b\_r)$$ $$\tilde{c}^{\langle t \rangle} = \tanh(W_c[\Gamma_r \times c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b\_c)$$
GRU其实只是一种LSTM的流行变体，其相关概念来自于2014年Cho等人发表的论文[[On the properties of neural machine translation: Encoder-decoder approaches]](https://arxiv.org/pdf/1409.1259.pdf)以及Chung等人的[[Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling]](https://arxiv.org/pdf/1412.3555.pdf)。

#### LSTM

1997年Hochreiter和Schmidhuber共同在论文[[Long short-term memory ]](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)中提出的**LSTM（Long Short Term Memory，长短期记忆）**网络比GRU更加通用及强大，其结构如下：
![LSTM](https://ws1.sinaimg.cn/large/82e16446gy1foqshutgv1j20ok0byq3b.jpg)

相比之前的简化版GRU，LSTM中多了**遗忘门（Forget Gate）**$\Gamma_f$和**输出门（Output Gate）**$\Gamma_o$，具体表达式如下：$$\tilde{c}^{\langle t \rangle} = \tanh(W_c[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b\_c)$$ $$\Gamma_u = \sigma(W_u[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b\_u)$$ $$\Gamma_f = \sigma(W_f[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f)$$ $$\Gamma_o=  \sigma(W_o[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o)$$ $$c^{\langle t \rangle} = \Gamma_f^{\langle t \rangle} \times c^{\langle t-1 \rangle} + \Gamma_u^{\langle t \rangle} \times \tilde{c}^{\langle t \rangle}$$ $$a^{\langle t \rangle} = \Gamma_o^{\langle t \rangle}\times \tanh(c^{\langle t \rangle})$$

更为常用的LSTM版本中，几个门值的计算不只取决于输入$x$和$a$值，有时也可以偷窥上一个细胞输入的$c$值，这叫**窥视孔连接（Peephole Connection)**。

多个LSTM单元连接在一起，形成一个LSTM网络：
![LSTM网络](https://ws1.sinaimg.cn/large/82e16446gy1fordgphswvj21330b8aan.jpg)

### BRNN与DRNN

前面介绍的循环神经网络在结构上都是单向的，也提到过它们具有某一时刻的预测结果仅使用了该时刻之前输入的序列信息的缺陷，而**双向循环神经网络（Bidirectional RNN）**弥补了这一缺陷。BRNN的结构图如下所示：
![BRNN](https://ws1.sinaimg.cn/large/82e16446gy1forgaw5bd4j21080codje.jpg)

此外，循环神经网络的每个时间步上也可以包含多个隐藏层，形成**深度循环神经网络（Deep RNN)**，如下图：
![DRNN](https://ws1.sinaimg.cn/large/82e16446gy1forgavws5jj20tq0da0wb.jpg)

**自然语言处理（Natural Language Processing，NLP)**是人工智能和语言学领域的学科分支，它研究实现人与计算机之间使用自然语言进行有效通信的各种理论和方法。

### 词嵌入

前面介绍过，处理文本序列时，通常用建立字典后以**one-hot**的形式表示某个词，进而表示某个句子的方法。这种表示方法孤立了每个词，无法表现各个词之间的相关性，满足不了NLP的要求。

**词嵌入（Word Embedding）**是NLP中语言模型与表征学习技术的统称，概念上而言，它是指把一个维数为所有词的数量的高维空间（one-hot形式表示的词）“嵌入”到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量。
![Word Embedding](https://ws1.sinaimg.cn/large/82e16446gy1foybku0afyj20t4099jvh.jpg)
如上图中，各列分别组成的向量是词嵌入后获得的第一行中几个词的词向量的一部分。这些向量中的值，可代表该词与第一列中几个词的相关程度。

使用2008年van der Maaten和Hinton在论文[[Visualizing Data using t-SNE](https://www.seas.harvard.edu/courses/cs281/papers/tsne.pdf)]中提出的**t-SNE**数据可视化算法，将词嵌入后获得的一些词向量进行非线性降维，可到下面的映射结果：
![t-SNE映射](https://ws1.sinaimg.cn/large/82e16446gy1foyjlkuge6j20lt0apwfb.jpg)
其中可发现，各词根据它们的语义及相关程度，分别汇聚在了一起。

对大量词汇进行词嵌入后获得的词向量，可用来完成**命名实体识别（Named Entity Recognition)**等任务。其中可充分结合迁移学习，以降低学习成本，提高效率。

好比前面讲过的用Siamese网络进行人脸识别过程，使用词嵌入方法获得的词向量可实现词汇的类比及相似度度量。例如给定对应关系“男性（Man）”对“女性（Woman）”，要求机器类比出“国王（King）”对应的词汇，通过上面的表格，可发现词向量存在数学关系“Man - Woman $\approx$ King - Queen”，也可以从可视化结果中看出“男性（Man）”到“女性（女性）”的向量与“国王（King）”到“王后（Queen）”的向量相似。词嵌入具有的这种特性，在2013年Mikolov等发表的论文[[Linguistic Regularities in Continuous Space Word Representations](http://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf)]中提出，成为词嵌入领域具有显著影响力的研究成果。

上述思想可写成一个余弦（cos）相似度函数：$$ sim(u, v) = \frac{u^T v}{\mid\mid u \mid\mid_2 \mid\mid v \mid\mid_2} $$以此度量词向量的相似度。

### 词嵌入方法

词嵌入的方法包括人工神经网络、对词语同现矩阵降维、概率模型以及单词所在上下文的显式表示等。以词汇的one-hot形式作为输入，不同的词嵌入方法能以不同的方式学习到一个**嵌入矩阵（Embedding Matrix）**，最后输出某个词的词向量。

将字典中位置为$i$的词以one-hot形式表示为$o_i$，嵌入矩阵用$E$表示，词嵌入后生成的词向量用$e_i$表示，则三者存在数学关系：$$E \cdot o_i = e_i$$

例如字典中包含10000个词，每个词的one-hot形式就是个大小为$10000 \times 1$的列向量，采用某种方法学习到的嵌入矩阵大小为$300 \times 10000$的话，将生成大小为$300 \times 1$的词向量。

#### 神经概率语言模型

采用神经网络建立语言模型是学习词嵌入的有效方法之一。2003年Bengio等人的经典之作[[A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)]中，提出的**神经概率语言模型**，是早期最成功的词嵌入方法之一。

模型中，构建了了一个能够通过上下文来预测未知词的神经网络，在训练这个语言模型的同时学习词嵌入。例如将下图中上面的句子作为下面的神经网络的输入：

![语言模型](https://ws1.sinaimg.cn/large/82e16446ly1fp36354vmlj20to0erwm7.jpg)

经过隐藏层后，最后经Softmax将输出预测结果。其中的嵌入矩阵$E$与$w$、$b$一样，是该网络中的参数，需通过训练得到。训练过程中取语料库中的某些词作为目标词，以目标词的部分上下文作为输入，训练网络输出的预测结果为目标词。得到了嵌入矩阵，就能通过前面所述的数学关系式求得词嵌入后的词向量。

#### Word2Vec

**Word2Vec（Word To Vectors）**是现在最常用、最流行的词嵌入算法，它由2013年由Mikolov等人在论文[[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)]中提出。

Word2Vec中的**Skip-Gram模型**，所做的是在语料库中选定某个词（Context），随后在该词的正负10个词距内取一些目标词（Target）与之配对，构造一个用Context预测输出为Target的监督学习问题，训练一个如下图结构的网络：
![Skip-Gram网络](https://ws1.sinaimg.cn/large/82e16446ly1fp4828eo4aj20up0dodhk.jpg)
该网络仅有一个Softmax单元，输出Context下Target出现的条件概率：$$p(t \mid c) = \frac{exp(\theta\_t^T e\_c)}{\sum_{j=1}^m exp(\theta\_j^T e\_c)} $$
上式中$\theta\_t$是一个与输出的Target有关的参数，其中省略了用以纠正偏差的参数。训练过程中还是用交叉熵损失函数。

选定的Context是常见或不常见的词将影响到训练结果，在实际中，Context并不是单纯地通过在语料库均匀随机采样得到，而是采用了一些策略来平衡选择。

Word2Vec中还有一种**CBOW（Continuous Bag-of-Words Model）模型**，它的工作方式是采样上下文中的词来预测中间的词，与Skip-Gram相反。

以上方法的Softmax单元中产生的计算量往往过大，改进方法之一是使用**分级Softmax分类器（Hierarchical Softmax Classifier）**，采用**霍夫曼树（Huffman Tree）**来代替隐藏层到输出Softmax层的映射。

此外，Word2Vec的作者在后续论文[[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)]中提出了**负采样（Negative Sampling）**模型，进一步改进和简化了词嵌入方法。

负采样模型中构造了一个预测给定的单词是否为一对Context-Target的新监督学习问题，采用的网络结构和前面类似：
![负采样](https://ws1.sinaimg.cn/large/82e16446ly1fp5easo665j20p10c1tae.jpg)
训练过程中，从语料库中选定Context，输入的词为一对Context-Target，则标签设置为1。另外任取$k$对非Context-Target，作为负样本，标签设置为0。只有较少的训练数据，$k$的值取5~20的话，能达到比较好的效果；拥有大量训练数据，$k$的取值取2~5较为合适。

原网络中的Softmax变成多个Sigmoid单元，输出Context-Target（c,t）对为正样本（$ y = 1 $)的概率：$$p(y = 1 \mid c, t) = \sigma(\theta\_t^T e\_c) $$
其中的$\theta\_t$、$e\_c$分别代表Target及Context的词向量。通过这种方法将之前的一个复杂的多分类问题变成了多个简单的二分类问题，而降低计算成本。

模型中还包含了对负样本的采样算法。从本质上来说，选择某个单词来作为负样本的概率取决于它出现频率，对于更经常出现的单词，将更倾向于选择它为负样本，但这样会导致一些极端的情况。模型中采用一下公式来计算选择某个词作为负样本的概率：$$p(w\_i) = \frac{f(w\_i)^{\frac{3}{4}}}{\sum\_{j=0}^m f(w\_j)^{\frac{3}{4}}} $$
其中$f(w\_i)$代表语料库中单词$w\_i$出现的频率。

#### GloVe

**GloVe（Global Vectors）**是另一种现在流行的词嵌入算法,它在2014年由Pennington等人在论文[[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)]中提出。

Glove模型中，首先基于语料库统计了词的**共现矩阵**$X$，$X$中的元素为$X\_{i,j}$，表示整个语料库中单词$i$和单词$j$彼此接近的频率，也就是它们共同出现在一个窗口中的次数。之后要做的，就是优化以下代价函数：$$J=\sum\_{i,j}^N f(X\_{i,j})(\theta\_i^T e\_j + b\_i + b\_j - log(X\_{i,j}))^2$$
其中$\theta\_i$、$e\_j$分是单词$i$和单词$j$的词向量，$b\_i$、$b\_j$是两个偏差项，$f()$是一个用以防止$X\_{i,j} = 0$时$log(X_{i,j})$无解的权重函数，词汇表的大小为$N$。

（以上优化函数的推导过程见参考资料中的“理解GloVe模型”）

最后要说明的是，使用各种词嵌入方法学习到的词向量，并不像最开始介绍词嵌入时展示的表格中Man、Woman、King、Queen的词向量那样，其中的值能够代表着与Gender、Royal等词的的相关程度，实际上它们大都超出了人们的能够理解范围。

### 词嵌入应用：情感分类器

NLP中的情感分类，是对某段文字中所表达的情感做出分类，它能在很多个方面得到应用。训练情感分类模型时，面临的挑战之一可能是标记好的训练数据不够多。然而有了词嵌入得到的词向量，只需要中等数量的标记好的训练数据，就能构建出一个表现出色的情感分类器。
![情感分类](https://ws1.sinaimg.cn/large/82e16446ly1fp6vs0prpxj20sw0czn1z.jpg)

如上图，要训练一个将左边的餐厅评价转换为右边评价所属星级的情感分类器，也就是实现$x$到$y$的映射。有了用词嵌入方法获得的嵌入矩阵$E$，一种简单的实现方法如下：

![简单方法](https://ws1.sinaimg.cn/large/82e16446ly1fp6vs1qlmrj20u50dp0wq.jpg)

方法中计算出句中每个单词的词向量后，取这些词向量的平均值输入一个Softmax单元，输出预测结果。这种简单的方法适用于任何长度的评价，但忽略了词的顺序，对于某些包含多个正面评价词的负面评价，很容易预测到错误结果。

采用RNN能实现一个表现更加出色的情感分类器，此时构建的模型如下：
![RNN情感分类](https://ws1.sinaimg.cn/large/82e16446ly1fp6vs331arj20sb0dcq5b.jpg)

这是一个“多对一”结构的循环神经网络，每个词的词向量作为网络的输入，由Softmax输出结果。由于词向量是从一个大型的语料库中获得的，这种方法将保证了词的顺序的同时能够对一些词作出泛化。

### 词嵌入除偏

在词嵌入过程中所使用的语料库中，往往会存在一些性别、种族、年龄、性取向等方面的偏见，从而导致获得的词向量中也包含这些偏见。比如使用未除偏的词嵌入结果进行词汇类比时，“男性（Man）”对“程序员（Computer Programmer）”将得到类似“女性（Woman）”对“家务料理人（Homemaker）”的性别偏见结果。2016年Bolukbasi等人在论文[[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/pdf/1607.06520.pdf)]中提出了一些消除词嵌入中的偏见的方法。

这里列举消除词向量存在的性别偏见的过程，来说明这些方法。（摘自第二周课后作业）

**1.中和本身与性别无关词汇**

**中和（Neutralize）**“医生（doctor）”、“老师（teacher）”、“接待员（receptionist）”等本身与性别无关词汇中的偏见，首先计算$g = e\_{woman}-e\_{man}$，用“女性（woman）”的词向量减去“男性（man）”的词向量，得到的向量$g$就代表了“性别（gender）”。假设现有的词向量维数为50，那么对某个词向量，将50维空间分成两个部分：与性别相关的方向$g$和与$g$**正交**的其他49个维度$g\_{\perp}$。如下左图：
![本身与性别无关](https://ws1.sinaimg.cn/large/82e16446ly1fp6xfmtnerj21980h4q5d.jpg)
除偏的步骤，是将要除偏的词向量，左图中的$e\_{receptionist}$，在向量$g$方向上的值置为$0$，变成右图所示的$e\_{receptionist}^{debiased}$。所用的公式如下:
$$e^{bias}\_{component} = \frac{e \cdot g}{||g||\_2^2} \times g$$ $$e\_{receptionist}^{debiased} = e - e^{bias}\_{component}$$

**2.均衡本身与性别有关词汇**

对“男演员（actor）”、“女演员（actress）”、“爷爷（grandfather）”等本身与性别有关词汇，如下左图，假设“女演员（actress）”的词向量比“男演员（actor）”更靠近于“婴儿看护人（babysit）”。中和“婴儿看护人（babysit）”中存在的性别偏见后，还是无法保证它到“女演员（actress）”与到“男演员（actor）”的距离相等。对一对这样的词，除偏的过程是**均衡（Equalization）**它们的性别属性。
![本身与性别有关](https://ws1.sinaimg.cn/large/82e16446ly1fp6xfm6hncj21fk0m2jue.jpg)

均衡过程的核心思想是确保一对词（actor和actress）到$g\_{\perp}$的距离相等的同时，也确保了它们到除偏后的某个词（babysit）的距离相等，如上右图。

对需要除偏的一对词$w1$、$w2$，选定与它们相关的某个未中和偏见的单词$B$之后，均衡偏见的过程如下公式：$$\mu = \frac{e\_{w1} + e\_{w2}}{2}$$ $$\mu_{B} = \frac {\mu \cdot \text{bias\_axis}}{||\text{bias_axis}||\_2^2} \times \text{bias\_axis}$$ $$\mu\_{\perp} = \mu - \mu\_{B}$$ $$e\_{w1B} = \frac {e\_{w1} \cdot \text{bias\_axis}}{||\text{bias\_axis}||\_2^2} \times \text{bias\_axis}$$ $$e\_{w2B} = \frac {e\_{w2} \cdot \text{bias\_axis}}{||\text{bias\_axis}||\_2^2} \times \text{bias\_axis}$$ $$e\_{w1B}^{corrected} = \sqrt{ |{1 - ||\mu\_{\perp} ||^2\_2} |} \times \frac{e\_{\text{w1B}} - \mu\_B} {||(e\_{w1} - \mu\_{\perp}) - \mu\_B)||\_2} $$ $$e\_{w2B}^{corrected} = \sqrt{ |{1 - ||\mu\_{\perp} ||^2\_2} |} \times \frac{e\_{\text{w2B}} - \mu\_B} {||(e\_{w1} - \mu\_{\perp}) - \mu\_B)||\_2} $$ $$e\_1 = e\_{w1B}^{corrected} + \mu\_{\perp}$$ $$e\_2 = e\_{w2B}^{corrected} + \mu\_{\perp}$$

采用循环神经网络能够建立各种各样的**序列模型（Sequence Model）**。加入一些注意力机制，能够使这些序列模型更加强大。

<!--more-->
### Seq2Seq模型

2014年Cho等人在论文[[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)]中首次提出了**Seq2Seq（Sequence-to-Sequence）**模型。从机器翻译到语音识别，这种模型能够在各种序列到序列的转换问题中得到应用。

一个Seq2Seq模型中可分成**编码器（Encoder）**和**译码器（Decoder）**两部分，它们通常是两个不同的神经网络。如下图是谷歌机器翻译团队的Sutskever等人2014年在论文[[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)]中提出的机器翻译模型：
![机器翻译](https://ws1.sinaimg.cn/large/82e16446ly1fp7j1jnp97j20nj0fegmc.jpg)

上面法语中的词依次输入作为编码器的RNN的时间步中，这个RNN可以是GRU或LSTM。将编码器的最后输出作为译码器的输入，译码器也是一个RNN，训练译码器作出正确的翻译结果。

这种Enconder-Decoder的结构，也可以应用在**图像标注（Image Caption）**上。2014年百度研究所的毛俊骅等人在论文[[DDeep Captioning with Multimodal Recurrent Neural Networks (m-RNN)](https://arxiv.org/pdf/1412.6632.pdf)]中提出了如下图中的结构：
![图像标注](https://ws1.sinaimg.cn/large/82e16446ly1fp7jql4cu4j21140g30wc.jpg)

上面将图像输入了一个作为编码器的AlexNet结构的CNN中，最后的Softmax换成一个RNN作为译码器，训练网络输出图像的标注结果。

另外两篇论文[[Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)]和 [[Deep Visual-Semantic Alignments for Generating Image Descriptions](https://arxiv.org/pdf/1412.2306.pdf)]中， 也提到了这种结构。

机器翻译用到的Seq2Seq模型中，译码器所做的工作与前面讲过的语言模型的采样过程类似，只不过在机器翻译中，用编码器的输出代替语言模型中的$0$作为译码器中第一个时间步的输入，如下图所示：
![类比](https://ws1.sinaimg.cn/large/82e16446ly1fp7lf9mdt6j20wo0cz750.jpg)

换句话说，机器翻译其实是一个输入为法语的条件下，希望以对应的英语作为输出的语言模型，所以机器翻译的过程也就相当于建立一个**条件语言模型（Conditional Language Model)**的过程。

采用大量的数据训练好一个机器翻译系统后，对于一个相同的句子，由于译码器进行的是随机采样过程，输出的可能会是多种或好或坏的结果：
![多种结果](https://ws1.sinaimg.cn/large/82e16446ly1fp7r6eq0cvj20zc0g5myi.jpg)

所以对训练好的机器翻译系统，还需要加入一些算法，使其总是输出最好的翻译结果。

考虑直接使用CS中的**贪心搜索（Greedy Search）**算法，让译码器中每个时间步都取概率最大的那个词，得到的翻译结果还是会不尽人意。

### 集束搜索

#### 步骤

Seq2Seq模型中，译码器的输出结果总是在RNN中采样后得到，造成模型训练完毕后，得到测试的结果参差不齐，**集束搜索（Beam Search）**算法能很好地解决这个问题。这里还是以机器翻译的例子来说明这种算法。

将集束搜索算法运用到机器翻译系统的第一步，是设定一个**束长（Bean Width）**$B$，它代表了译码器中每个时间步的预选单词数量。如下图中$B = 3$，则将第一个时间步中预测出的概率最大的$3$个词作为首词的预选词，同时保存它们的概率值大小$p(y^{\langle 1 \rangle} \mid x)$:
![Step 1](https://ws1.sinaimg.cn/large/82e16446ly1fp7tedr6tsj20u20h074w.jpg)

如果第一步得到的三个预选词分别为“in”、“jane”和“September”，如下图所示，则第二步中，分别将三个预选词作为第一个时间步的预测结果$y^{\langle 1 \rangle}$输入第二个时间步，得到预测结果${\hat y}^{\langle 2 \rangle}$，也就是条件概率值$p({\hat y}^{\langle 2 \rangle} \mid x, y^{\langle 1 \rangle})$:
![Step 2](https://ws1.sinaimg.cn/large/82e16446ly1fp7uohbk26j20wb0jp0ui.jpg)

根据**条件概率**公式，有：$$p(y^{\langle 1 \rangle}, {\hat y}^{\langle 2 \rangle} \mid x) = p(y^{\langle 1 \rangle} \mid x) p({\hat y}^{\langle 2 \rangle} \mid x, y^{\langle 1 \rangle}) $$

分别以三个首词预选词作为$y^{\langle 1 \rangle}$进行计算，将得到$30000$个$p(y^{\langle 1 \rangle}, {\hat y}^{\langle 2 \rangle} \mid x)$。之后还是取其中概率值最大的$B = 3$个，作为对应首词条件下的第二个词的预选词。比如第二个词的预选词分别是“in”为首词条件下的“September”，”jane"为首词条件下的“is”和“visits”，这样的话首词的预选词就只剩下了“in”和“jane”而排除了“September”。后面的过程以此类推，最后将输出一个最优的翻译结果。

#### 优化

总的来说，集束搜索算法所做的工作就是找出符合以下公式的结果：$$ arg\ max\ \prod^{T\_y}\_{t = 1} p(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle},...,y^{\langle t-1 \rangle})$$
然而概率值都是小于$1$的值，多个概率值相乘后的结果的将会是一个极小的浮点值，累积到最后的效果不明显且在一般的计算机上达不到这样的计算精度。改进的方法，是取上式的$log$值并进行标准化：$$ arg\ max\ \frac{1}{T^{\alpha}\_y} \sum^{T\_y}\_{t = 1} log \ p(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle},...,y^{\langle t-1 \rangle})$$
其中的$\alpha$是一个需要根据实际情况进行调节的超参数。

与CS中的精确查找算法--**广度优先查找（Breadth First Search，BFS）**、**深度优先查找（Depth First Search，DFS）**算法不同，集束搜索算法运行的速度虽然很快，但是并不保证能够精确地找到满足$arg\ max\ p(y \mid x)$的结果。

关于束长$B$的取值，较大的$B$值意味着同时考虑了更多的可能，最后的结果也可能会更好，但会带来巨大的计算成本；较小的$B$值减轻了计算成本的同时，也可能会使最后的结果变得糟糕。通常情况下，$B$值取一个$10$以下地值较为合适。还是要根据实际的应用场景，适当地选取。要注意的是，当$B = 1$时，这种算法就和贪心搜索算法没什么两样了。

#### 错误分析

在前面的结构化机器学习项目课程中，已经了解过错误分析。集束搜索是一种**启发式（Heuristic）**搜索算法，它的输出结果不是总为最优的。结合Seq2Seq模型与集束搜索算法构建的机器翻译等系统出错时，差错到底是出现在前者的RNN还是后者的算法中，还是需要通过一些手段，来进行错误分析。
![错误分析](https://ws1.sinaimg.cn/large/82e16446ly1fp80bugqgwj20pe0d4q6i.jpg)

例如对图中的法语，发现机器翻译的结果$\hat y$与专业的人工翻译的结果$y^{\*}$存在较大的差别。要找到错误的根源，首先将翻译没有差别的一部分“Jane visits Africa"分别作为译码器中其三个时间步的输入，得到第四个时间步的输出为“in”的概率$p(y^{\*} \mid x)$和“last”的概率$p(\hat{y} \mid x)$，比较它们的大小并分析：
* 若$p(y^{\*} \mid x) \gt p(\hat{y} \mid x)$，说明是集束搜索时出现错误，没有选择到概率最大的词；
* 若$p(y^{\*} \mid x) \le p(\hat{y} \mid x)$，说明是RNN模型的表现不够好，预测的第四个词为“in”的概率小于“last”。

![分析表格](https://ws1.sinaimg.cn/large/82e16446ly1fp817gxhmdj20tq0c9q5c.jpg)

分析过程中，可以建立一个如上所示的表格，提高错误查找效率。

### 机器翻译评估：BLEU指标

BLEU（Bilingual Evaluation Understudy）是一种用来评估机器翻译质量的指标，它早在2002年由Papineni等人在论文[[BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)]中提出。BLEU的设计思想与评判机器翻译好坏的思想是一致的：机器翻译的结果越接近专业的人工翻译，则评估的分值越高。

最原始的BLEU算法很简单：统计机器翻译结果中的每个单词在参考翻译中出现的次数作为分子，机器翻译结果的总词数作为分母。然而这样得到结果很容易出现错误。
![原始BLEU](https://ws1.sinaimg.cn/large/82e16446ly1fp8mj3du1yj20qj0ciaas.jpg)
如上图的例子中，机器翻译得到的结果是$7$个“the”，分母为$7$，每个“the”都在参考翻译中有出现，分子为$7$，得到翻译精确度为$1.0$，这显然是不对的。改进这种算法，将参考翻译中“the”出现的最高次数作为分子，机器翻译结果中“the”的出现次数作为分母，可得精度为$\frac{2}{7}$。

上面的方法是一个词一个词进行统计，这种以一个单词为单位的集合统称为**uni-gram（一元组）**。以uni-gram统计得到的精度$p\_1$体现了翻译的充分性，也就是逐字逐句地翻译能力。
![Bi-gram](https://ws1.sinaimg.cn/large/82e16446ly1fp8omp7er8j20u007gaad.jpg)
两个单词为单位的集合则称为**bi-gram（二元组）**，例如对以上机器翻译结果（$count$）及参考翻译（$count\_{clip}$）以二元组统计有：

bi-gram |  $count$ | $count\_{clip}$
------- |----------|----------
the cat |    2     |     1
cat the |    1     |     0
cat on  |    1     |     1
on the  |    1     |     1
the mat |    1     |     1
Count   |    6     |     4

根据上表，可得到机器翻译精度为$\frac{4}{6}$。

以此类推，以n个单词为单位的集合为**n-gram（多元组）**，采用n-gram统计的翻译精度$p\_n$的计算公式为：$$p\_n = \frac{\sum\_{\text{n-gram} \in \hat{y}} count\_{clip}(\text{n-gram})}{\sum\_{\text{n-gram} \in \hat{y}} count(\text{n-gram})}$$

以n-gram统计得到的$p\_n$体现了翻译的流畅度。将uni-gram下的$p\_1$到n-gram下的$p\_n$组合起来，对这$N$个值进行**几何加权平均**得到：$$p\_{ave}=exp(\frac{1}{N}\sum\_{i=1}^{N}log^{p\_{n}})$$

此外，注意到采用n-gram时，机器翻译的结果在比参考翻译短的情况下，很容易得到较大的精度值。改进的方法是设置一个**最佳匹配长度（Best Match Length）**，机器翻译的结果未达到该最佳匹配长度时，则需要接受**简短惩罚（Brevity Penalty，BP）**：$$BP = \begin{cases} 1,  & \text{(MT\_length $\ge$ BM\_length)} \\\ exp(1 - \frac{\text{MT\_length}}{\text{BM\_length}}), & \text{(MT\_length $\lt$ BM\_length)} \end{cases}$$
最后得到BLEU指标为：$$ BLEU = BP \times exp(\frac{1}{N}\sum\_{i=1}^{N}log^{p\_{n}}) $$

### 注意力模型

人工翻译一大段文字时，一般都是阅读其中的一小部分后翻译出这一部分，在一小段时间里注意力只能集中在一小段文字上，而很难做到把整段读完后一口气翻译出来。用Seq2Seq模型构建的机器翻译系统中，输出结果的BLEU评分会随着输入序列长度的增加而下降，其中的道理就和这个差不多。

2014年Bahdanau等人在论文[[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)]中提出了**注意力模型（Attention Model）**。最初只是用这种模型对机器翻译作出改进，之后其思想也在其他领域也得到了广泛应用，并成为深度学习领域最有影响力的思想之一。

注意力模型中，网络的示例结构如下所示：
![注意力模型](https://ws1.sinaimg.cn/large/82e16446ly1fp8z5mj8afj20xg0he40a.jpg)

底层是一个双向循环神经网络，需要处理的序列作为它的输入。该网络中每一个时间步的激活$a^{\langle t' \rangle}$中，都包含前向传播产生的和反向传播产生的激活：$$a^{\langle t' \rangle} = ({\overrightarrow a}^{\langle t' \rangle}, {\overleftarrow a}^{\langle t' \rangle})$$

顶层是一个“多对多”结构的循环神经网络，第$t$个时间步以该网络中前一个时间步的激活$s^{\langle t-1 \rangle}$、输出$y^{\langle t-1 \rangle}$以及底层的BRNN中多个时间步的激活$c$作为输入。对第$t$个时间步的输入$c$有：$$c = \sum\_{t'} {\alpha}^{\langle t,t' \rangle}a^{\langle t' \rangle} $$

其中的参数${\alpha}^{\langle t,t' \rangle}$意味着顶层RNN中，第$t$个时间步输出的$y^{\langle t \rangle}$中，把多少“注意力”放在了底层BRNN的第$t'$个时间步的激活$a^{\langle t' \rangle}$上。它总有：$$\sum\_{t'} {\alpha}^{\langle t,t' \rangle} = 1$$

为确保参数${\alpha}^{\langle t,t' \rangle}$满足上式，常用Softmax单元来计算顶层RNN的第$t$个时间步对底层BRNN的第$t'$个时间步的激活的“注意力”：$${\alpha}^{\langle t,t' \rangle} = \frac{exp(e^{\langle t,t' \rangle})}{\sum_{t'=1}^{T\_x} exp(e^{\langle t,t' \rangle})} $$
其中的$e^{\langle t,t' \rangle}$由顶层RNN的激活$s^{\langle t - 1 \rangle}$和底层BRNN的激活$a^{\langle t' \rangle}$一起输入一个隐藏层中得到的，因为$e^{\langle t,t' \rangle}$也就是${\alpha}^{\langle t,t' \rangle}$的值明显与$s^{\langle t \rangle}$、$a^{\langle t' \rangle}$有关，由于$s^{\langle t \rangle}$此时还是未知量，则取上一层的激活$s^{\langle t-1\rangle}$。在无法获知$s^{\langle t-1 \rangle}$、$a^{\langle t' \rangle}$与$e^{\langle t,t' \rangle}$之间的关系下，那就用一个神经网络来进行学习，如下图所示：
![参数阿尔法](https://ws1.sinaimg.cn/large/82e16446gy1fp91hlkm8wj20lk07xglp.jpg)

要注意的是，该模型的运算成本将达到$N^2$。此外，在2015年Xu等人发表的论文[[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)]中，这种模型也被应用到了图像标注中。

### 应用

#### 语音识别

在语音识别中，要做的是将输入的一段语音$x$转换为一段文字副本作为输出。
![语音识别](https://ws1.sinaimg.cn/large/82e16446gy1fp95qtnapgj20ww0ajwhv.jpg)
曾经的语音识别系统都是采用人工设计出的**音素（Phonemes）**识别单元来构建，音素指的是一种语言中能区别两个词的最小语音单位。现在有了端对端深度学习，已经完美没有必要采用这种识别音素的方法实现语音识别。

采用深度学习方法训练语音识别系统的前提条件是拥有足够庞大的训练数据集。在学术界的研究中，3000小时的长度被认为是训练一个语音识别系统时，需要的较为合理的音频数据大小。而训练商用级别的语音识别系统，需要超过一万小时甚至十万小时以上的音频数据。

语音识别系统可以采用注意力模型来构建：
![Attention-语音识别](https://ws1.sinaimg.cn/large/82e16446gy1fp95qu7pagj20wp0fjgm8.jpg)

2006年Graves等人在论文[[Connectionist Temporal Classification: Labeling unsegmented sequence data with recurrent neural networks](http://people.idsia.ch/~santiago/papers/icml2006.pdf)]中提出了一种名为**CTC（Connectionist Temporal Classification）**损失函数计算方法，给语音识别系统的训练过程带来很大帮助。

由于输入的是音频数据，采用RNN建立的语音识别系统中将包含多个时间步，且整个网络中输出的数量往往是小于输入的数量的，也就是说不是每一个时间步都有对于的输出。而CTC的主要优点，是可对没有对齐的数据进行自动对齐。
![CTC](https://ws1.sinaimg.cn/large/82e16446gy1fp96lxvx58j20qb0dzmxh.jpg)
如上图中，以一句意为图中下面的句子，长度为10s频率为100Hz的语音作为输入，则这段语音序列可分为1000个部分，分别输入RNN的时间步中，而RNN可能不会将1000个作为输出。

CTC损失计算方法允许RNN输出一个如图中所示的结果，允许以“空白（Blank）”作为输出的同时，也会识别出词之间存在的“空格（Space）”标记，CTC还将把未被“空白”分隔的重复字符折叠起来。

关于CTC的更多细节详见论文内容。

#### 触发词检测

**触发词检测（Trigger Word Detection）**现在已经被应用在各种语音助手以及智能音箱上。例如在Windows 10上能够设置微软小娜用指令“你好，小娜”进行唤醒，安卓手机上的Google Assistant则可以通过“OK，Google”唤醒。

想要训练一个触发词检测系统，同样需要有大量的标记好的训练数据。使用RNN训练语音识别系统实现触发词词检测的功能时，可以进行如下图所示的工作：
![触发词检测](https://ws1.sinaimg.cn/large/82e16446ly1fp97z3o4ldj210s0cx77m.jpg)
在以训练的语音数据中输入RNN中，将触发词结束后的一小段序列的标签设置为“$1$”，以此训练模型对触发词的检测。


***
### 相关程序


### 参考资料
1. [吴恩达-序列模型-网易云课堂](http://mooc.study.163.com/course/2001280005?tid=2001391038#/info)
2. [Andrew Ng-Sequence Model-Coursera](https://www.coursera.org/learn/nlp-sequence-models/)
3. [零基础入门深度学习-循环神经网络](https://zybuluo.com/hanbingtao/note/541458)
4. [Deep Learning in NLP（一）词向量和语言模型](http://licstar.net/archives/328#s20)
5. [从SNE到t-SNE再到LargeVis](http://bindog.github.io/blog/2016/06/04/from-sne-to-tsne-to-largevis/)
6. [word2vec前世今生](https://www.cnblogs.com/iloveai/p/word2vec.html)
7. [Word2Vec导学第二部分-负采样-csdn](http://blog.csdn.net/qq_28444159/article/details/77514563)
8. [理解GloVe模型-csdn](http://blog.csdn.net/codertc/article/details/73864097)
9. [sequence to sequence model小记-知乎专栏](https://zhuanlan.zhihu.com/p/27766645)
10. [seq2seq中的beam search算法过程-知乎专栏](https://zhuanlan.zhihu.com/p/28048246)
11. [机器翻译自动评估-BLEU算法详解](http://blog.csdn.net/qq_31584157/article/details/77709454)
12. [深度学习方法：自然语言处理中的Attention Model注意力模型-csdn](http://blog.csdn.net/xbinworld/article/details/54607525)
13. [白话CTC算法讲解-csdn](http://blog.csdn.net/luodongri/article/details/77005948)

#### 更新历史
* 2019.04.14 完成初稿