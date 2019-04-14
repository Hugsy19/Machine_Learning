机器学习领域所说的神经网络，指的是一种模仿生物神经网络的结构和功能而建立的数学或计算模型，用于对函数进行估计或近似。前面介绍过的Logistic回归，就可以用一个简单的神经网络模型表示如下：

![Logistic回归](https://ws1.sinaimg.cn/large/82e16446ly1g1xnu0507oj20o007kaaj.jpg)

Logistic回归以及线性回归都可以看作一个单层的神经网络。虽然上图中可分为两层——由训练样本组成的**输入层（input layer）**和负责输出结果的**输出层（output layer）**，但由于输入层中并不涉及运算而不计入层数中，因此上面所示的的神经网络中层数为$1$。其中输出层是一个**神经元（nurual unit）**，它与上一层中的各个输入完全连接，因此又把输出层称为**全连接层（fully-connecte layer）**或**稠密层（dense layer）**。

此外，输出层中的这个神经元所做的工作，是用其中的$w$、$b$对输入进行线性变换得到中间量$z$后，再使用sigmoid函数对$z$进行非线性变换得到$a$。在神经网络中，将一个神经元中用来进行非线性变换的函数称为**激活函数(activation function)**，变换后输出的结果则称为**激活（activation）**。Logistic回归中，使用的是sigmoid函数作为激活函数，而得到的激活$a$就是等于最后的输出结果$\hat{y}$。

### 激活函数
在神经网络中，可以证明，如果所有的神经元都只对输入作线性变换，那么不管神经网络中包含再多的隐藏层，最后输出层输出的结果依然只是对初始输入数据作线性变换，这样的神经网络只能拟合出线性函数。因此，在神经元中引入非线性的激活函数是必不可少的。

除了Logistic回归中用到的sigmoid函数可作为激活函数，神经网络中常用的激活函数还有以下几种：

#### tanh函数
tanh函数又称为双曲正切函数（Hyperbolic Tangent Function），其表达式为： $$tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

图像为：

![tanh函数](https://ws1.sinaimg.cn/large/82e16446ly1g1z02aonvnj20iv0870ta.jpg)

tanh函数其实上是经过平移的sigmoid函数。对于神经网络中的隐藏单元，如果选用tanh函数作为激活函数，因为它的值限定在$-1$到$1$之间，激活后所有输出结果的平均值将趋近于$0$，而采用sigmoid函数时平均值将为的$0.5$，更小的平均值可以使下一层网络的学习过程变得更为轻松，因此它的效果总比sigmoid函数好。

对于二分类问题，为确保输出值在$0$到$1$之间，将仍然采用sigmiod函数作为输出层的激活函数。

然而tanh、sigmoid函数都存在一个缺点：由求导法则，tanh函数的导函数为：$$tanh'(z) = 1 - tanh^2(z)$$

sigmoid函数的导函数则为：$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) $$

在$z$的值比较大时，这两个函数的导数也就是梯度值将会趋近于$0$，从而导致梯度下降的速度变得非常缓慢。

#### ReLU函数

ReLU函数又称为线性修正单元（Rectified Linear Unit），它是当前神经网络的隐藏层中用得最多的激活函数。它是一个分段函数，提供了一个很简单的非线性变换，其表达式为：$$g(z) = max(0,z) =\begin{cases} 0,  & \text{($z$ $\le$ 0)} \\\ z, & \text{($z$ $\gt$ 0)} \end{cases} $$

图像为：

![ReLU函数](https://ws1.sinaimg.cn/large/82e16446ly1g1z0ou64faj20qk0b4t8v.jpg)

导函数为：$$g'(z) =\begin{cases} 0,  & \text{($z \lt 0$)} \\\ 1, & \text{($z \gt 0$)} \end{cases} $$

当$z \gt 0$时，ReLU函数的导数一直为1，所以采用ReLU函数作为激活函数时，随机梯度下降的收敛速度会比tanh、sigmoid快得多。然而当$z \lt 0$时，ReLU函数将一直输出$0$，且导数也一直为$0$，且在$z = 0$处不可导。

要解决这个问题，存在一个称为**Leaky-ReLU**的ReLU函数修正版本，其表达式及图像如下：$$g(z) = max(0,z) =\begin{cases} \alpha z,  & \text{($z$ $\le$ 0)} \\\ z, & \text{($z$ $\gt$ 0)} \end{cases} $$

![Leaky-ReLU](https://ws1.sinaimg.cn/large/82e16446ly1fjujlnn9tgj20de07m3ye.jpg)

其中$ \alpha $是一个很小的常数，用来保留一部非负数轴的值。

### 正向传播

下图所示是一个比前面的Logistic回归稍复杂的神经网络：

![两层神经网络](https://ws1.sinaimg.cn/large/82e16446ly1g1z132bjuoj20jr0axjt9.jpg)

根据前面所述，这是一个层数为$L = 2$的神经网络。由于在训练神经网络的过程中，我们无法观察到处在输入层和输出层之间的网络层中产生的值，因此将这些中间的网络层统称为**隐藏层（hidden layer）**。对上图中的网络，只存在一层隐藏层。

这里，我们在上标中用$[i]$来表示一个神经网络中的第$i$层，且将输入层标识为第$0$层并用$a^{[0]}$来表示。将一个包含$n^{[0]} = 3$种特征的训练样本向量化，则有：$$a^{[0]} = x = \begin{bmatrix} x_1 \\\ x_2 \\\ x_3 \end{bmatrix} $$

将输入层的$a^{[0]}$向前传递到隐藏层中，隐藏层包含$n^{[1]} = 4$个神经元。在神经网络中，各神经元中权重$w$的初始值不能像Logistic回归中那样直接设为$0$，这会导致网络中隐藏层的各神经元中一直都在进行相同的运算，所以$w$的值必须进行**随机初始化**。

那么该神经网络的各神经元中，权重$w$会是个经过随机初始化的大小为$1 \times n^{[0]}$的行向量，偏差$b$则直接初始化为常数$0$，且均采用ReLU为激活函数，那么该隐藏层中存在如下计算过程：$$z^{[1]}\_1 = w^{[1]}\_1a^{[0]} + b^{[1]}\_1, \ \ \  a^{[1]}\_1 = g(z^{[1]}\_1)$$ $$z^{[1]}\_2 = w^{[1]}\_2a^{[0]} + b^{[1]}\_2, \ \ \  a^{[1]}\_2 = g(z^{[1]}\_2)$$ $$z^{[1]}\_3 = w^{[1]}\_3a^{[0]} + b^{[1]}\_3, \ \ \  a^{[1]}\_3 = g(z^{[1]}\_3)$$ $$z^{[1]}\_4 = w^{[1]}\_4a^{[0]} + b^{[1]}\_4, \ \ \  a^{[1]}\_4 = g(z^{[1]}\_4)$$

显然，上面的计算过程可以矩阵化为：$$ \begin{aligned} z^{[1]} = \begin{bmatrix} z^{[1]}\_1 \\\ z^{[1]}\_2 \\\ z^{[1]}\_3 \\\ z^{[1]}\_4 \end{bmatrix} & = \begin{bmatrix} w^{[1]}\_1 \\\ w^{[1]}\_2 \\\ w^{[1]}\_3\\\ w^{[1]}\_4 \end{bmatrix} a^{[0]}+ \begin{bmatrix} b^{[1]}\_1 \\\ b^{[1]}\_2 \\\ b^{[1]}\_3 \\\ b^{[1]}\_4 \end{bmatrix} \\\ & = W^{[1]}a^{[0]} + b^{[1]} \end{aligned}$$ $$ \begin{aligned} a^{[1]} =  \begin{bmatrix} a^{[1]}\_1 \\\ a^{[1]}\_2 \\\ a^{[1]}\_3 \\\ a^{[1]}\_4 \end{bmatrix} & = g(\begin{bmatrix} z^{[1]}\_1 \\\ z^{[1]}\_2 \\\ z^{[1]}\_3 \\\ z^{[1]}\_4 \end{bmatrix}) \\\ & = g(z^{[1]})\end{aligned}$$

隐藏层中所有的权重组成的矩阵$W^{[1]}$的大小为$n^{[1]} \times n^{[0]}$, 偏差$b^{[1]}$的大小为$n^{[1]}\times 1$。得到的中间值$z^{[1]}$以及激活$a^{[1]}$大小皆为$n^{[1]} \times 1$。

隐藏层的激活$a^{[1]}$继续向前传递到输出层，输出层中包含的神经元个数$n^{[2]} = 1$，且以sigmoid作为激活函数，则又有:$$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$$ $$a^{[2]} = σ(z^{[2]})$$ $$\hat{y} = a^{[2]}$$ 

其中$W^{[2]}$的大小为$n^{[2]} \times n^{[1]}$，$b^{[2]}$的大小则为$n^{[2]}\times 1$，运算后得到大小为$n^{[2]} \times 1$的激活$a^{[2]}$，这个神经网络中，该激活也就等于最后的输出$\hat{y}$。

在训练该神经网络时，紧接着是计算损失，神经网络中依然采用前面介绍的交叉熵损失函数作为损失函数，即有：
$$\begin{aligned} \mathcal{L}(\hat{y}, y) & =- y\log \hat{y} - (1-y)\log(1 - \hat{y}) \\\ & = - y\log a^{[2]} - (1-y)\log(1 - a^{[2]})\end{aligned}$$

对上面的所有过程，可以用一个简单的流程图表示如下：
![前向传播](https://ws1.sinaimg.cn/large/82e16446ly1g1zoh3ibisj20r706fmxi.jpg)

在神经网络中，所谓的**正向传播（Forward Propagation）**，就是这样一个从前往后递进传播的计算过程。

通过上面的过程可以发现，对第$l$层共有$n^{[l]}$个神经元的神经网络，第$l$层中的权重$W^{[l]}$是一个大小为$n^{[l]} \times n^{[l-1]}$的矩阵，偏差$b^{[l]}$则是大小为$n^{[l]} \times 1$的列向量。

类似梯度下降中的学习率$\alpha$，神经网络中的层数$l$、第$l$层包含的神经元数量$n^{[l]}$以及各层选用的激活函数，都是需要人为设置的超参数，最后训练出来的模型的表现情况将受它们的影响。

### 反向传播
前面我们通常采用梯度下降法来将损失进行最小化，在神经网络中则采用**BP算法**也就是**反向传播（Back Propagation）** 来实现这一过程。还是以前面的那个$2$层神经网络为例，其反向传播的完整过程如下图中的红色标识所示：
![反向传播](https://ws1.sinaimg.cn/large/82e16446ly1fjupevz9aaj20r00aqt99.jpg)

反向传播的过程，是使损失函数$\mathcal{L}(a^{[L]}, y)$向前对网络中各参数分别进行求导。首先对输出层的激活$a^{[2]}$，根据求导法则有：$$da^{[2]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} = -\frac{y}{a^{[2]}} + \frac{1 - y}{1 - a^{[2]}}$$

输出层使用的是sigmoid激活函数，则激活$a^{[2]}$对其中间量$z^{[2]}$有：$$\frac{\partial a^{[2]}}{\partial z^{[2]}} = a^{[2]}(1 - a^{[2]})$$

根据链式求导法则，有：$$ dz^{[2]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} = a^{[2]} - y$$

进而对输出层中的各参数，有： $$ dW^{[2]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}}\cdot \frac{\partial z^{[2]}}{\partial W^{[2]}} = dz^{[2]}\cdot a^{[1]T}$$ $$db^{[2]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}}\cdot \frac{\partial z^{[2]}}{\partial b^{[2]}} = dz^{[2]}$$ 

对隐藏层，其中使用的激活函数是ReLU函数，则有：$$da^{[1]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}} = W^{[2]T} \cdot dz^{[2]} $$ $$ dz^{[1]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}}  \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}}= W^{[2]T} \cdot dz^{[2]} \times g'(z^{[1]}) $$ 

进而对其各参数，有：$$ dW^{[1]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}}  \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial W^{[1]}}= dz^{[1]} \cdot a^{[0]T} $$ $$ db^{[1]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}}  \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial b^{[1]}}= dz^{[1]}$$

其中“$\cdot$”号代表的是矩阵相乘，“$\times$”则代表逐个元素对应相乘，各参数求导后的维度与求导前将保持一致。用BP算法求得各参数的导数后，接下来更新参数的方法与梯度下降中的一致，随后迭代进行正向、反向传播过程来训练出神经网络中各参数的最优解。

在实际训练模型时为方便计算，通常将$m$训练样本放在一起组成样本空间$X$作为神经网络中的输入，此时$a^{[0]}$的大小为$n^{[0]} \times m$，可改用$A^{[0]}$进行表示。第一个隐藏层中的权重$W^{[1]}$和偏差$b^{[1]}$大小仍为$n^{[1]} \times n^{[0]}$、$n^{[1]} \times 1$，且偏差$b$将利用广播机制添加到矩阵相乘后所有的结果中，后面各隐藏层中的各参数的大小可根据前面所述的规律递归。最后的第$L$层也就是输出层中，输出的激活$A^{[L]}$大小将为$n^{[L]} \times m$。

***
#### 相关程序
* [Ng-DL1-week3-猫图分类器](https://github.com/BinWeber/Machine_Learning/blob/master/Ng_Deep_Learning/1_Neural_Network/week_3/Cat_Classfication_Neural_NetWork_Numpy.ipynb)
* [Ng-DL1-week4-花瓣分类器](https://github.com/BinWeber/Machine_Learning/blob/master/Ng_Deep_Learning/1_Neural_Network/week_4/Planar_Date_Classification.ipynb)

#### 参考资料
1. [吴恩达-神经网络与深度学习-网易云课堂](http://mooc.study.163.com/learn/deeplearning_ai-2001281002)
2. [Andrew Ng-Neural Networks and Deep Learning-Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/)
3. [动手学深度学习](http://zh.d2l.ai/index.html)

#### 更新历史
* 2019.04.12 完成初稿
