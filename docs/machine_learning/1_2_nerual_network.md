机器学习领域所说的神经网络，指的是一种模仿生物神经网络的结构和功能而建立的数学或计算模型，用于对函数进行估计或近似。前面介绍过的Logistic回归，就可以用一个简单的神经网络模型表示如下：

![Logistic回归](https://ws1.sinaimg.cn/large/82e16446ly1g1xnu0507oj20o007kaaj.jpg)

Logistic回归以及线性回归都可以看作一个单层的神经网络。虽然上图中可分为两层——由训练样本组成的**输入层（input layer）**和负责输出结果的**输出层（output layer）**，但由于输入层中并不涉及运算而不计入层数中，因此上面所示的的神经网络中层数为$1$。其中输出层是一个**神经元（nurual unit）**，它与上一层中的各个输入完全连接，因此又把输出层称为**全连接层（fully-connecte layer）**或**稠密层（dense layer）**。

此外，输出层中的这个神经元所做的工作，是用其中的$w$、$b$对输入进行线性变换得到中间量$z$后，再使用sigmoid函数对$z$进行非线性变换得到$a$。在神经网络中，将一个神经元中用来进行非线性变换的函数称为**激活函数(activation function)**，变换后输出的结果则称为**激活（activation）**。Logistic回归中，使用的是sigmoid函数作为激活函数，而得到的激活$a$就是等于最后的输出结果$\hat{y}$。

### 前向传递
下图所示是一个稍复杂的神经网络：

![两层神经网络](https://ws1.sinaimg.cn/large/82e16446ly1fjthu97petj20ix0dimyp.jpg)

根据前面所述，这是一个层数为$2$的神经网络。由于在训练神经网络的过程中，我们无法观察到处在输入层和输出层之间的网络层中产生的值，因此将这些中间的网络层称为**隐藏层（hidden layer）**。对上图中的网络，只存在一层隐藏层。

这里，我们在上标中用$[i]$来表示一个神经网络中的第$i$层，且输入层标识为第$0$层且用$a^{[0]}$来表示。将包含$n = 3$种特征的$m$个训练样本分别向量化，之后矩阵化为大小为$3 \times m$的样本空间$X$，则有：$$a^{[0]} = X$$

将输入层的$a^{[0]}$向前传递到隐藏层中，隐藏层包含$n^{[1]} = 4$个神经元。如果各神经元中都采用sigmoid作为激活函数，那么对第$j$个神经元有：$$z^{[1]}\_j = w^{[1]T}\_jX + b^{[1]}_j$$ $$ a^{[1]}_j = \sigma(z^{[1]}_j)$$


将输入集传递给隐藏层后，隐藏层随之产生激活表示为$a^{[1]}$，而隐藏层的第一节点生成的激活表示为$a^{[1]}_1$，第二个节点产生的激活为$a^{[1]}_2$，以此类推，则：$${ a^{[1]} = \begin{bmatrix} a^{[1]}_1 \\\ a^{[1]}_2 \\\ a^{[1]}_3 \\\ a^{[1]}_4 \end{bmatrix}\quad}$$
最后，输出层输出的值表示为$a^{[2]}$，则$\hat{y} = a^{[2]}$。


### 正向传播

如图所示，将样本输入隐藏层中的第一个节点后，可得；$$ z^{[1]}_1 = w^{[1]T}_1X + b^{[1]}_1, a^{[1]}_1 = σ(z^{[1]}_1)$$
以此类推：$$ z^{[1]}_2 = w^{[1]T}_2X + b^{[1]}_2, a^{[1]}_2 = σ(z^{[1]}_2)$$ $$ z^{[1]}_3 = w^{[1]T}_3X + b^{[1]}_3, a^{[1]}_3 = σ(z^{[1]}_3)$$ $$ z^{[1]}_4 = w^{[1]T}_4X + b^{[1]}_4, a^{[1]}_4 = σ(z^{[1]}_4)$$
将它们都表示成矩阵形式：$${ z^{[1]} = \begin{bmatrix} w^{[1]}_1 &  w^{[1]}_1 & w^{[1]}_1\\\ w^{[1]}_2 & w^{[1]}_2 & w^{[1]}_2 \\\ w^{[1]}_3 & w^{[1]}_3 & w^{[1]}_3 \\\ w^{[1]}_4 & w^{[1]}_4 & w^{[1]}_4 \end{bmatrix}\quad}\begin{bmatrix} x_1 \\\ x_2 \\\ x_3 \end{bmatrix}\quad + \begin{bmatrix} b^{[1]}_1 \\\ b^{[1]}_2 \\\ b^{[1]}_3 \\\ b^{[1]}_4 \end{bmatrix} = \begin{bmatrix} z^{[1]}_1 \\\ z^{[1]}_2 \\\ z^{[1]}_3 \\\ z^{[1]}_4 \end{bmatrix}\quad\quad$$
即：$$ z^{[1]} = w^{[1]}X + b^{[1]} $$ $$a^{[1]} = σ(z^{[1]})$$
![神经网络的表示](https://ws1.sinaimg.cn/large/82e16446ly1fjug7o5ldfj20fh0e8my3.jpg)
进过隐藏层后进入输出层，又有:$$ z^{[2]} = w^{[2]}a^{[1]} + b^{[2]}$$ $$a^{[2]} = σ(z^{[2]})$$

可以发现，在一个的共有l层，且第l层有$n^{[l]}$个节点的神经网络中，参数矩阵 $w^{[l]}$的大小为$n^{[l]}$\*$n^{[l-1]}$，$b^{[l]}$的大小为$n^{[l]}$\*1。

逻辑回归中，直接将两个参数都初始化为零。而在神经网络中，通常将参数w进行**随机初始化**，参数b则初始化为0。

除w、b外的各种参数，如学习率$\alpha$、神经网络的层数$l$，第$l$层包含的节点数$n^{[l]}$及隐藏层中用的哪种激活函数，都称为**超参数（Hyper Parameters）**，因为它们的值决定了参数w、b最后的值。

### 反向传播

如图，通过输入样本$x$及参数$w^{[1]}$、$b^{[1]}$到隐藏层，求得$z^{[1]}$，进而求得$a^{[1]}$；再将参数$w^{[2]}$、$b^{[2]}$和$a^{[1]}$一起输入输出层求得$z^{[2]}$，进而求得$a^{[2]}$，最后得到损失函数$\mathcal{L}(a^{[2]},y)$，这样一个从前往后递进传播的过程，就称为**前向传播（Forward Propagation）**。
![前向传播](https://ws1.sinaimg.cn/large/82e16446ly1fjupgnkcbtj20r4089q32.jpg)
前向传播过程中：$$ z^{[1]} = w^{[1]T}X + b^{[1]} $$ $$a^{[1]} = g(z^{[1]})$$  $$ z^{[2]} = w^{[2]T}a^{[1]} + b^{[2]}$$ $$a^{[2]} = σ(z^{[2]}) = sigmoid(z^{[2]})$$ $${\mathcal{L}(a^{[2]}, y)=-(ylog\ a^{[2]} + (1-y)log(1-a^{[2]}))}$$

在训练过程中，经过前向传播后得到的最终结果跟训练样本的真实值总是存在一定误差，这个误差便是损失函数。想要减小这个误差，当前应用最广的一个算法便是梯度下降，于是用损失函数，从后往前，依次求各个参数的偏导，这就是所谓的**反向传播（Back Propagation）**，一般简称这种算法为**BP算法**。
![反向传播](https://ws1.sinaimg.cn/large/82e16446ly1fjupevz9aaj20r00aqt99.jpg)
sigmoid函数的导数为：$${a^{[2]'} = sigmoid(z^{[2]})' =  \frac{\partial a^{[2]}}{\partial z^{[2]}} = a^{[2]}(1 - a^{[2]})}$$

由复合函数求导中的链式法则，反向传播过程中：$$ da^{[2]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} = -\frac{y}{a^{[2]}} + \frac{1 - y}{1 - a^{[2]}}$$ $$ dz^{[2]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} = a^{[2]} - y$$ $$ dw^{[2]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}}\cdot \frac{\partial z^{[2]}}{\partial w^{[2]}} = dz^{[2]}\cdot a^{[1]T}$$ $$ db^{[2]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}}\cdot \frac{\partial z^{[2]}}{\partial b^{[2]}} = dz^{[2]}$$ $$ da^{[1]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}} = dz^{[2]} \cdot w^{[2]} $$ $$ dz^{[1]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}}  \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}}= dz^{[2]} \cdot w^{[2]} × g^{[1]'}(z^{[1]}) $$ $$ dw^{[1]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}}  \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial w^{[1]}}= dz^{[1]} \cdot X^T $$ $$ db^{[1]} = \frac{\partial \mathcal{L}(a^{[2]}, y)}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}}  \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial b^{[1]}}= dz^{[1]}$$
这便是反向传播的整个推导过程。

在具体的算法实现过程中，还是需要采用逻辑回归中用到梯度下降的方法，将各个参数进行向量化、取平均值，不断进行更新。

### 激活函数

建立一个神经网络时，需要关心的一个问题是，在每个不同的独立层中应当采用哪种激活函数。逻辑回归中，一直采用sigmoid函数作为激活函数，此外还有一些更好的选择。

**tanh函数（Hyperbolic Tangent Function，双曲正切函数）**的表达式为：$$tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
函数图像为：
![tanh函数](https://ws1.sinaimg.cn/large/82e16446ly1fjuhv55rcsj208w05m744.jpg)

tanh函数其实是sigmoid函数的移位版本。对于隐藏单元，选用tanh函数作为激活函数的话，效果总比sigmoid函数好，因为tanh函数的值在-1到1之间，最后输出的结果的平均值更趋近于0，而不是采用sigmoid函数时的0.5，这实际上可以使得下一层的学习变得更加轻松。对于二分类问题，为确保输出在0到1之间，将仍然采用sigmiod函数作为输出的激活函数。

然而sigmoid函数和tanh函数都具有的缺点之一是，在z接近无穷大或无穷小时，这两个函数的导数也就是梯度变得非常小，此时梯度下降的速度也会变得非常慢。

**ReLU函数（Rectified Linear Unit，修正线性单元）**线性修正单元，也就是上面举例解释什么是神经网络时用到的ReLU函数也是机器学习中常用到的激活函数之一，它的表达式为：$$g(z) = max(0,z) =\begin{cases} 0,  & \text{($z$ $\le$ 0)} \\\ z, & \text{($z$ $\gt$ 0)} \end{cases} $$
函数图像为：
![ReLU函数](https://ws1.sinaimg.cn/large/82e16446ly1fjujlni8kyj20dp07d744.jpg)

当z大于0时是，ReLU函数的导数一直为1，所以采用ReLU函数作为激活函数时，随机梯度下降的收敛速度会比sigmoid及tanh快得多，但负数轴的数据都丢失了。

ReLU函数的修正版本，称为**Leaky-ReLU**，其表达式为：$$g(z) = max(0,z) =\begin{cases} \alpha z,  & \text{($z$ $\le$ 0)} \\\ z, & \text{($z$ $\gt$ 0)} \end{cases} $$
函数图像为：
![Leaky-ReLU](https://ws1.sinaimg.cn/large/82e16446ly1fjujlnn9tgj20de07m3ye.jpg)

其中$ \alpha $是一个很小的常数，用来保留一部非负数轴的值。

可以发现，以上所述的几种激活函数都是非线性的，原因在于使用线性的激活函数时，输出结果将是输入的线性组合，这样的话使用神经网络与直接使用线性模型的效果相当，此时神经网络就类似于一个简单的逻辑回归模型，失去了其本身的优势和价值。



### 深层神经网络

深层神经网络含有多个隐藏层，构建方法如前面所述，训练时根据实际情况选择激活函数，进行前向传播获得成本函数进而采用BP算法，进行反向传播，梯度下降缩小损失值。

拥有多个隐藏层的深层神经网络能更好得解决一些问题。如图，例如利用神经网络建立一个人脸识别系统，输入一张人脸照片，深度神经网络的第一层可以是一个特征探测器，它负责寻找照片里的边缘方向，**卷积神经网络（Convolutional Neural Networks，CNN）**专门用来做这种识别。

![深层神经网络](https://ws1.sinaimg.cn/large/82e16446ly1fjxyv40x0kj20kj09l419.jpg)

深层神经网络的第二层可以去探测照片中组成面部的各个特征部分，之后一层可以根据前面获得的特征识别不同的脸型的等等。这样就可以将这个深层神经网络的前几层当做几个简单的探测函数，之后将这几层结合在一起，组成更为复杂的学习函数。从小的细节入手，一步步建立更大更复杂的模型，就需要建立深层神经网络来实现。

***
#### 相关程序


#### 参考资料
1. [吴恩达-神经网络与深度学习-网易云课堂](http://mooc.study.163.com/learn/deeplearning_ai-2001281002)
2. [Andrew Ng-Neural Networks and Deep Learning-Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/)
3. [deeplearning.ai](https://www.deeplearning.ai/)
4. [课程代码与资料-GitHub](https://github.com/BinWeber/Machine-Learning)

#### 更新历史
* 2019.04.10 完成初稿
