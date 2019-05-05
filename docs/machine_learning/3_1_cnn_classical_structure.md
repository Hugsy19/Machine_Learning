通常一个卷积神经网络是由**输入层（Input）**、**卷积层（Convolution）**、**池化层（Pooling）**、**全连接层（Fully Connected）**组成。

这里介绍几种经典、流行的CNN结构。

### LeNet-5

LeNet-5是LeCun等人1998年在论文[[Gradient-based learning applied to document recognition]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791)中构建的，用来实现手写数字识别的CNN，其结构如下图：

![LeNET-5](https://i.loli.net/2019/05/05/5cce9ade388e2.jpg)

该CNN的结构为Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output，且随着网络层的增加，$n\_H$、$n\_W$不断减小，$n\_C$却不断增大。这两种特性在当今的CNN中也很常见。

### AlexNet

AlexNet是Krizhevsky等人2012年在论文[[ImageNet classification with deep convolutional neural networks]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)中提出的卷积网络。其结构如下图：

![AlexNet](https://i.loli.net/2019/05/05/5cce9afd339f0.jpg)

AlexNet卷积网络和LeNet-5有些类似，但是比后者大得多，大约有6千万个参数。

### VGG-16

VGG-16是Simonyan和Zisserman 2015年在论文[[Very deep convolutional networks for large-scale image recognition]](https://arxiv.org/pdf/1409.1556.pdf)中提出的卷积网络。其结构如下图：

![VGG-16](https://i.loli.net/2019/05/05/5cce9b0f5a141.jpg)

VGG-16卷积网络的结构比较简单，它主要通过大量的池化层来压缩数据，且VGG-16中的16指的是它有16个有权重的层。它是个比上面两个大得多的卷积网络，大约有13800万个参数。

### ResNets

前面提过，神经网络达到某个深度后经常将会出现**梯度消失（Vanishing Gradient）**和**梯度爆炸（Exploding Gradient）**等问题，CNN中也不例外。微软研究院2015年在论文[[Deep Residual Learning for Image Recognition]](https://arxiv.org/pdf/1512.03385.pdf)中提出的**残差网络（Residual Networks)**,能很好得解决这些问题。

![残余块](https://i.loli.net/2019/05/05/5cce9b2757e91.jpg)

如上图是某个神经网络中，某几层的正向传播的过程，其中具体的计算过程为：$$ z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]} $$  $$a^{[l+1]} = g(z^{[l+1]}) $$ $$ z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]} $$ $$a^{[l+2]} = g(z^{[l+2]}) $$

在残差网络中，通过开通一条“捷径”，跳过中间的过程，而直接把$a^{[l]}$添加到图中最后一层的激活函数中，即最后的计算过程变为：$$ a^{[l+2]} = g(z^{[l+2]} + W_s a^{[l]}) $$

其中$a^{[l]}$需要乘以一个矩阵$W_s$使得它的大小和$z^{[l+2]}$匹配。

上面这种结构被称为一个**残差块（Residual Blocks）**，深度神经网络通过这种跳跃网络层的方式能获得更好的训练效果。在神经网络中使用多个这样的残差块，就能组成一个残差网络：

![残差网络](https://i.loli.net/2019/05/05/5cce9b3f0679e.jpg)

普通的神经网络随着梯度下降的进行，理论上成本是不断下降的，而实际上当神经网络达到一定的深度时，成本值降低到一定程度后又会趋于上升，残差神经网络则能解决这个问题。

对于一个神经网络中存在的一些恒等函数（Identity Function），残差网络在不影响这个神经网络的整体性能下，使得对这些恒等函数的学习更加容易，而且很多时候还能提高整体的学习效率。

### Network In NetWork

2013年新加坡国立大学的林敏等人在论文[[Network In NetWork]](https://arxiv.org/pdf/1312.4400.pdf)中提出了$1\times1$卷积及NIN网络。

![1×1卷积](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif)

使用大小为$1\times1$卷积核进行卷积的过程如上图所示。如果网络中的当前一层和下一层的通道数不同时，进行$1\times1$卷积就能跨通道将特征聚合起来，实现降低（或升高）数据的维度，而达到减少参数的目的。

![1×1卷积](https://i.loli.net/2019/05/05/5cce9b69919ff.jpg)

如上面的例子中，用$32$个大小为$1\times1\times192$的滤波器进行卷积，就能使原先的$192$个通道压缩为$32$个。在此注意到，池化能压缩数据的高度（$n\_H$）及宽度（$n\_W$），而$1\times1$卷积能压缩数据的通道数（$n\_C$）。

### GoogLeNet

最早的Inception结构的V1版本是由Google的Szegedy 2014年在论文[[Going deeper with convolutions]](https://arxiv.org/pdf/1409.4842.pdf)中提出的，它是ILSVRC 2014中取得最好成绩的GoogLeNet中采用的的核心结构。

早期的V1版本的结构借鉴了NIN的设计思路，对网络中的传统卷积层进行了修改，其结构大致如下面的例子中所示：

![Inception](https://i.loli.net/2019/05/05/5cce9b7f47ffa.jpg)

通常在设计一个卷积网络的结构时，需要考虑卷积时的卷积核、池化时的窗口大小，甚至考虑是否进行$1\times1$卷积。在Inception结构中，考虑到多个不同大小的卷积核能够增强网络的泛化能力，于是分别使用了三个大小分别为$1\times1$、$3\times3$、$5\times5$的卷积核进行same卷积，与此同时又加入了一次same最大池化。它们各自得输出的结果讲放在一起，最后统一输出图中所示的大小为$28\times28\times256$的结果。

然而，这种结构中包含的参数量庞大，需要大量的计算资源。上面的例子中，与大小为$5times5$的卷积核进行same卷积就会产生1亿多个参数。

![降维](https://i.loli.net/2019/05/05/5cce9ba8f18b0.jpg)

在前面的模型中加入$1\times1$卷积，能有效减少参数量。如上图中所示，$1\times1$卷积后得到的中间层就像是一个沙漏的瓶颈部分，所以这一层又被称为**瓶颈层（Bottleneck Layer）**。

在论文中提出的完整的一个Inception模型结构如下图所示：

![Inception Module](https://i.loli.net/2019/05/05/5cce9bbe7b38b.jpg)

在一个卷积网络中加入多个这种模型，就构成了一个Inception网络，也就是GoogLeNet：

![Inception Network](https://i.loli.net/2019/05/05/5cce9bda575df.jpg)

其中还包含一些额外的最大池化层用来聚合特征，以及最后的全连接层。此外还可以从中间层的一些Inception结构中直接进行输出（图中没有画出），也就是中间的隐藏层也可以直接用来参与特征的计算及结果预测，这样能起到调整的作用，防止过拟合的发生。

Inception模型后续有人提出了V2、V3、V4的改进，以及引入残差网络的版本，这些变体都源自于这个V1版本。

最后，值得一提的是，Inception这个名字来自于电影《盗命空间》，用其中"We need to go deeper"这个梗，表明作者建立更深层次更加强悍的神经网络的决心！

![Inception](https://i.loli.net/2019/05/05/5cce9c191f60b.jpg)

***
#### 相关程序

#### 参考资料
1. [吴恩达-卷积神经网络-网易云课堂](http://mooc.study.163.com/course/2001281004#/info)
2. [Andrew Ng-Convolutional Neural Networks-Coursera](https://www.coursera.org/learn/convolutional-neural-networks/)
3. [LeNet-5官网](http://yann.lecun.com/exdb/lenet/index.html)
4. [梯度消失、梯度爆炸-csdn](http://blog.csdn.net/cppjava_/article/details/68941436)
5. [残差resnet网络原理详解-csdn](http://blog.csdn.net/mao_feng/article/details/52734438)
6. [关于CNN中1×1卷积核和Network in Network的理解-csdn](http://blog.csdn.net/haolexiao/article/details/77073258)
7. [GoogLeNet 之 Inception(V1-V4)-csdn](http://blog.csdn.net/diamonjoy_zone/article/details/70576775)

#### 更新历史
* 2019.04.21 完成初稿