通常一个卷积神经网络是由**输入层（Input）**、**卷积层（Convolution）**、**池化层（Pooling）**、**全连接层（Fully Connected）**组成。

在输入层输入原始数据，卷积层中进行的是前面所述的卷积过程，用它来进行提取特征。全连接层就是将识别到的所有特征全部连接起来，并输出到分类器（如Softmax）。

### LeNet-5

LeNet-5是LeCun等人1998年在论文[[Gradient-based learning applied to document recognition]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791)中提出的卷积网络。其结构如下图：

![LeNET-5](https://ws1.sinaimg.cn/large/82e16446ly1fm613aeqfbj20qv08i74i.jpg)

LeNet-5卷积网络中，总共大约有6万个参数。随着深度的增加，$n\_H$、$n\_W$的值在不断减小，$n\_C$却在不断增加。其中的Conv-Pool-Conv-Pool-FC-FC-Output是现在用到的卷积网络中的一种很常见的结构。

### AlexNet

AlexNet是Krizhevsky等人2012年在论文[[ImageNet classification with deep convolutional neural networks]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)中提出的卷积网络。其结构如下图：

![AlexNet](https://ws1.sinaimg.cn/large/82e16446ly1fm613apphnj20r60bgdgf.jpg)

AlexNet卷积网络和LeNet-5有些类似，但是比后者大得多，大约有6千万个参数。

### VGG-16

VGG-16是Simonyan和Zisserman 2015年在论文[[Very deep convolutional networks for large-scale image recognition]](https://arxiv.org/pdf/1409.1556.pdf)中提出的卷积网络。其结构如下图：

![VGG-16](https://ws1.sinaimg.cn/large/82e16446ly1fm613b1uuzj20qw0cb3yz.jpg)

VGG-16卷积网络的结构比较简单，其只要通过池化过程来压缩数据。VGG-16中的16指的是它有16个有权重的层。它是个比上面两个大得多的卷积网络，大约有13800万个参数。

### ResNets

当一个神经网络某个深度时，将会出现**梯度消失（Vanishing Gradient）**和**梯度爆炸（Exploding Gradient）**等问题。而ResNets能很好得解决这些问题。

ResNets全称为**残差网络（Residual Networks)**，它是微软研究院2015年在论文[[Deep Residual Learning for Image Recognition]](https://arxiv.org/pdf/1512.03385.pdf)中提出的卷积网络。

![残余块](https://ws1.sinaimg.cn/large/82e16446ly1fm8it0py7sj20a105d0sq.jpg)


如上图是一个神经网络中的几层，它一般的前向传播的过程，也称为“主要路径（main path）”为$a^{[l]}$-linear-ReLU-linear-ReLU-$a^{[l+2]}$，计算过程如下：$$ z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]} $$ 
$$a^{[l+1]} = g(z^{[l+1]}) $$
$$ z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]} $$
$$a^{[l+2]} = g(z^{[l+2]}) $$

在残差网络中，通过“捷径（short cut）”直接把$a^{[l]}$添加到第二个ReLu过程里，也就是最后的计算过程中：$$ a^{[l+2]} = g(z^{[l+2]} + W_s a^{[l]}) $$

其中$a^{[l]}$需要乘以一个矩阵$W_s$使得它的大小和$z^{[l+2]}$匹配。

深度神经网络通过这种跳跃网络层的方式能获得更好的训练效果。上面这种结构被称为一个**残差块（Residual Blocks）**。

![残差网络](https://ws1.sinaimg.cn/large/82e16446ly1fm8ix05uzjj20pv066gm1.jpg)

在普通神经网络（Plain NetWork）中使用多个这种残余块的结构，就组成了一个残差网络。两者的成本曲线分别如下：

![成本曲线](https://ws1.sinaimg.cn/large/82e16446ly1fm8j58oeuhj20mv082q3f.jpg)

普通的神经网络随着梯度下降的进行，理论上成本是不断下降的，而实际上当神经网络达到一定的深度时，成本值降低到一定程度后又会趋于上升，残差神经网络则能解决这个问题。

对于一个神经网络中存在的一些恒等函数（Identity Function），残差网络在不影响这个神经网络的整体性能下，使得对这些恒等函数的学习更加容易，而且很多时候还能提高整体的学习效率。

### Network In NetWork

2013年新加坡国立大学的林敏等人在论文[[Network In NetWork]](https://arxiv.org/pdf/1312.4400.pdf)中提出了1×1卷积核及NIN网络。

![1×1卷积](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif)

使用1×1卷积核进行卷积的过程如上图，它就是在卷积过程中采用大小为1×1的滤波器。如果神经网络的当前一层和下一层都只有一个信道，也就是$n\_C = 1$，那么采用1×1卷积核起不到什么作用的。但是当它们分别为有m和n个信道时，采用1×1卷积核就可以起到跨信道聚合的作用，从而降低（或升高）数据的维度，可以达到减少参数的目的。换句话说，1×1的卷积核操作实现的其实就是一个特征数据中的多个Feature Map的线性组合，所以这个方法就可以用来改变特征数据的信道数。

![1×1卷积](https://ws1.sinaimg.cn/large/82e16446ly1fma9k5sfnrj20fy08p3yh.jpg)

如上面的例子中，用32个大小为1×1×192的滤波器进行卷积，就能使原先数据包含的192个信道压缩为32个。在此注意到，池化能压缩数据的高度（$n\_H$）及宽度（$n\_W$），而1×1卷积核能压缩数据的信道数（$n\_C$）。

### Inception Network

最早的Inception结构的V1版本是由Google的Szegedy 2014年在论文[[Going deeper with convolutions]](https://arxiv.org/pdf/1409.4842.pdf)中提出的，它是ILSVRC 2014中取得最好成绩的GoogLeNet中采用的的核心结构。通过不断改进，现在已经衍生有了V4版本。

早期的V1版本的结构借鉴了NIN的设计思路，对网络中的传统卷积层进行了修改，其结构大致如下面的例子中所示：

![Inception](https://ws1.sinaimg.cn/large/82e16446ly1fmab3u4njpj20oi0c53zs.jpg)

通常在设计一个卷积网络的结构时，需要考虑卷积过程、池化过程的滤波器的大小，甚至是要不要使用1×1卷积核。在Inception结构中，考虑到多个不同大小的卷积核（滤波器）能够增强网络的适应力，于是分别使用三个大小分别为1×1、3×3、5×5的卷积核进行same卷积，同时再加入了一个same最大池化过程。最后将它们各自得到的结果放在一起，得到了图中一个大小为28×28×256的结果。然而，这种结构中包含的参数数量庞大，对计算资源有着极大的依赖，上面的例子中光是与大小为5×5的滤波器进行卷积的过程就会产生1亿多个参数！

![降维](https://ws1.sinaimg.cn/large/82e16446ly1fmabxtju4mj20p70ebjsj.jpg)

在其中的过程中，再加入1×1卷积能有效地对输出进行降维。如上图中所示，中间的一层就像是一个沙漏的瓶颈部分，所以这一层有时被称为**瓶颈层（Bottleneck Layer）**。通过1×1卷积，最后产生参数数量有1240万左右，相比起原来的1亿多要小了许多。

在论文中提出的整个Inception模型结构如下：

![Inception Module](https://ws1.sinaimg.cn/large/82e16446ly1fmacelp3tdj20mi0dxdgp.jpg)

在一个卷积网络中加入多个这种模型，就构成了一个Inception网络，也就是GoogLeNet：

![Inception Network](https://ws1.sinaimg.cn/large/82e16446ly1fmacj2e74zj20pw0dp78m.jpg)

其中还包含一些额外的最大池化层用来聚合特征，以及最后的全连接层。此外还可以从中间层的一些Inception结构中直接进行输出（图中没有画出），也就是中间的隐藏层也可以直接用来参与特征的计算及结果预测，这样能起到调整的作用，防止过拟合的发生。

Inception模型后续有人提出了V2、V3、V4的改进，以及引入残差网络的版本，这些变体都源自于这个V1版本。

最后，值得一提的是，Inception这个名字来自于电影《盗命空间》，用其中"We need to go deeper"这个梗，表明作者建立更深层次更加强悍的神经网络的决心！

![Inception](https://ws1.sinaimg.cn/large/82e16446ly1fmad47wvpij20qz0f8e2y.jpg)

***
#### 相关程序

#### 参考资料
1. [吴恩达-卷积神经网络-网易云课堂](http://mooc.study.163.com/course/2001281004#/info)
2. [Andrew Ng-Convolutional Neural Networks-Coursera](https://www.coursera.org/learn/convolutional-neural-networks/)
1. [LeNet-5官网](http://yann.lecun.com/exdb/lenet/index.html)
2. [梯度消失、梯度爆炸-csdn](http://blog.csdn.net/cppjava_/article/details/68941436)
3. [残差resnet网络原理详解-csdn](http://blog.csdn.net/mao_feng/article/details/52734438)
4. [关于CNN中1×1卷积核和Network in Network的理解-csdn](http://blog.csdn.net/haolexiao/article/details/77073258)
5. [GoogLeNet 之 Inception(V1-V4)-csdn](http://blog.csdn.net/diamonjoy_zone/article/details/70576775)

#### 更新历史
* 2019.04.19 完成初稿