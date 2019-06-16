在神经网络中，一般所处的网络层级越深，提取到的特征的抽象层次也越高。为此，在CNN发展的近二十余年时间里，研究者们一直致力于构建更深层次、更加强悍的CNN，其间涌现了一大批典型的CNN结构。

### LeNet
LeNet是一个早期用来识别手写数字图像的CNN，它由Yann LeCun等人于1998年在论文[[Gradient-based learning applied to document recognition]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791)中提出。该网络定义了CNN中的三大基本组件——卷积层、池化层、全连接层，展示了通过梯度下降法（Gradient Descent）来训练CNN的可能。其结构如下图所示：

![LeNet](https://i.loli.net/2019/06/14/5d03053947fe718651.jpg)

该CNN的结构为Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output，且随着网络层的增加，$n\_H$、$n\_W$不断减小，$n\_C$却不断增大。这两种特性在当今的CNN中也很常见。LeNet称得上是CNN的开山鼻祖，后面的各种CNN结构都是在它的基础上改进而来。

### AlexNet
LeNet提出后的十几年间里，神经网络技术因一度被其他诸如支持向量机等机器学习技术超越而几乎无人问津。直到2012年Alex Krizhevsky等人年在论文[[ImageNet classification with deep convolutional neural networks]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)中提出的AlexNet在当年的ImageNet竞赛中以领先第二名10.9%的绝对优势一举夺冠，深度学习和神经网络才得以名声鹊起，并掀起了CNN的研究热潮。其结构如下图：

![AlexNet](https://i.loli.net/2019/06/14/5d0306675086379972.jpg)

AlexNet共包含8个网络层，其中前5层为卷积层（卷积层和池化层统称一个卷积层），最后的3层为全连接层。整个网络共包含6千万个需要学习的参数，当时使用了两块各包含3GB显存的GTX 580 GPU才训练完成。相比于LeNet，AlexNet在增加了网络深度的同时，提出使用ReLU来代替传统的Sigmoid函数作为网络中各神经元的激活函数。用局部响应归一化（Local Response Normalization， LRN）来标准化初始数据的同时，还用DropOut和数据扩充法来预防CNN出现过拟合问题。

### VGGNet
VGGNet是2014年ImageNet竞赛中定位任务的冠军，分类任务的季军。它由牛津大学视觉几何组（Visual Geometry Group）和开发出AlphaGo的DeepMind公司[11]于2015年在LCLR（International Conference on Learning Representations）会议上，在论文[[Very deep convolutional networks for large-scale image recognition]](https://arxiv.org/pdf/1409.1556.pdf)中提出。VGGNet存在多个变种，其中最常见的结构是VGG-16和VGG-19。

![VGG-16](https://i.loli.net/2019/06/14/5d0308cb2d2e770589.jpg)

上图即为VGG-16的网络结构示意图，它可以简单看作是AlexNet的加深版本。其中总共包含16个层级，网络中大约有13800万个需要学习的参数。其最大的创新点在于提出了重复使用简单的基础模块来构建深度网络模型的思想。

![卷积核堆叠](https://i.loli.net/2019/06/14/5d0308ffdde9956022.png)

如上图所示，将2个3×3大小的卷积核堆叠起来对输入进行卷积运算时，其有效感受野相当于1个5×5大小的卷积核。因此在卷积层中，可以使用多个小卷积核的堆叠形式代替大卷积核完成卷积运算，另外通过卷积核的数量逐渐加倍的策略来加深网络，这样不仅能够减少参数数量，而且结合了多个激活单元，有效增强了决策函数的判别能力。

### ResNet
在构建深层次的神经网络过程中，由于BP算法的先天缺陷，当网络达到某个深度后，随之而来的便是梯度消失（Vanishing Gradient）或梯度爆炸（Exploding Gradient）等问题，研究者们为此提出了一系列的解决方案，例如正则化、批标准化等策略，然而在深层次的神经网络中，这些策略往往会导致网络的性能出现退化。2015年MSRA的何恺明等人在论文[[Deep Residual Learning for Image Recognition]](https://arxiv.org/pdf/1512.03385.pdf)提出的ResNet中，引入了**残差块（Residual Block）**结构来解决深度网络的退化问题，而训练了两个分别包含101、152个网络层的ResNet-101和ResNet-152。在当年的ImageNet竞赛中，ResNet轻松夺得分类、检测、定位三大任务的冠军。

![残差块](https://i.loli.net/2019/06/14/5d030b4c2112c93931.png)

上图即为ResNet中残差块的示意图。设残差块的输入为$x$，期望通过卷积网络得到基础映射（underlying mapping）$H(x)$，则两者的残差$F(x)=H(x)-x$，因此，基础映射可另外表示成$H(x)=F(x)+x$。

假设模型学习残差映射（residual mapping）要比学习基础映射的过程更为容易，因此训练堆叠的非线性卷积网络层来学习残差映射$F(x)$，另通过“捷径（shortcut）”跳过部分网络层进行恒等映射（identity mapping），而直接将原始输入$x$前递。只要将残差映射和恒等映射这两部分的值相加，最后得到的结果仍是基础映射$H(x)$。

在神经网络中使用多个这样的残差块，就能组成一个残差网络：

![残差网络](https://i.loli.net/2019/05/05/5cce9b3f0679e.jpg)

增加CNN的网络层数量后，如果某些层级中进行的是恒等映射，引入残差块可以简化这些恒等映射的学习过程，从而有效防止神经网络在加深层次后，产生性能退化问题。

### Network In NetWork

2013年新加坡国立大学的林敏等人在论文[[Network In NetWork]](https://arxiv.org/pdf/1312.4400.pdf)中提出了$1\times1$卷积及NIN网络。

![1×1卷积](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif)

使用大小为$1\times1$卷积核进行卷积的过程如上图所示。如果网络中的当前一层和下一层的通道数不同时，进行$1\times1$卷积就能跨通道将特征聚合起来，实现降低（或升高）数据的维度，而达到减少参数的目的。

![1×1卷积](https://i.loli.net/2019/05/05/5cce9b69919ff.jpg)

如上面的例子中，用$32$个大小为$1\times1\times192$的滤波器进行卷积，就能使原先的$192$个通道压缩为$32$个。在此注意到，池化能压缩数据的高度（$n_H$）及宽度（$n_W$），而$1\times1$卷积能压缩数据的通道数（$n_C$）。

### GoogLeNet
在2014年ImageNet竞赛的分类任务上，谷歌团队在论文[[Going deeper with convolutions]](https://arxiv.org/pdf/1409.4842.pdf)中提出的GoogLeNet击败了VGGNet夺得冠军。与前面几种网络有所不同，GoogLeNet不是单纯依靠加深神经网络层数来提高网络的性能，而在加深网络的同时，借鉴了Network in Network中的设计思想，从而引入了Inception结构，来代替卷积层中先进行卷积，后进行激活、池化等操作的传统结构。

Inception这个名字来自于电影《盗命空间》，该结构中，考虑到不同大小的卷积核能够增强网络的泛化能力，于是分别使用了三个大小分别为1×1、3×3、5×5的卷积核同时对输入进行卷积运算，期间多次使用1×1大小的卷积核来压缩数据，其卷积后得到的中间层就像是一个沙漏的瓶颈部分，所以这一层又被称为**瓶颈层（Bottleneck Layer）**。另外还加入了一次最大池化。完成这些操作后，将得到的各种输出放在同一个特征图的不同通道上，而得到最终的输出结果。

![Inception结构](https://i.loli.net/2019/06/14/5d030c581e9f010985.png)

在一个卷积网络中加入多个这种模型，就构成了一个Inception网络，也就是GoogLeNet：

![Inception Network](https://i.loli.net/2019/05/05/5cce9bda575df.jpg)

GoogLeNet的v1版本中，利用Inception结构，构建了一个包含22个网络层的CNN，网络中还使用了Network in Network中提出的全局平均池化层（Global Average Pooling）来代替传统CNN中参数密集的全连接层。在后续的v2、v3、v4版本中，则分别引入VGGNet中使用多个小卷积核代替大卷积核、ResNet中的残差块等思想，进一步完善了Inception结构。

### MobileNet

MobileNet是论文[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)出提取的，针对边缘设备而设计的CNN。它基于深度可分离卷积（depthwise separable convolution）而构建，该卷积是一类factorized卷积，它将一个完整的卷积运算分为两步——Depthwise Convolution和Pointwise Convolution。

常规的卷积过程为：
![常规卷积](https://i.loli.net/2019/05/11/5cd69d7809e09.png)

其中包含$4\times3\times3\times3=108$个参数。而进行DSC时，DC的过程是在二维平面上用三个卷积核为对各通道分别卷积：
![Depthwise Convolution](https://i.loli.net/2019/05/11/5cd69e00b7ffa.png)

之后PC的过程则用多个$1\times1$大小的卷积核进行卷积：
![Pointwise Convolution](https://i.loli.net/2019/05/11/5cd69edcda7c7.png)

这样总共包含$3\times3\times3 + 1\times1\times3\times4 = 39$个参数，大大缩减参数数量。

MobileNet v1的整个架构如下图所示：
![MobileNet v1](https://i.loli.net/2019/05/11/5cd6a4fe9caba.png)

除了最后的全连接层外，每一层后面都进行批标准化后使用ReLU激活：
![结构对比](https://i.loli.net/2019/05/11/5cd6a5d61c7ad.png)

其中下采样通过增加卷积的步幅来实现，通过后面的平均池化将特征图的空间分辨率变为$1$进而输入后面的全连接层。DC、PC各算一层，则整个MobileNet共包含$28$层。

模型中将大部分计算复杂度都放到了$1\times1$卷积中，它可以通过高度优化的通用矩阵乘法（GEMM）功能来实现。由于模型较小，训练期间没有使用太多预防过拟合的措施

为适应特定应用场景，引入了称为width multiplier的超参数$\alpha$和Resolution Multiplier的超参数$\rho$，它们的值在$(0, 1]$之间选取，前者的作用是给每层均匀进行减负，把某一层中包含$M$个通道的输入变成$\alpha M$，$N$个通道的输入变成$\alpha N$，以此重新定义一个计算量更小的模型，不过该模型需要重新训练；后者的作用是设置输入的分辨率，以此减少计算复杂度。

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
8. [卷积神经网络中的Separable Convolution](https://yinguobing.com/separable-convolution/#fn2)

#### 更新历史
* 2019.04.21 完成初稿
* 2019.05.11 加入MobileNet
* 2019.06.14 修改完善