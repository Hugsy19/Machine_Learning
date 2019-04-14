在深度神经网络中，如果权重$W$的初始值都大于$1$，那么正向传播过程中，$W$会随着层数的增加而呈指数级增加，从而在反向传播时造成**梯度爆炸**；都小于$1$时则会呈指数级衰减，最后造成**梯度消失**。此外，激活函数选择不当时也容易出现这两种问题。

利用后面所述的正则化，除了能够预防过拟合外，也它能在一定程度上避免梯度爆炸的问题的发生。且在深度学习中，往往更容易出现梯度消失问题。

前面提到过，对线性回归、Logistic回归等简单的模型，其所有的参数都可以直接初始化为$0$，而对神经网络中的权重$W$,必须采用一些特定的方法的对它进行初始化。

### Xavier初始化
常见的初始化方法，是使用编程语言中内置的randn方法产生一些服从正态分布的随机数，用它们来对$W$进行**随机初始化（random initialization）**。然后当随机分布的选择不当时，还是容易导致梯度消失问题。

2010年Xavier等人发表的论文[[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)]中，提出了一种另一种神经网络中的权重初始化方法——**Xavier初始化**。使用该方法对网络中第$l$层的权重$W$进行初始化，有：
```
w_l = np.random.randn(layers_dims[l],layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
```
该方法中，依然使用randn来产出服从标准正态分布的随机数，紧接着，使这些随机数除以数据的输入量，也就是上层网络中神经元的个数$n^{[l-1]}$开根号后的值，以保持输入和输出的方差一致，从而避免了所有输出值都趋向于0。

### He初始化
对使用ReLU作为激活函数的隐藏层，2015年何恺明等人在论文[[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)]中提出被称为**He初始化**的方法。对于采用ReLU的网络层，该方法假定其中只有一半的神经元被激活，要保持方差不变，需要在Xavier初始化方法的基础上再除以$2$：
```
w_l = np.random.randn(layers_dims[l],layers_dims[l-1]) / np.sqrt(layers_dims[l-1]/2)
```
### 批标准化

**批标准化（Batch Normalization）**和之前的数据集标准化类似，是将分散的数据进行统一的一种做法。具有统一规格的数据，能让机器更容易学习到数据中的规律。

对于含有$m$个节点的某一层神经网络，对$z$进行操作的步骤为：$$\mu = \frac{1}{m} \sum\_{i=1}^m z^{(i)}$$ $$\sigma^2 = \frac{1}{m} \sum\_{i=1}^m (z^{(i)}-\mu)^2$$ $$z\_{norm}^{(i)} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$$ $$\tilde{z}^{(i)} = \gamma z\_{norm}^{(i)}+\beta$$
其中的$\gamma$、$\beta$并不是超参数，而是两个需要学习的参数，神经网络自己去学着使用和修改这两个扩展参数。这样神经网络就能自己慢慢琢磨出前面的标准化操作到底有没有起到优化的作用。如果没有起到作用，就使用 $\gamma$和$\beta$来抵消一些之前进行过的标准化的操作。例如当$\gamma = \sqrt{\sigma^2+\epsilon}, \beta = \mu$，就抵消掉了之前的正则化操作。

![Batch Norm](https://ws1.sinaimg.cn/large/82e16446ly1fk9r4pxdodj20ik05smxu.jpg)

将图中的神经网络中的$z^{[1]}$、$z^{[2]}$进行批标准化后，$z^{[1]}$、$z^{[2]}$将变成$\tilde{z}^{[1]}$、$\tilde{z}^{[2]}$。

当前的获得的经验无法适应新样本、新环境时，便会发生“Covariate Shift”现象。对于一个神经网络，前面权重值的不断变化就会带来后面权重值的不断变化，批标准化减缓了隐藏层权重分布变化的程度。采用批标准化之后，尽管每一层的z还是在不断变化，但是它们的均值和方差将基本保持不变，这就使得后面的数据及数据分布更加稳定，减少了前面层与后面层的耦合，使得每一层不过多依赖前面的网络层，最终加快整个神经网络的训练。

批标准化还有附带的有正则化的效果：当使用小批量梯度下降时，对每个小批量进行批标准化时，会给这个小批量中最后求得的$z$带来一些干扰，产生类似与DropOut的正则化效果，但效果不是很显著。当这个小批量的数量越大时，正则化的效果越弱。

需要注意的是，**批标准化并不是一种正则化的手段**，正则化效果只是其顺带的小副作用。另外，在训练时用到了批标准化，则在测试时也必须用到批标准化。

训练时，输入的是小批量的训练样本，而测试时，测试样本是一个一个输入的。这里就又要用到指数加权平均，在训练过程中，求得每个小批量的均值和方差的数加权平均值，之后将最终的结果保存并应用到测试过程中。

***
#### 相关程序


#### 参考资料
1. [吴恩达-改善深层神经网络-网易云课堂](http://mooc.study.163.com/course/deeplearning_ai-2001281003#/info)
2. [Andrew Ng-Improving Deep Neural Networks-Coursera](https://www.coursera.org/learn/deep-neural-network/)
3. [聊一聊深度学习的weight initialization-知乎专栏](https://zhuanlan.zhihu.com/p/25110150)
4. [什么是批标准化-知乎专栏](https://zhuanlan.zhihu.com/p/24810318)


#### 更新历史
* 2019.04.14 完成初稿
