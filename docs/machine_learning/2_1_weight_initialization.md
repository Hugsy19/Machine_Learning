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
要解决梯度消失问题，除了上述两种权值初始化方法，还可以使用数据标准化中的z-score方法，在神经网络各层中的神经元进行非线性激活前，对中间值$z$进行标准化。后面会提到，学习参数时常用一小批训练样本进行批梯度下降，因此进行批梯度下降时对$z$进行标准化的过程又称为**批标准化（Batch Normalization）**。

对神经网络中包含$n$个神经元的某一层的中间值$z$，进行批标准化的具体过程为：$$\mu = \frac{1}{n} \sum_{i=1}^n z^{(i)}$$ $$\sigma^2 = \frac{1}{n} \sum_{i=1}^n (z^{(i)}-\mu)^2$$ $$z_\text{norm}^{(i)} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$$ $$\tilde{z}^{(i)} = \gamma z_\text{norm}^{(i)}+\beta$$

和z-score标准化过程一样，前面先求出$z$的均值$\mu$、方差$\sigma^2$，再求得$z$标准化后的值$z_\text{norm}$。批标准化中，随后还对$z_\text{norm}$进行了一次线性变换，其中$\gamma$、$\beta$是两个需通过学习获得的参数，而非超参数。该线性变换过程又称为反标准化，设置该变换的目的是为了将选择权交给神经网络，当对$z$的正则化没有起优化作用甚至是负优化时，通过该变换来抵消一部分前面的正则化过程带来的影响

进行批标准化后，尽管每一层的$z$还是依输入的不同而不断发生变化，但是它们的均值和方差将基本保持不变，这可以使各网络层中的数据分布更加稳定，减少了前后层的依赖性，从而加快模型的训练过程。

要注意，训练时输入的是小批量的训练样本，而测试时的测试样本通常是一个一个输入的。因此对神经网络中的某一层，可在训练时保留该层过去一段时间内求得的$z$的均值及方差，进而用后面介绍的EWMA法分别求得均值、方差的EWMA值，测试时就可以用该值来标准化测试样本。

***
#### 相关程序


#### 参考资料
1. [吴恩达-改善深层神经网络-网易云课堂](http://mooc.study.163.com/course/deeplearning_ai-2001281003#/info)
2. [Andrew Ng-Improving Deep Neural Networks-Coursera](https://www.coursera.org/learn/deep-neural-network/)
3. [聊一聊深度学习的weight initialization-知乎专栏](https://zhuanlan.zhihu.com/p/25110150)
4. [什么是批标准化-知乎专栏](https://zhuanlan.zhihu.com/p/24810318)


#### 更新历史
* 2019.04.14 完成初稿
