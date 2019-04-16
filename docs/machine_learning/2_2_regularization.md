下图是基于相同的训练样本得到的几个不同的模型：

![分类问题](https://ws1.sinaimg.cn/large/82e16446ly1fk3xje7ndkj20pt09zq44.jpg)

可以看出，其中中间所示的决策界限最为理想，而另外两个模型中出现的问题在机器学习中分别称为：
* **欠拟合（underfitting）**：模型的表现还不够好，与实际相比存在较大的**偏差（bias）**，如上面的左图
* **过拟合（overfitting）**：模型的表现过于“优秀”，与实际相比较大的**方差（variance）**，如上面的右图

训练出来的模型存在欠拟合问题，常常是因为模型还训练得不够好，可以尝试使用更复杂的模型、采用更好优化算法等方法来解决。出现过拟合问题，则往往是由于训练出来得模型过于复杂，导其泛化能力不够强，该问题比欠拟合要难解决得多，当前常用**正则化（regularization）**、**Dropout**、增加训练样本等方法来预防模型出现过拟合问题。

### 正则化
![过拟合](https://ws1.sinaimg.cn/large/82e16446ly1g1dxy8qp7gj20pj08qq4o.jpg)

上面的右图所示的回归模型明显存在过拟合，要解决该问题，最简单的做法就是想办法使模型中的参数$\theta_3$、$\theta_4$尽可能地接近于$0$，而使其近似地等于左图中最为理想的模型。

进一步地，我们知道，参数$\theta$的值都是通过最小化成本函数而求得的。那么要达到上述目的，可以考虑将这几个参数“捆绑”在成本函数上：$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 + 1000\theta_3^2 + 1000\theta_4^2$$

这样，在最小化成本函数时，参数$\theta_3$、$\theta_4$将受到“惩罚”而一同被最小化，以此达到防止过拟合的目的。所谓的正则化过程也与此类似。

建立机器学习模型时，学习到的参数值较小，就意味着假设函数会是一个相对“简单”的函数，过拟合也更不容易发生。正则化的思想即在于此，其具体做法是不更改原来的模型，而在成本函数中加入由模型中的主要参数，比如权重$w$组成的正则化项，通过使所有的参数与成本一起被最小化，而达到防止过拟合的目的。

在线性回归、Logistic回归等模型中，权重$w$是一个维度为$n_x$的向量。对于它们的成本函数，常添加的是$L2$正则化项，它源自$L2$范数：$$\vert \vert x \vert \vert \_2 = \sqrt{\sum_{i=1}^k \vert x_i \vert^2}$$

向Logistic回归的成本函数中加入$L2$正则化后,有：$$\mathcal{J}(w, b) = -\frac{1}{m} \sum_{i=1}^m y^{(i)} \log \hat{y}^{(i)} - (1 - y^{(i)})\log(1 - \hat{y}^{(i)}) + \frac{\lambda}{2m} \vert \vert w \vert \vert^2\_2$$

其中：$$\vert \vert w \vert \vert ^2\_2 = \sum_{j=1}^{n_x} w^2\_i = w^Tw$$

超参数$\lambda$用以控制正则化的程度小，加入常数$\frac{1}{2}$是为了方便后面的求导。

另外还有源于$L1$范数的$L1$正则化项，其表达式为：$$\vert \vert w \vert \vert_1 = \sum_{j=1}^{n_x} \vert w_i \vert$$

在神经网络等模型中，权重$w$是一个矩阵，例如第$l$层神经网络中权重$W$的大小为$n^{[l]} \times n^{[l-1]}$，此时常用的正则化项源自$Frobenius$范数：$$||A||\_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n  \vert a_{ij} \vert^2}$$

向某个神经网络的成本函数中加入$F$正则化后,有：$$\mathcal{J}(W, b) = \frac{1}{m} \sum^m_{i=1} \mathcal{L}({\hat y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum_{l=1}^L \vert \vert W^{[l]} \vert \vert^2\_F$$

其中：$$ \vert \vert W^{[l]}\vert \vert^2\_F = \sum_{i=1}^{n^{[l]}} \sum_{j=1}^{n^{[l-1]}} (w_{ij}^{[l]})^2$$

进行正则化后，反向传播时将有：$$dW^{[l]} = \frac{\partial \mathcal{L}(W, b)}{\partial W^{[l]}} + \frac{\lambda}{m} W^{[l]}$$

更新权重时将有：$$ W^{[l]} := W^{[l]} - \alpha \frac{\partial \mathcal{L}(W, b)}{\partial W^{[l]}} - \alpha \frac{\lambda}{m} W^{[l]}$$

此外，线性回归中采用正规方程直接求解模型中的参数$w$时，也可以进行正则化。其表达式为：$$w = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty$$

其中的正则化项$L$是大小是$(n+1)\times(n+1)$的矩阵：$$L = \begin{bmatrix} 0 & & & & \\\ & 1 & & & \\\ & & 1 & & \\\ & & & \ddots & \\\ & & & & 1 \\\ \end{bmatrix}$$

在讲解正规方程时，曾经提到过正规方程种$X^TX$为奇异矩阵或非方阵时，它将不存在逆矩阵。对正规方程进行正规化后，便不会出现这种情况，因为$X^TX + \lambda \cdot L$一定是可逆的。

### Dropout
2012年，将深度学习应用在计算机视觉领域的开山之作[[ImageNet Classification with Deep Convolutional](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)]中提出了神经网络中另外一种用于预防过拟合的算法——Dropout。正如其名，所谓的Dropout，就是使神经网络中的部分神经元失活，以对整个模型进行简化，如下图所示：
![Dropout](https://ws1.sinaimg.cn/large/82e16446ly1g215n5oj7ij20wk08a41p.jpg)

具体实现Dropout，传统也就上述论文中的做法，是在训练阶段对网络中的各神经元都设置一个保留概率$\text{keep_prob}$，则各神经元将以$\text{keep_prob}$的概率保留下来，而以$1 - \text{keep_prob}$的概率失活。在测试阶段则不执行Dropout，但是训练过程中失活的神经元此时输出的激活需要乘上$\text{keep_prob}$。

而当前的主流做法，是在训练过程中使用Inverted Dropout来实现Dropout，具体过程如下：
1. 为进行Dropout的网络层设定好该层神经元的保留概率$\text{keep_prob}$；
2. 随机生成一个与该层输出的激活矩阵$A^{[l]}$大小相等的矩阵$D^{[l]}$，并将其与$\text{keep_prob}$比较，使其变成元素都为$1$（保留）或$0$（失活）的布尔矩阵;
3. 计算$A^{[l]} \times D^{[l]}$（对应元素相乘），将其结果重新赋给$A^{[l]}$；
4. 计算$A^{[l]}/\text{keep_prob}$，从而保证失活后该层的输出结果的期望（均值）不发生改变，最后将该结果作为该层的输出。

例如对上图网络中的第三层进行Inverted Dropout，在python中实现如下：
```python
import numpy as np

keep_prob = 0.8
d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep.prob 
a3 = np.multiply(a3, d3)
a3 /= keep.prob
z4 = np.dot(w4, a3) + b4
```
要注意的是，Inverted Dropout的第$4$步已经保证了输出的期望不变，所以在测试过阶段不需要再使用Dropout。

进行Dropout后，网络中的每个神经元都存在失活的可能，以此防止神经网络对某些神经元即某些输入特征过于依赖，实现对权重的压缩，而达到类似正则化的效果。

### 其他方法

增加训练样本，也能在一定程度上预防过拟合问题的出现，但收集一些额外的样本有时也不是一件易事。这种情况下，使用现有的样本进行**数据扩充（Data Augmentation）**也不失为一种可行的办法。

![数据扩增法](https://ws1.sinaimg.cn/large/82e16446ly1g215uao0i9j20t7068te4.jpg)

如上图所示，数据扩充就是对已有的数据做一些简单的变换，例如对一张图片进行翻转、放大，以此就可以拥有更多的训练样本。

另外，某些情况下还可以用**早停止法（Early Stopping）**来预防过拟合:

![早停止法](https://ws1.sinaimg.cn/large/82e16446ly1g21a0a8sz1j20jv08vdgr.jpg)

训练出一个模型后，将其在训练集和验证集下分别进行梯度下降时成本变化曲线绘制在一起，从而找出两者开始出现较大偏差的一个时间点。如果有把握认为该时间点下训练出来的模型已经足够好，则可以在时间点下便停止学习。

***
#### 相关程序


#### 参考资料
1. [Andrew Ng-Machine Learning-Coursera](https://www.coursera.org/learn/machine-learning/)
2. [吴恩达-机器学习-网易云课堂](https://study.163.com/course/introduction/1004570029.htm)
3. [Andrew Ng-Improving Deep Neural Networks-Coursera](https://www.coursera.org/learn/deep-neural-network/)
4. [吴恩达-改善深层神经网络-网易云课堂](http://mooc.study.163.com/course/deeplearning_ai-2001281003#/info)
5. [机器学习中常常提到的正则化到底是什么意思？-知乎](https://www.zhihu.com/question/20924039/answer/131421690)
6. [关于范数的知识整理-百家号](https://baijiahao.baidu.com/s?id=1607333156323286278&wfr=spider&for=pc)
7. [机器学习中的范数规则化-CSDN](https://blog.csdn.net/zouxy09/article/details/24971995/)
8. [正则化之dropout(随机失活)详细介绍-CSDN](https://blog.csdn.net/sinat_29957455/article/details/81023154)
9. [神经网络正则化方法dropout细节探索-爱慕课](https://www.imooc.com/article/30129)


#### 更新历史
* 2019.04.13 完成初稿

