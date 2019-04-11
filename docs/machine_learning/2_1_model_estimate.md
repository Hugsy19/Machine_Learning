在机器学习中，训练出来的模型经常存在下面几种情况：
![回归问题](https://ws1.sinaimg.cn/large/82e16446ly1g1dqurgbcfj20ut09ogp3.jpg)

对上面的回归问题，最为理想的情况是中间所示的模型，另外两种情况则分别称为：
* **欠拟合（underfitting）**：模型与样本间存在较大的**偏差（bias）**，如上面的左图
* **过拟合（overfitting）**：模型与样本间存在较大的**方差（variance）**，如上面的右图

用Logistic回归解决分类问题时，这些情况也是存在的：
![分类问题](https://ws1.sinaimg.cn/large/82e16446ly1g1drbasaf8j20vs0c6gvi.jpg)

其中，过拟合将导致模型的**泛化（generalization）**能力不够强，而它很容易在训练样本包含的特征种类过多而训练样本的数量却相对较少时发生，因此常用来预防过拟合的方法有：
* 通过人工筛选或模型选择算法丢弃部分训练样本的特征
* **正则化（regularization）**成本函数

### 正则化
![过拟合](https://ws1.sinaimg.cn/large/82e16446ly1g1dxy8qp7gj20pj08qq4o.jpg)

上面的右图所示的回归模型明显存在过拟合，要解决该问题，最简单的做法就是想办法使模型中的参数$\theta_3$、$\theta_4$尽可能地接近于$0$，而使其近似地等于左图中最为理想的模型。

进一步地，我们知道，参数$\theta$的值都是通过最小化成本函数而求得的。那么要达到上述目的，可以考虑将这几个参数“捆绑”在成本函数上：$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\theta_3^2 + 1000\theta_4^2$$

这样，在最小化成本函数时，参数$\theta_3$、$\theta_4$将受到“惩罚”而一同被最小化，以此达到防止过拟合的目的。所谓的正则化过程也与此类似。

建立机器学习模型时，学习到的参数值较小，就意味着假设函数会是一个相对“简单”的函数，过拟合也更不容易发生。正则化的思想即在于此，其具体做法是在成本函数后加入正则化项，对线性回归模型为：$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{i=1}^n \theta_j^2$$

对Logistic回归模型则为：$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m} \sum_{i=1}^n \theta_j^2$$

其中$\lambda$为正则化参数，需选取适当的值，其值过大容易导致欠拟合；$\sum_{i=1}^n \theta_j^2$是最常用的正则化项，它被称为**L2范数**，另有**L0、L1范数**。正则化保留了所有的特征，而通过使所有参数值最小化来防止过拟合。

对线性回归或Logistic回归模型，用梯度下降法最小化正则化后的成本函数的过程均为：$$\begin{aligned} & \text{Repeat}\ \lbrace \\\ & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \\\ & \ \ \ \  \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace \\\ & \rbrace \end{aligned}$$

由于$\theta_0$的值恒为$1$，不需要将它正则化，所以迭代过程种分成了两步。

另外，采用正规方程直接求解线性回归模型中的参数$\theta$时，进行正规化的表达式为：$$\theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty$$ 
其中$L$是个大小为$(n+1)\times(n+1)$的矩阵：$$L = \begin{bmatrix} 0 & & & & \\\ & 1 & & & \\\ & & 1 & & \\\ & & & \ddots & \\\ & & & & 1 \\\ \end{bmatrix}$$

前面提到过正规方程种$X^TX$为奇异矩阵或非方阵时，它将不存在逆矩阵。对正规方程进行正规化后，就不会出现这种情况，$X^TX + \lambda \cdot L$将一定是可逆的。

### 模型估计
![分类问题](https://ws1.sinaimg.cn/large/82e16446ly1fk3xje7ndkj20pt09zq44.jpg)
如图中的左图，对图中的数据采用一个简单的模型，例如线性拟合，并不能很好地对这些数据进行分类，分类后存在较大的**偏差（Bias）**，称这个分类模型**欠拟合（Underfitting）**。右图中，采用复杂的模型进行分类，例如深度神经网络模型，当模型复杂度过高时变容易出现**过拟合（Overfitting）**，使得分类后存在较大的**方差（Variance）**。中间的图中，采用一个恰当的模型，才能对数据做出一个差不多的分类。

![偏差和方差](https://ws1.sinaimg.cn/large/82e16446ly1fk3xjdw5ksj20oq0bh0vr.jpg)

通常采用开发集来诊断模型中是否存在偏差或者时方差：
* 当训练出一个模型后，发现训练集的错误率较小，而开发集的错误率却较大，这个模型可能出现了过拟合，存在较大方差;
* 发现训练集和开发集的错误率都都较大时，且两者相当，这个模型就可能出现了欠拟合，存在较大偏差；
* 发现训练集的错误率较大时，且开发集的错误率远较训练集大时，这个模型就有些糟糕了，方差和偏差都较大。
* 只有当训练集和开发集的错误率都较小，且两者的相差也较小，这个模型才会是一个好的模型，方差和偏差都较小。

模型存在较大偏差时，可采用增加神经网络的隐含层数、隐含层中的节点数，训练更长时间等方法来预防欠拟合。而存在较大方差时，则可以采用引入更多训练样本、对样本数据**正则化（Regularization）**等方法来预防过拟合。

### 预防过拟合

#### L2正则化

向Logistic回归的成本函数中加入L2正则化（也称“L2范数”）项：$${J(w,b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}({\hat y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}||w||\_2^2}$$
其中：$$||w||\_2^2 = \sum\_{j=1}^n{w\_j^2} = w^T w$$
L2正则化是最常用的正则化类型,也存在L1正则化项：$$\frac{\lambda}{m}||w||\_1 = \frac{\lambda}{m}\sum_{j=1}^n |w\_j|$$
由于L1正则化最后得到w向量中将存在大量的0，使模型变得稀疏化,所以一般都使用L2正则化。其中的参数$\lambda$称为正则化参数，这个参数通常通过开发集来设置。
向神经网络的成本函数加入正则化项：
$${J(w^{[1]},b^{[1]},...,w^{[L]},b^{[L]}) = \frac{1}{m} \sum\_{i=1}^m \mathcal{L}({\hat y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum\limits\_{l=1}^L ||w^{[l]}}||^2\_F$$

因为w是一个$n^{[l]} \times n^{[l-1]}$矩阵所以：$${||w^{[l]}||^2\_F = \sum\limits\_{i=1}^{n^{[l-1]}} \sum\limits\_{j=1}^{n^{[l]}} (w\_{ij}^{[l]})^2}$$
这被称为**弗罗贝尼乌斯范数（Frobenius Norm）**,所以神经网络的中的正则化项被称为弗罗贝尼乌斯范数矩阵。

加入正则化项后，反向传播时：$$dw^{[l]} = \frac{\partial \mathcal{L}}{\partial w^{[l]}} + \frac{\lambda}{m} w^{[l]}$$
更新参数时：$$ w^{[l]} := w^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial w^{[l]}} - \alpha \frac{\lambda}{m} w^{[l]}$$
有了新定义的$dw^{[l]}$，参数$w^{[L]}$在更新时会多减去一项$\alpha \frac{\lambda}{m} w^{[l]}$，所以L2正则化项也被称为**权值衰减（Weight Decay）**。

参数$\lambda$用来调整式中两项的相对重要程度，较小$\lambda$偏向于最后使原本的成本函数最小化，较大的$\lambda$偏向于最后使权值$w$最小化。当$\lambda$较大时，权值$w^{[L]}$便会趋近于$0$，相当于消除深度神经网络中隐藏单元的部分作用。另一方面，在权值$w^{[L]}$变小之下，输入样本$X$随机的变化不会对神经网络模造成过大的影响，神经网络受局部噪音的影响的可能性变小。这就是正则化能够降低模型方差的原因。

#### 随机失活正则化

**随机失活（DropOut）**正则化，就是在一个神经网络中对每层的每个节点预先设置一个被消除的概率，之后在训练随机决定将其中的某些节点给消除掉，得到一个被缩减的神经网络，以此来到达降低方差的目的。DropOut正则化较多地被使用在**计算机视觉（Computer Vision）**领域。

使用Python编程时可以用**反向随机失活（Inverted DropOut）**来实现DropOut正则化：

对于一个神经网络第3层
```python
keep.prob = 0.8
d3 = np.random.randn(a3.shape[0],a3.shape[1]) < keep.prob
a3 = np.multiply(a3,d3)
a3 /= keep.prob
z4 = np.dot(w4,a3) + b4
```
其中的d3是一个随机生成，与第3层大小相同的的布尔矩阵，矩阵中的值为0或1。而keep.prob ≤ 1，它可以随着各层节点数的变化而变化，决定着失去的节点的个数。

例如，将keep.prob设置为0.8时，矩阵d3中便会有20%的值会是0。而将矩阵a3和d3相乘后，便意味着这一层中有20%节点会被消除。需要再除以keep_prob的原因，是因为后一步求z4中将用到a3，而a3有20%的值被清零了，为了不影响下一层z4的最后的预期输出值，就需要这个步骤来修正损失的值，这一步就称为反向随机失活技术，它保证了a3的预期值不会因为节点的消除而收到影响，也保证了进行测试时采用的是DropOut前的神经网络。

与之前的L2正则化一样，利用DropOut，可以简化神经网络的部分结构，从而达到预防过拟合的作用。另外，当输入众多节点时，每个节点都存在被删除的可能，这样可以减少神经网络对某一个节点的依赖性，也就是对某一特征的依赖，扩散输入节点的权重，收缩权重的平方范数。

#### 数据扩增法

![数据扩增法](https://ws1.sinaimg.cn/large/82e16446ly1fk3xjdlp35j20pr0bgwiv.jpg)

**数据扩增（Data Augmentation）**是在无法获取额外的训练样本下，对已有的数据做一些简单的变换。例如对一张图片进行翻转、放大扭曲，以此引入更多的训练样本。

#### 早停止法

![早停止法](https://ws1.sinaimg.cn/large/82e16446ly1fk3xjcx912j20n10cqdh2.jpg)

**早停止（Early Stopping）**是分别将训练集和开发集进行梯度下降时成本变化曲线画在同一个坐标轴内，在箭头所指两者开始发生较大偏差时就及时进行纠正，停止训练。在中间箭头处，参数w将是一个不大不小的值，理想状态下就能避免过拟合的发生。然而这种方法一方面没有很好得降低成本函数，又想以此来避免过拟合，一个方法解决两个问题，哪个都不能很好解决。

***
#### 相关程序


#### 参考资料
1. [Andrew Ng-Machine Learning-Coursera](https://www.coursera.org/learn/machine-learning/)
2. [吴恩达-机器学习-网易云课堂](https://study.163.com/course/introduction/1004570029.htm)
3. [Andrew Ng-Improving Deep Neural Networks-Coursera](https://www.coursera.org/learn/deep-neural-network/)
4. [吴恩达-改善深层神经网络-网易云课堂](http://mooc.study.163.com/course/deeplearning_ai-2001281003#/info)
5. [机器学习中常常提到的正则化到底是什么意思？-知乎](https://www.zhihu.com/question/20924039/answer/131421690)
6. [机器学习中的范数规则化-CSDN](https://blog.csdn.net/zouxy09/article/details/24971995/)
8. [范数的定义-知乎回答](https://zhihu.com/question/20473040/answer/102907063)

#### 更新历史
* 2019.04.08 完成初稿

