分类问题的输出值是离散的，其中最简单的是**二分类（binary classification）**问题，其分类结果只有两种，一般选其中一类作为**正类（positive class）**，并使其标记$y = 1$，另一类则作为**反类（negative class)**，使其标记$y = 0$，因而$y \in \lbrace0, 1\rbrace$，该类问题常用**Logistic回归**模型来解决。

此外，还有更为**多分类（multi-classfication)**问题，其分类结果常用**独热编码（one-hot code)**的形式来表示，且该类问题常用**Softmax回归**模型来解决。

由于历史原因，上面提及的模型虽然也称为“回归”，却是用以分类问题的学习算法，要注意将其区分。

### 假设函数
我们知道，分类与回归同属于监督学习问题，两者只是在输出的值上有所区别。因此先考虑继续用线性回归算法来解决分类问题：

![线性回归](https://ws1.sinaimg.cn/large/82e16446ly1g1cqx7qv09j20uy0aa0x3.jpg)

上图中的要根据已有数据构建一个肿瘤分类器，由训练样本得到其中红线所示的线性回归模型后，为了达到将连续的输出离散化，可以在中间设定一个阈值，使预测结果高于该阈值的输出为正类，低于则输出为反类，以此达到分类的效果。

然而该模型很容易受到一些特殊情况的影响，而产生图中的蓝线所示的偏差较大的结果。且线性回归模型的输出值不都被限定在$0$到$1$之间，与分类问题中的要求不符。

沿用线性回归中的假设函数$w^Tx + b$，而另外采用**sigmoid函数**将其值约束在$0$到$1$的范围之内。sigmoid函数又称Logistic函数，其表达式及图像如下：$$\sigma{(z)} = \frac{1}{1 + e^{-z}}$$

![sigmoid函数图像](https://i.loli.net/2019/01/10/5c37550ace941.png)

由sigmoid函数的图像可知，其值一直保持在$0$到$1$之间。由此就有了我们Logistic回归中用到的假设函数：$$\hat{y} = σ(w^Tx + b) = \frac{1}{1+e^{-(w^Tx + b)}}$$

其中预测结果$\hat{y}$的值在此被赋予了新的含义，它表示的是给定参数$w$、$b$，且输入为$x$的条件下，标签$y=1$的概率，用数学表达式表示即为：$$\hat{y} = P(y=1|x;w, b)$$

例如对训练好的肿瘤分类器输入肿瘤的大小$x$后，输出的$h_\theta(x) = 0.7$，就表示分类器预测这种大小的肿瘤有$70\%$的概率是恶性的。

### 决策界限
有了适用上面所述的假设函数$\hat{y}$，还需要设定一个阈值作为分类的标准，以将假设函输出的连续值进一步转化为离散的标记值。注意到，sigmoid函数有以下性质：
* $z \ge 0$时，$0.5 \le \sigma(z) \lt 1$
* $z \lt 0$时，$0 \lt \sigma(z) \lt 0.5$

进而有：
* $w^Tx + b \ge 0$时，$0.5 \le \hat{y} \lt 1$
* $w^Tx + b\lt 0$时，$0 \lt \hat{y} \lt 0.5$

因此可粗略地将$0.5$设为阈值，也就当某个输入对应的预测结果大于或等于$0.5$时，我们就认为它是正类，否则为反类。

![决策边界](https://ws1.sinaimg.cn/large/82e16446ly1g1ctumieloj20s807t40h.jpg)

如果由上图左边的训练样本，得到了右边成本函数中的各参数值，那么当$0.5 \le \hat{y}$时，有：$$ w^Tx + b  = -3 + x_1 + x_2 \ge 0$$
即：$$x_1 + x_2 \ge 3$$

该不等式取的是图中分界线以上的部分，也就是正类所在的区域。同理$h_\theta(x) \lt 0.5$时，将取图中的反类。图中的那条分界线，称为两类的**决策界限（decision boundary）**，显然，其位置由参数$w$、$b$的值决定。

此外，sigmoid函数可用于非线性函数，而得到复杂的分类问题的决策界限：
![非线性决策边界](https://ws1.sinaimg.cn/large/82e16446ly1g1cuznrzsqj20ua0at0yk.jpg)

### 成本函数
Logistic回归中假设函数$\hat{y}$是非线性的，如仍采用均方误差作为损失函数，得到的成本函数会是非凸（non-convex）函数，难以找到最优解。

由$\hat{y}$的值在Logistic回归中赋予的新含义，我们希望满足如下条件概率：$$P(y^{(i)}|x^{(i)}; w, b) = \begin{cases} \hat{y}^{(i)},  & \text{$y^{(i)}=1$} \\\ 1 - \hat{y}^{(i)}, & \text{$y^{(i)}=0$} \end{cases}$$

将上下两个式子合二为一，可写成：$$P(y^{(i)}|x^{(i)}; w,b) = \hat{y}^{{(i)}y^{(i)}} (1 - \hat{y}^{(i)})^{(1 - y^{(i)})} $$

该式就是Logistic回归中目标函数（分布函数）。为获得唯一的最优模型，需要对模型中的参数进行估计。

引入数理统计中的参数估计方法——**[最大似然估计法](https://baike.baidu.com/item/最大似然估计)（Maximum Likelihood Estimation）**，有如下似然函数：$$\prod_{i=1}^m P(y^{(i)}|x^{(i)}; w,b) = \prod_{i=1}^m  \hat{y}^{{(i)}y^{(i)}} (1 - \hat{y}^{(i)})^{(1 - y^{(i)})} $$

对上式两边同取对数，有：
$$\log P(y^{(i)}|x^{(i)}; w,b) = \sum_{i = 1}^m y^{(i)}\log \hat{y}^{(i)}+(1-y^{(i)})\log\ (1-\hat{y}^{(i)}) $$

最大似然估计法中，后面的步骤是使上式对各参数的导数等于$0$,而得到使上式的值最大化的参数值，而对于成本函数，我们希望找到使它的值最小化的参数值。因此，在上式中添加负号，并取所有训练样本的平均值，就可以将它作为Logistic回归中的成本函数：$$\mathcal{J}(w,b) =  - \frac{1}{m} \sum_{i=1}^m y^{(i)}\log\hat{y}^{(i)} - (1-y^{(i)}) \log(1-\hat{y}^{(i)})$$ 

其中右半部分被称为**交叉熵（cross entropy）**损失函数：$$\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) = - y^{(i)}\log\hat{y}^{(i)} - (1-y^{(i)}) \log(1-\hat{y}^{(i)}) $$

该损失函数又可写成：$$\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) = \begin{cases} - \log \hat{y}^{(i)},& y^{(i)} = 1 \\\ - \log(1-\hat{y^{(i)}}), & y^{(i)} = 0 \end{cases}$$

当$y = 1$即某个对象为正类时，该函数图像为：

![成本函数图像1](https://ws1.sinaimg.cn/large/82e16446ly1g1d0pn8akyj20d809eq33.jpg)

此时，当$\hat{y}=1$时损失为$0$，而$\hat{y} \to 0$时，成本趋近$\infty$。

反之，当$y = 0$时，函数图像为：

![成本函数图像2](https://ws1.sinaimg.cn/large/82e16446ly1g1d0p5bnjhj20de09dwen.jpg)

此时，当$\hat{y}=0$时损失为$0$，而$\hat{y}\to 1$时，成本趋近$\infty$。

### 梯度下降
有了成本函数后，依然采用梯度下降法来将其最小化，以学习Logistic回归模型中的参数值。

下面对该成本函数的导数进行推导：

为方便计算，成本函数中的$\log$默认以$\exp$为底。展开其中的几项，有：$$ \log \hat{y}^{(i)} = \log \frac{1}{1 + e^{-(w^Tx^{(i)} + b)}} = -\log(1 + e^{-(w^Tx^{(i)} + b)}) $$ $$\begin{aligned} \log(1 - \hat{y}^{(i)}) & = \log \frac{e^{-(w^Tx^{(i)} + b)}}{1 + e^{-(w^Tx^{(i)} + b)}} \\\ &= \log e^{-(w^Tx^{(i)} + b)} - \log (1 + e^{-(w^Tx^{(i)} + b)}) \\\ & = -(w^Tx^{(i)} + b) - \log (1 + e^{-(w^Tx^{(i)} + b)}) \end{aligned}$$

由此：$$\begin{aligned} \mathcal{J}(w, b) & = -\frac{1}{m} \sum_{i=1}^m [-y^{(i)} \log(1 + e^{-(w^Tx^{(i)} + b)}) - (1 - y^{(i)})(w^Tx^{(i)} + b + \log (1 + e^{-(w^Tx^{(i)} + b)}))] \\\ & = -\frac{1}{m}\sum_{i=1}^m [y^{(i)}(w^Tx^{(i)} + b) - (w^Tx^{(i)} + b) - \log (1 + e^{-(w^Tx^{(i)} + b)})] \\\ &= -\frac{1}{m}\sum_{i=1}^m [y^{(i)}(w^Tx^{(i)} + b) - (\log e^{(w^Tx^{(i)} + b)} + \log (1 + e^{-(w^Tx^{(i)} + b)}))] \\\ & = \frac{1}{m}\sum_{i=1}^m [\log (1 + e^{(w^Tx^{(i)} + b)}) - y^{(i)}(w^Tx^{(i)} + b)] \end{aligned}$$

对其求导，有：$$ \begin{aligned} \frac{\partial}{\partial w_j} \mathcal{J}(w, b) & = \frac{1}{m}\sum_{i=1}^m [\frac{\partial}{\partial w_j} \log (1 + e^{(w^Tx^{(i)} + b)}) - \frac{\partial}{\partial w_j} y^{(i)}(w^Tx^{(i)} + b)] \\\ &  =  \frac{1}{m}\sum_{i=1}^m \frac{x^{(i)}\_j e^{(w^Tx^{(i)} + b)}}{1 + e^{(w^Tx^{(i)} + b)}} - y^{(i)}x^{(i)}\_j \\\ &  = \frac{1}{m}\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})x^{(i)}\_j \end{aligned}$$ 

$$ \frac{\partial}{\partial b} \mathcal{J}(w, b) = \frac{1}{m}\sum_{i=1}^m \hat{y}^{(i)} - y^{(i)}$$

从而梯度下降更新参数值的过程为：$$\begin{aligned} & \text{Repeat} \ \lbrace \\\ & \ \ \ \ w_j := w_j - \frac{\alpha}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} \\\ & \ \ \ \ b := b - \frac{\alpha}{m} \sum_{i=1}^m \hat{y}^{(i)} - y^{(i)}\\\ & \rbrace \\\ & \text{直到其中各参数的值收敛}\end{aligned}$$

除了梯度下降以外，还存在一些常用的较复杂的优化算法，如**共轭梯度法（onjugate gradient）**、**BFGS**、**L-BFGS**等，它们都可以用来代替梯度下降法进行参数学习。

### Softmax回归

### 应用：猫图分类器
用Logistic回归构建一个猫图分类器，即输入一张图片，希望该分类器准确判断出该图片是否是一张猫图。

拥有的训练样本是一些分辨率为$64 \times 64$的猫图。图片是一类非结构化的数据，一张图片在计算机中以RGB编码时，是将每个像素点上三基色对应的量（即“亮度”）编码为数字后进行存储。因此在计算机上，一张图片可以用大小与图片一致的三个矩阵来表示：

![猫图分类器](https://ws1.sinaimg.cn/large/82e16446ly1fjers3qr63j20r706e41q.jpg)

还是用一个特征向量来表示一个训练样本，那么代表一张猫图的特征向量可由上面提及的三个矩阵拆分重塑而得，其维数$n_x = 64 \times 64 \times 3 = 12288$：

![一张猫图的特征向量](https://i.loli.net/2019/01/10/5c3720add829f.png)

为方便运算，通常在训练模型时将所有的训练样本向量化后再矩阵化，也就是把它们放到一个矩阵中。对这个猫图分类器，将$m$个训练样本都向量化为$n_x$维的特征向量后，可以进一步将它们矩阵化为一个$m \times n_x$维的样本空间$X$。在Logistic回归中，通常将参数$w$、$b$初始化为$0$，因此将$w$初始化为$n_x$维的零向量，$b$则可以利用python等编程语言中**广播（broadcast）**机制，直接初始化为常数$0$。这样，成本函数中的累加过程不经过编程语言中费时的for循环，而直接经过矩阵点乘而实现。

***
#### 相关程序
* [Ng-DL1-week2-猫图分类器](https://github.com/BinWeber/Machine_Learning/blob/master/Ng_Deep_Learning/1_Neural_Network/week_2/Cat_Classification_Logistic_Regression_Numpy.ipynb)

#### 参考资料
1. [Andrew Ng-Machine Learning-Coursera](https://www.coursera.org/learn/machine-learning/)
2. [Andrew Ng-Neural Networks and Deep Learning-Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/)
3. [吴恩达-机器学习-网易云课堂](https://study.163.com/course/introduction/1004570029.htm)
4. [吴恩达-神经网络与深度学习-网易云课堂](http://mooc.study.163.com/learn/deeplearning_ai-2001281002)
5. [交叉熵损失函数-博客园](https://www.cnblogs.com/lijie-blog/p/10166002.html)
6. [交叉熵代价函数(损失函数)及其求导推导-CSDN](https://blog.csdn.net/jasonzzj/article/details/52017438)
7. [牛顿法 拟牛顿法DFP BFGS L-BFGS的理解-CSDN](https://blog.csdn.net/ustbbsy/article/details/82497872)

#### 更新历史
* 2019.04.08 完成初稿
