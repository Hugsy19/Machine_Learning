前面介绍梯度下降法时，曾提到过当目标函数（即成本函数）存在多个极值点时，下降到最后可能落入局部最小值点上：

![局部最小值](https://i.loli.net/2019/05/05/5cce9ddfc296a.jpg)

而事实上，深度学习里面涉及的模型通常是高维度的，目标函数中存在的**鞍点（saddle point）**往往比极值点要多得多：

![鞍点](https://i.loli.net/2019/05/05/5cce9dec64642.jpg)

如此看来，在深度学习中要找到目标函数的全局最优解异常困难，这也是深度学习一直面临的挑战之一，而实际中，也因此研究出来了许多针对梯度下降的优化算法。

### 小批量随机梯度下降

使用梯度下降法学习参数过程中，当训练样本数量$m$较小时，通常使用所有的样本来迭代更新参数值，此时用到所用的梯度下降被称为**批梯度下降（Batch Gradient Descent，BGD）**。然而当$m$比较大时，使用BGD的话每次迭代过程运算量会很大，而使训练过程变得缓慢，此时可以考虑将训练样本平均切分成几小批，用它们进行**小批量梯度下降（Mini-Batch Gradient Descent，MBGD）**。

将$m$个训练样本分为$k$小批（batch），除了余下的$m\%k$个外，每小批将有$t = m/k$个训练样本，在上标中用$\\{i\\}$标识第$i$小批，则MBGD的过程中有：$$X = \\{x^{\\{1\\}},x^{\\{2\\}},\cdots,x^{\\{k\\}}\\}$$ $$Y = \\{y^{\\{1\\}},y^{\\{2\\}},\cdots,y^{\\{k\\}}\\}$$

其中： $$x^{\\{k\\}} = [x^{(k)},x^{(2k)},\cdots,x^{(tk)}]$$ $$y^{\\{k\\}} = [y^{(k)},y^{(2k)},\cdots,y^{(tk)}]$$

之后，每次都使用一小批训练样本来更新参数，并在最后对剩余的$m\%k$个样本进行处理，这样用所有的训练样本完成一次迭代（iteration）的训练过程称为一个**epoch**。

MBGD中，每小批内的样本数量$t$是一个超参数，为了契合计算机的数据存储方式，通常取根据$m$的大小取$2$的幂次方。且通常使用MBGD得到的成本变化曲线会比使用BGD多出很多噪声：

![MBGD](https://i.loli.net/2019/05/05/5cce9dfbdaeed.jpg)

另外还有**随机梯度下降（Stochastic Gradient Descent，SGD）**，每次通过随机采样一个训练样本来进行参数更新，而大大加快训练速度。然而该方法难以收敛，或者很容易收敛到局部最优解上。

当前较常用的，是将小批量和随机梯度下降方法结合起来的**小批量随机梯度下降（Mini-Batch Stochastic Gradient Descent，MSGD)**。在MSGD中，通过随机均匀采样训练样本来得到一个小批量，期间可以设置允许重复采样与否。

### 指数加权移动平均
**指数加权移动平均（Exponentially Weight Moving Average，EWMA）**是一种经济学中常用的数据处理方式，它将加权平均和移动平均结合起来，把当前和前一段时期内的真实值进行指数加权后，用来平滑修改当前的值，从而生成平稳的趋势曲线。

在$t$时刻，移动平均值$v_t$的计算公式为：$$v_t = \beta v_{t-1} + (1-\beta)\theta_t$$

其中$\theta_t$为$t$时刻下的实际值，$\beta$为权值，它的大小决定了对过去数据的偏重程度。

例如收集到了伦敦一年时间里每日的气温数据，将这些按日期绘制成下图中右边的散点图：

![气温散点图](https://i.loli.net/2019/05/05/5cce9e080952e.jpg)

要根据这些数据绘制一条气温的变化趋势线，就可以用到上述的公式来计算每日的移动平均值。这里先直接令$v_0 = 0$，并取$\beta = 0.9$，则有：
$$v_1 = 0.9v_0 + 0.1\theta_1$$ $$\begin{aligned} v_2 & = 0.9v_1 + 0.1 \theta_2 \\\ & = 0.9 \times 0.1\theta_1 +  0.1 \theta_2\end{aligned}$$ $$ \vdots $$ $$\begin{aligned} v_{100} & = 0.9v_{99} + 0.1 \theta_{100} \\\ & = (0.9)^{99} \times 0.1\theta_{1} + \cdots + 0.9 \times 0.1\theta_{99} + 0.1\theta_{100}\end{aligned}$$ $$ \vdots $$

如此就可以得到图中的红色曲线。从上面的式子可以看出，各真实值的权值中，均会包含$\beta$的幂次方项。

令$n = \frac{1}{1 - \beta}$，则有：$$(1 - \frac{1}{n})^n = \beta^{\frac{1}{1 - \beta}}$$

根据重要极限，有：$$\lim_{n \to \infty} (1- \frac{1}{n})^n = \frac{1}{e} \approx 0.3679 $$

当$n \to \infty$时$\beta \to 1$，则：$$\lim_{\beta \to 1} \beta^{\frac{1}{1 - \beta}} = \frac{1}{e}$$

所以当$\beta = 0.9$，有：$$\beta^{\frac{1}{1 - \beta}}=0.9^{10} \approx \frac{1}{e}$$

将此时也就是$\beta$取$\frac{1}{1 - \beta}$次方时得到的值认为已经足够小，从而把EWMA公式中该项以及后面更高阶的项作为权值求出的结果均忽略不计。由此，可以认为由EWMA公式得到的$t$时刻的移动平均值$v_t$，是对该时刻前的$\frac{1}{1 - \beta}$个$\theta$值进行指数加权平均后而得到，且距离时刻$t$越近，其权值也就越大。

由此，原来得到的红色变化趋势线相当把过去 $\frac{1}{1 - 0.9} = 10$天的气温值指数加权平均后作为当日的气温。

想得到更为平滑的变化趋势线，可以增大$\beta$的值。例如$\beta = 0.98$时，可得下图中绿色曲线。而当$\beta=0.5$时，会得到图中波动剧烈的黄色曲线。$\beta$越大则越偏重于考察以往的数据变化，从而使整体的变化趋势不易于受数据中某些噪点的影响。

![变化趋势线](https://i.loli.net/2019/05/05/5cce9e188d783.jpg)

要注意，如果将$v_0$直接初始化为$0$，前期的运算将产生一定的偏差。该偏差可通过下式进行修正：$$\hat{v}\_t = \frac{v\_t}{1-\beta^t}$$

修正后，对应上面的例子，将得到下面紫色的变化曲线：
![偏差校正](https://i.loli.net/2019/05/05/5cce9e260cfd4.jpg)

当$t$不断增大时，$\beta^t$的值趋近于$0$，所以随着时间推移紫色的曲线就慢慢与原来的绿色的变化曲线重合了。

### 动量法
对某些目标函数，使用梯度下降参数学习时，如果某个方向上的梯度值过大，训练过程中会出现如下图所示的上下震荡，而导致收敛速度变得十分缓慢：

![Hassian病态矩阵](https://i.loli.net/2019/05/05/5cce9e37ba862.jpg)

这种情况下，调整学习率可能又会影响到其他方向上的梯度值。可行的优化方法之一，就是引入上面所述的EWMA，采用论文[[On the momentum term in gradient descent learning algorithms](https://www.sciencedirect.com/science/article/pii/S0893608098001166/pdfft?md5=c301bb4f47a9c792c7bae7b878b57a28&pid=1-s2.0-S0893608098001166-main.pdf)]中提出的带**动量（Momentum）**的梯度下降，来使下降的过程变得更为平缓。

使用动量法，梯度下降后更新参数的过程将变为：$$v_{dW_t} = \beta v_{dW_{t-1}} + (1-\beta)dW_t$$ $$v_{db_t} = \beta v_{db_{t-1}} + (1-\beta)db_t$$ $$W_t = W_{t-1} - \alpha v_{dW_t}$$ $$ b_t = b_{t-1}  -\alpha v_{db_t}$$

其中的$v_t$代表各梯度值的EWMA值，初始值$v_0 = 0$。超参数也就是EWMA中的权值$\beta$通常取$0.9$，学习率$\alpha$则依然需要根据实际情况进行调参。

使用带动量的梯度下降，梯度下降的过程将变成下图中红色的曲线所示：

![Momentum](https://i.loli.net/2019/05/05/5cce9e57ee134.jpg)

引入动量法后，更新参数时用到梯度值将是前面几次的梯度值进行EWMA后的值，这样能有效平滑下降过程，从而提高收敛速度。

### AdaGrad算法
进行参数学习时，对于某些方向上的梯度值存在的较大差别的问题，动量法中是利用EWMA来使得参数更新的方向更一致来解决。在论文[[Adaptive Subgradient Methods forOnline Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)]中提出的AdaGrad算法，则根据各方向上的梯度值来动态调整各方向上的学习率，来解决该问题。

使用AdaGrad算法进行参数时，有：$$s_{dW_t} = s_{dW_{t-1}} + dW\_t^2$$ $$s_{db_t} = s_{db_{t-1}} + db\_t^2$$ $$W_t = W_{t-1}  - \frac{\alpha}{\sqrt{s_{dW_t}+\epsilon}}dW_t$$ $$b_t = b_{t-1}  - \frac{\alpha}{\sqrt{s_{db_t}+\epsilon}} db_t$$

其中的$s_t$是各梯度值的平方累加值，初始值$s_0 = 0$，用它作分母来控制学习率的大小。当目标函数某个方向上的梯度较大时，学习率也会减小得较快，反之学习率将减小得较慢。另外，$\epsilon$用来维护数据稳定性，防止$s_t = 0$时出现分母为$0$的错误，通常可以取$\epsilon=10^{-8}$。

使用AdaGrad算法迭代到后期，学习率将变得很小，此时可能将难以找到一个有用的解。

### RMSProp算法
为解决AdaGrad算法后期难以找到有用解的问题，深度学习邻域的三巨头之一——Geoff Hinton在Coursera上讲授的[Neural Networks for Machine Learning]课程中首次提出了**RMSProp（Root Mean Square Prop）**算法，它在AdaGrad的基础上引入了动量法中用所用的EWMA。具体的参数更新过程为：$$s_{dW_t} = \beta s_{dW_{t-1}} + (1-\beta)dW^2\_t$$ $$s_{db_t} = \beta s_{db_{t-1}} + (1-\beta)db\_t^2$$ $$W_t := W_{t-1}  - \frac{\alpha}{\sqrt{s_{dW_t}+\epsilon}}dW_t$$ $$b_t := b_{t-1} - \frac{\alpha}{\sqrt{s_{db_t}+\epsilon}} db_t$$

这里的$s_t$表示的是各梯度值平方后的EWMA值且$s_0 = 0$。添加$\epsilon$的目的和前面一样，且通常可以取$\epsilon=10^{-8}$。

### AdaDelta算法
和RMSProp算法一样，论文[[ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/pdf/1212.5701)]中提出的AdaDelta算法，也是在AdaGrad的基础上进行改进。更新参数时，前面的过程和RMSProp算法一致：$$s_{dW_t} = \beta s_{dW_{t-1}} + (1-\beta)dW^2\_t$$ $$s_{db_t} = \beta s_{db_{t-1}} + (1-\beta)db\_t^2$$

不同的是，AdaDelta算法中新增了$\Delta$项来替代学习率$\alpha$，从而求得各梯度的变化量：
$$dW_t' = \sqrt{\frac{\Delta dW_{t-1} + \epsilon}{s_{dW_t}+\epsilon}} dW_t$$ $$ db_t' = \sqrt{\frac{\Delta db_{t-1} + \epsilon}{s_{db_t}+\epsilon}} db_t$$ 

其中$\epsilon$的作用依然和前面的算法相同，可以取$\epsilon=10^{-8}$。随后，更新参数，有：$$W_t = W_{t-1}  - dW_t'$$ $$b_t = b_{t-1} - db_t'$$

而$\Delta$项则记录各梯度的变化量平方后的EWMA值，其在$t=0$时的初始值为$0$：$$\Delta dW_t = \beta \Delta dW_{t-1} + (1-\beta)dW'^2\_t$$ $$\Delta db_t = \beta \Delta db_{t-1} + (1-\beta)db'^2\_t$$

### Adam算法
论文[[Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980v8)]中提出的**Adam(Adaptive Moment Estimation)**算法中，结合了动量法和RMSProp，该算法适用于多种深度学习模型。首先分别计算各梯度值、各梯度值平方后的EWMA值$v_t$、$s_t$：$$v_{dW_t} = \beta_1 v_{dW_{t-1}} + (1-\beta_1)dW_t $$ $$v_{db_t} = \beta_1 v_{db_{t-1}} + (1-\beta_1)db_t$$ $$s_{dW_t} = \beta_2 s_{dW_{t-1}} + (1-\beta_2)dW\_t^2$$ $$s_{db_t} = \beta_2 s_{db_{t-1}} + (1-\beta_2)db\_t^2$$ 

上式中，超参数$\beta_1$被称为**第一阶矩**，一般取$0.9$，$\beta_2$被称为**第二阶矩**，一般取$0.999$。不同于前面几种用到EWMA的算法，这里需要对计算出来的EWMA值进行偏差修正，而得到$\hat{v}\_t$、$\hat{s}\_t$：$$\hat{v}\_{dW\_t} = \frac{v_{dW_t}}{(1-\beta_1^t)},\ \ \ \hat{v}\_{db\_t} = \frac{v_{db_t}}{(1-\beta_1^t)}$$ $$\hat{s}\_{dW\_t} = \frac{s_{dW_t}}{(1-\beta_2^t)},\ \ \ \hat{s}\_{db\_t} = \frac{s_{db_t}}{(1-\beta_2^t)}$$ 

最后更新参数：$$W_t = W_{t-1} - \alpha \frac{\hat{v}\_{dW\_t}}{\sqrt{\hat{s}\_{dW\_t} +\epsilon}}$$ $$b_t = b_{t-1} - \alpha \frac{\hat{v}\_{db\_t}}{\sqrt{\hat{s}\_{db\_t} +\epsilon}}$$

其中的学习率$\alpha$需要根据实际情况进行调参，$\epsilon$的作用和前面一样，且通常取$\epsilon=10^{-8}$。

下图是上面所述的几种算法在某个目标函数上的实际表现：

![算法对比](https://i.loli.net/2019/05/05/5cce9e8b14391.gif)

***
#### 相关程序
* [Ng-DL2-week2-梯度下降优化](https://github.com/BinWeber/Machine_Learning/blob/master/Ng_Deep_Learning/2_Neural_Network_Improve/week_2/Gradient_Improving_Methods.ipynb)

#### 参考资料
1. [吴恩达-改善深层神经网络-网易云课堂](http://mooc.study.163.com/course/deeplearning_ai-2001281003#/info)
2. [Andrew Ng-Improving Deep Neural Networks-Coursera](https://www.coursera.org/learn/deep-neural-network/)
3. [动手学深度学习](http://zh.d2l.ai/index.html)
4. [梯度下降法的三种形式-博客园](http://www.cnblogs.com/maybe2030/p/5089753.html)
5. [优化算法之指数移动加权平均-知乎专栏](https://zhuanlan.zhihu.com/p/32335746)
6. [梯度下降优化算法综述-博客园](https://www.cnblogs.com/ranjiewen/p/5938944.html)

#### 更新历史
* 2019.04.16 完成初稿
