
### 小批量随机梯度下降
使用梯度下降法学习参数过程中，当训练样本数量$m$较小时，通常使用所有的样本来迭代更新参数值，此时用到所用的梯度下降被称为**批梯度下降（Batch Gradient Descent，BGD）**。然而当$m$比较大时，使用BGD的话每次迭代过程运算量会很大，而使训练过程变得缓慢，此时可以考虑将训练样本平均切分成几小批，用它们进行**小批量梯度下降（Mini-Batch Gradient Descent，MBGD）**。

将$m$个训练样本分为$k$小批（batch），除了余下的$m\%k$个外，每小批将有$t = m/k$个训练样本，在上标中用$\\{i\\}$标识第$i$小批，则MBGD的过程中有：$$X = \\{x^{\\{1\\}},x^{\\{2\\}},\cdots,x^{\\{k\\}}\\}$$ $$Y = \\{y^{\\{1\\}},y^{\\{2\\}},\cdots,y^{\\{k\\}}\\}$$

其中： $$x^{\\{k\\}} = [x^{(k)},x^{(2k)},\cdots,x^{(tk)}]$$ $$y^{\\{k\\}} = [y^{(k)},y^{(2k)},\cdots,y^{(tk)}]$$

之后，每次都使用一小批训练样本来更新参数，并在最后对剩余的$m\%k$个样本进行处理，这样用所有的训练样本完成一次迭代（iteration）的训练过程称为一个**epoch**。

MBGD中，每小批内的样本数量$t$是一个超参数，为了契合计算机的数据存储方式，通常取根据$m$的大小取$2$的幂次方。且通常使用MBGD得到的成本变化曲线会比使用BGD多出很多噪声：

![MBGD](https://ws1.sinaimg.cn/large/82e16446ly1g22cfzh7apj20v70b70tk.jpg)

另外还有**随机梯度下降（Stochastic Gradient Descent，SGD）**，每次通过随机采样一个训练样本来进行参数更新，而大大加快训练速度。然而该方法难以收敛，或者很容易收敛到局部最优解上。

当前较常用的，是将小批量和随机梯度下降方法结合起来的**小批量随机梯度下降（Mini-Batch Stochastic Gradient Descent，MSGD)**。在MSGD中，通过随机均匀采样训练样本来得到一个小批量，期间可以设置允许重复采样与否。

### 指数加权平均

**指数加权平均（Exponentially Weight Average）**是一种常用的序列数据处理方式，其计算公式为：$$S\_t = \begin{cases} Y\_1, & t=1 \\\ \beta S\_{t-1} + (1-\beta)Y\_{t}, & t>1 \end{cases}$$
其中$Y_t$为t下的实际值，$S_t$为t下加权平均后的值，$\beta$为权重值。
给定一个时间序列，例如伦敦一年每天的气温值：

![气温时间图](https://ws1.sinaimg.cn/large/82e16446ly1fk9ig3abvhj20oa077jsf.jpg)

其中蓝色的点代表了真实的数据值。
对于一个即时的温度值，取权重值$\beta$为0.9，则有：$$ v\_0 = 0 $$ $$v\_1 = 0.9v\_0 + 0.1\theta\_1$$ $$... \ ... $$ $$v\_{100} = 0.1\theta\_{100}+0.1 \times 0.9\theta\_{99} +0.1 \times 0.9^2\theta\_{98} \ ...$$ $$ v\_t =  0.9v\_{t-1} + 0.1\theta\_t $$
根据：$$\lim_{\epsilon \to 0} (1-\epsilon)^{\frac{1}{\epsilon}} = \frac{1}{e} \approx 0.368$$

$\beta=1 - \epsilon = 0.9$时相当于把过去 $\frac{1}{\epsilon} = 10$天的气温值指数加权平均后，作为当日的气温，且只取10天前的气温值的$0.368$，也就是$\frac{1}{3}$多一些。

由此求得的值即得到图中的红色曲线，它反应了温度变化的大致趋势。

![EWA曲线](https://ws1.sinaimg.cn/large/82e16446ly1fk9j8j981jj20k00a1q4f.jpg)

当取权重值$\beta=0.98$时，可以得到图中更为平滑的绿色曲线。而当取权重值$\beta=0.5$时，得到图中噪点更多的黄色曲线。$\beta$越大相当于求取平均利用的天数就越多，曲线自然就会越平滑而且越滞后。

当进行指数加权平均计算时，第一个值$v_o$被初始化为$0$，这样将在前期的运算用产生一定的偏差。为了矫正偏差，需要在每一次迭代后用以下式子进行偏差修正：$$v_t := \frac{v_t}{1-\beta^t}$$

### 动量法

**动量梯度下降（Gradient Descent with Momentum）**是计算梯度的指数加权平均数，并利用该值来更新参数值。具体过程为：$$v_{dw} = \beta v_{dw} + (1-\beta)dw$$ $$v_{db} = \beta v_{db} + (1-\beta)db$$ $$w := w-\alpha v_{dw}$$ $$ b := b-\alpha v_{db}$$
其中的动量衰减参数$\beta$一般取0.9。

![Momentum](https://ws1.sinaimg.cn/large/82e16446ly1fk9mtb1xvcj20m204tq3q.jpg)

进行一般的梯度下降将会得到图中的蓝色曲线，而使用Momentum梯度下降时，通过累加减少了抵达最小值路径上的摆动，加快了收敛，得到图中红色的曲线。

当前后梯度方向一致时，Momentum梯度下降能够加速学习；前后梯度方向不一致时,Momentum梯度下降能够抑制震荡。

### RMSProp算法

**RMSProp(Root Mean Square Prop，均方根支)**算法在对梯度进行指数加权平均的基础上，引入平方和平方根。具体过程为：$$s\_{dw} = \beta s\_{dw} + (1-\beta)dw^2$$ $$s\_{db} = \beta s\_{db} + (1-\beta)db^2$$ $$w := w-\alpha \frac{dw}{\sqrt{s\_{dw}+\epsilon}}$$ $$b := b-\alpha \frac{db}{\sqrt{s\_{db}+\epsilon}}$$
其中的$\epsilon=10^{-8}$，用以提高数值稳定度，防止分母太小。

当$dw$或$db$较大时，$dw^{2}$、$db^{2}$会较大，造成$s\_{dw}$、 $s\_{db}$也会较大，最终使$\frac{dw}{\sqrt{s\_{dw}}}$、 $\frac{db}{\sqrt{s\_{db}}}$较小，减小了抵达最小值路径上的摆动。

### Adam优化算法
**Adam(Adaptive Moment Estimation，自适应矩估计)**优化算法适用于很多不同的深度学习网络结构，它本质上是将Momentum梯度下降和RMSProp算法结合起来。具体过程为：$$v\_{dw} = \beta\_1 v\_{dw} + (1-\beta\_1)dw, \ v\_{db} = \beta\_1 v\_{db} + (1-\beta\_1)db$$ $$s\_{dw} = \beta\_2 s_{dw} + (1-\beta\_2)dw^2,\ s\_{db} = \beta\_2 s\_{db} + (1-\beta\_2)db^2$$ $$v^{corrected}\_{dw} = \frac{v\_{dw}}{(1-\beta\_1^t)},\ v^{corrected}\_{db} = \frac{v\_{db}}{(1-\beta\_1^t)}$$ $$s^{corrected}\_{dw} = \frac{s\_{dw}}{(1-\beta\_2^t)},\ s^{corrected}\_{db} = \frac{s\_{db}}{(1-\beta\_2^t)}$$ $$w := w-\alpha \frac{v^{corrected}\_{dw}}{\sqrt{s^{corrected}\_{dw}}+\epsilon}$$ $$b := b-\alpha \frac{v^{corrected}\_{db}}{\sqrt{s^{corrected}\_{db}}+\epsilon}$$
其中的学习率$\alpha$需要进行调参，超参数$\beta\_1$被称为第一阶矩，一般取0.9，$\beta\_2$被称为第二阶矩，一般取0.999，$\epsilon$一般取$10^{-8}$。

### 学习率衰减

随着时间推移，慢慢减少学习率$\alpha$的大小。在初期$\alpha$较大时，迈出的步长较大，能以较快的速度进行梯度下降，而后期逐步减小$\alpha$的值，减小步长，有助于算法的收敛，更容易接近最优解。
常用到的几种学习率衰减方法有：$$\alpha = \frac{1}{1+\text{decay_rate }\* \text{epoch_num}} \* \alpha\_0$$ $$\alpha = 0.95^{\text{epoch_num}} \* \alpha\_0$$ $$\alpha = \frac{k}{\sqrt{\text{epoch_num}} }\* \alpha\_0$$
其中的decay_rate为衰减率，epoch_num为将所有的训练样本完整过一遍的次数。


### 梯度检验

梯度检验的实现原理，是根据导数的定义，对成本函数求导，有：$$ J'(\theta) = \frac{\partial J(\theta)}{\partial \theta}= \lim_{\epsilon\rightarrow 0}\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$$

则梯度检验公式：$$J'(\theta) = \frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$$

其中当$\epsilon$越小时，结果越接近真实的导数也就是梯度值。可以使用这种方法，来判断反向传播进行梯度下降时，是否出现了错误。

梯度检验的过程，是对成本函数的每个参数$\theta\_{[i]}$加入一个很小的$\epsilon$，求得一个梯度逼近值$d\theta\_{approx[i]}$：
$$d\theta\_{approx[i]} = \frac{J(\theta\_{[1]},\theta\_{[2]},...,\theta\_{[i]}+\epsilon)-J(\theta\_{[1]},\theta\_{[2]},...,\theta\_{[i]}-\epsilon)}{2\epsilon}$$

以解析方式求得$J'(\theta)$在$\theta$时的梯度值$d\theta$,进而再求得它们之间的欧几里得距离：
$$\frac{||d\theta\_{approx[i]}-d\theta||\_2}{||d \theta\_{approx[i]}||\_2+||dθ||\_2}$$

其中$||x||_2$表示向量x的2范数（各种范数的定义见参考资料）：
$$||x||\_2 = \sum\limits\_{i=1}^N |x\_i|^2$$

当计算的距离结果与$\epsilon$的值相近时，即可认为这个梯度值计算正确，否则就需要返回去检查代码中是否存在bug。

需要注意的是，不要在训练模型时进行梯度检验，当成本函数中加入了正则项时，也需要带上正则项进行检验，且不要在使用随机失活后使用梯度检验。
***
#### 相关程序


#### 参考资料
1. [吴恩达-改善深层神经网络-网易云课堂](http://mooc.study.163.com/course/deeplearning_ai-2001281003#/info)
2. [Andrew Ng-Improving Deep Neural Networks-Coursera](https://www.coursera.org/learn/deep-neural-network/)
3. [动手学深度学习](http://zh.d2l.ai/index.html)
4. [梯度下降法的三种形式-博客园](http://www.cnblogs.com/maybe2030/p/5089753.html)
5. [什么是批标准化-知乎专栏](https://zhuanlan.zhihu.com/p/24810318)

#### 更新历史
* 2019.04.04 完成初稿
