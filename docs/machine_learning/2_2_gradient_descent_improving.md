### 特征缩放
在训练机器学习模型的实际过程中，我们需要掌握一些技巧。**特征放缩（feature scaling）**便是其中之一，它可以有效地使梯度下降的过程变得更快。

所谓的特征放缩，是分别将训练数据中各个特征的值放缩到一个较小的统一区间内，该范围通常取$[-1,1]$，即使得$-1 \leq x_i \leq 1$。
![特征放缩](https://ws1.sinaimg.cn/large/82e16446gy1g117hj7tfwj20w00efac1.jpg)

如上图所示，进行特征放缩后，成本函数的等高线图将从椭圆变成圆形，而加快梯度下降的速度。

通常采用下式进行特征缩放：$$x_i := \frac{x_i}{max(x_i)-min(x_i)} $$
类似的还有**均值归一化（mean normalization）**，它能使放缩后的均值为$0$，其表达式为：$$x_i := \frac{x_i - mean(x_i)}{max(x_i)-min(x_i)} $$

### 标准化
对训练及测试集进行标准化的过程为：$$ \bar{x} = \frac{1}{m} \sum\_{i=1}^m x^{(i)} $$ $$ x^{(i)} := x^{(i)} - \bar{x} $$ $$ \sigma^2 = \frac{1}{m} \sum\_{i=1}^m {x^{(i)}}^2$$ $$x^{(i)}:= \frac{x^{(i)}}{\sigma^2} $$
原始数据为：
![原始数据](https://ws1.sinaimg.cn/large/82e16446ly1fk3xjcoozpj20c308yjrf.jpg)

经过前两步，x减去它们的平均值后：
![减去平均值](https://ws1.sinaimg.cn/large/82e16446ly1fk3xjckx23j20ch092wek.jpg)

经过后两步，x除以它们的方差后：
![除以方差](https://ws1.sinaimg.cn/large/82e16446ly1fk3xjcftu7j20d909vmxa.jpg)

数据集未进行标准化时，成本函数的图像及梯度下降过程将是：
![未进行标准化](https://ws1.sinaimg.cn/large/82e16446ly1fk3xjc4jazj20c40cbwff.jpg)

而标准化后，将变为：
![标准化后](https://ws1.sinaimg.cn/large/82e16446ly1fk3xjbq2p9j20bk0cjab1.jpg)

### 初始化权重

在之前的建立神经网络的过程中，提到权重w不能为0，而将它初始化为一个随机的值。然而在一个深层神经网络中，当w的值被初始化过大时，进入深层时呈指数型增长，造成**梯度爆炸**；过小时又会呈指数级衰减，造成**梯度消失**。

Python中将w进行随机初始化时，使用numpy库中的np.random.randn()方法，randn是从均值为0的单位标准正态分布（也称“高斯分布”）进行取样。随着对神经网络中的某一层输入的数据量n的增长，输出数据的分布中，方差也在增大。结果证明，可以除以输入数据量n的平方根来调整其数值范围，这样神经元输出的方差就归一化到1了，不会过大导致到指数级爆炸或过小而指数级衰减。也就是将权重初始化为：
```
w = np.random.randn(layers_dims[l],layers_dims[l-1]) \* np.sqrt(1.0/layers_dims[l-1])
```

这样保证了网络中所有神经元起始时有近似同样的输出分布。
当激活函数为ReLU函数时，权重最好初始化为：
```
w = np.random.randn(layers_dims[l],layers_dims[l-1]) \* np.sqrt(2.0/layers_dims[l-1])
```
以上结论的证明过程见参考资料。

### 梯度下降法

#### 批梯度下降法（BGD）

**批梯度下降法（Batch Gradient Descent，BGD）**是最常用的梯度下降形式，前面的Logistic回归及深层神经网络的构建中所用到的梯度下降都是这种形式。其在更新参数时使用所有的样本来进行更新，具体过程为：$${X = [x^{(1)},x^{(2)},…,x^{(m)}]}$$ $$z^{[1]} = w^{[1]}X + b^{[1]}$$ $$a^{[1]} = g^{[1]}(z^{[1]})$$ $$... \ ...$$ $$z^{[l]} = w^{[l]}a^{[l-1]} + b^{[l]}$$ $$a^{[l]} = g^{[l]}(z^{[l]})$$ $$ {J(\theta) = \frac{1}{m} \sum\_{i=1}^m \mathcal{L}({\hat y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum\limits\_{l=1}^L ||w^{[l]}}||^2_F$$ $$ {\theta\_j:= \theta\_j -\alpha\frac{\partial J(\theta)}{\partial \theta\_j}} $$
示例图：

![BGD](https://ws1.sinaimg.cn/large/82e16446ly1fk8n1qto83j20j209sq3s.jpg)

优点：最小化所有训练样本的损失函数，得到全局最优解；易于并行实现。
缺点：当样本数目很多时，训练过程会很慢。

#### 随机梯度下降法（SGD）

**随机梯度下降法（Stochastic Gradient Descent，SGD）**与批梯度下降原理类似，区别在于每次通过一个样本来迭代更新。其具体过程为：$${X = [x^{(1)},x^{(2)},…,x^{(m)}]}$$ $$ for\ \ \ i=1,2,...,m\ \\{ \ \ \ \ \ \ \ \ \ \ \ \ $$ $$z^{[1]} = w^{[1]}x^{(i)} + b^{[1]}$$ $$a^{[1]} = g^{[1]}(z^{[1]})$$ $$... \ ...$$ $$z^{[l]} = w^{[l]}a^{[l-1]} + b^{[l]}$$ $$a^{[l]} = g^{[l]}(z^{[l]})$$ $$ {J(\theta) = \mathcal{L}({\hat y}^{(i)}, y^{(i)}) + \frac{\lambda}{2} \sum\limits\_{l=1}^L ||w^{[l]}}||^2_F$$ $$ \theta\_j:= \theta\_j -\alpha\frac{\partial J(\theta)}{\partial \theta\_j} \\} $$
示例图：

![SGD](https://ws1.sinaimg.cn/large/82e16446ly1fk8n7rb8tvj20i709twfq.jpg)

优点：训练速度快。
缺点：最小化每条样本的损失函数，最终的结果往往是在全局最优解附近，不是全局最优；不易于并行实现。

#### 小批量梯度下降法（MBDG）

**小批量梯度下降法（Mini-Batch Gradient Descent，MBGD）**是批量梯度下降法和随机梯度下降法的折衷,对用m个训练样本，，每次采用t（1 < t < m）个样本进行迭代更新。具体过程为：$${X = [x^{\\{1\\}},x^{\\{2\\}},…,x^{\\{k = \frac{m}{t}\\}}]}$$ 其中： $$x^{\\{1\\}} = x^{(1)},x^{(2)},…,x^{(t)}$$ $$x^{\\{2\\}} = x^{(t+1)},x^{(t+2)},…,x^{(2t)}$$ $$... \ ...$$之后：$$ for\ \ \ i=1,2,...,k\ \\{ \ \ \ \ \ \ \ \ \ \ \ \ $$ $$z^{[1]} = w^{[1]}x^{\\{i\\}} + b^{[1]}$$ $$a^{[1]} = g^{[1]}(z^{[1]})$$ $$... \ ...$$ $$z^{[l]} = w^{[l]}a^{[l-1]} + b^{[l]}$$ $$a^{[l]} = g^{[l]}(z^{[l]})$$ $$ {J(\theta) = \frac{1}{k} \sum\_{i=1}^k \mathcal{L}({\hat y}^{(i)}, y^{(i)}) + \frac{\lambda}{2k} \sum\limits\_{l=1}^L ||w^{[l]}}||^2_F$$ $$ \theta\_j:= \theta\_j -\alpha\frac{\partial J(\theta)}{\partial \theta\_j} \\} $$
示例图：

![MBGD](https://ws1.sinaimg.cn/large/82e16446ly1fk8n62qc2rj20hr08zjsc.jpg)

样本数t的值根据实际的样本数量来调整，为了和计算机的信息存储方式相适应，可将t的值设置为2的幂次。将所有的训练样本完整过一遍称为一个**epoch**。

### 梯度下降优化

#### 指数加权平均

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

#### Momentum梯度下降

**动量梯度下降（Gradient Descent with Momentum）**是计算梯度的指数加权平均数，并利用该值来更新参数值。具体过程为：$$v\_{dw} = \beta v\_{dw} + (1-\beta)dw$$ $$v\_{db} = \beta v\_{db} + (1-\beta)db$$ $$w := w-\alpha v\_{dw}$$ $$ b := b-\alpha v\_{db}$$
其中的动量衰减参数$\beta$一般取0.9。

![Momentum](https://ws1.sinaimg.cn/large/82e16446ly1fk9mtb1xvcj20m204tq3q.jpg)

进行一般的梯度下降将会得到图中的蓝色曲线，而使用Momentum梯度下降时，通过累加减少了抵达最小值路径上的摆动，加快了收敛，得到图中红色的曲线。

当前后梯度方向一致时，Momentum梯度下降能够加速学习；前后梯度方向不一致时,Momentum梯度下降能够抑制震荡。

#### RMSProp算法

**RMSProp(Root Mean Square Prop，均方根支)**算法在对梯度进行指数加权平均的基础上，引入平方和平方根。具体过程为：$$s\_{dw} = \beta s\_{dw} + (1-\beta)dw^2$$ $$s\_{db} = \beta s\_{db} + (1-\beta)db^2$$ $$w := w-\alpha \frac{dw}{\sqrt{s\_{dw}+\epsilon}}$$ $$b := b-\alpha \frac{db}{\sqrt{s\_{db}+\epsilon}}$$
其中的$\epsilon=10^{-8}$，用以提高数值稳定度，防止分母太小。

当$dw$或$db$较大时，$dw^{2}$、$db^{2}$会较大，造成$s\_{dw}$、 $s\_{db}$也会较大，最终使$\frac{dw}{\sqrt{s\_{dw}}}$、 $\frac{db}{\sqrt{s\_{db}}}$较小，减小了抵达最小值路径上的摆动。

#### Adam优化算法
**Adam(Adaptive Moment Estimation，自适应矩估计)**优化算法适用于很多不同的深度学习网络结构，它本质上是将Momentum梯度下降和RMSProp算法结合起来。具体过程为：$$v\_{dw} = \beta\_1 v\_{dw} + (1-\beta\_1)dw, \ v\_{db} = \beta\_1 v\_{db} + (1-\beta\_1)db$$ $$s\_{dw} = \beta\_2 s_{dw} + (1-\beta\_2)dw^2,\ s\_{db} = \beta\_2 s\_{db} + (1-\beta\_2)db^2$$ $$v^{corrected}\_{dw} = \frac{v\_{dw}}{(1-\beta\_1^t)},\ v^{corrected}\_{db} = \frac{v\_{db}}{(1-\beta\_1^t)}$$ $$s^{corrected}\_{dw} = \frac{s\_{dw}}{(1-\beta\_2^t)},\ s^{corrected}\_{db} = \frac{s\_{db}}{(1-\beta\_2^t)}$$ $$w := w-\alpha \frac{v^{corrected}\_{dw}}{\sqrt{s^{corrected}\_{dw}}+\epsilon}$$ $$b := b-\alpha \frac{v^{corrected}\_{db}}{\sqrt{s^{corrected}\_{db}}+\epsilon}$$
其中的学习率$\alpha$需要进行调参，超参数$\beta\_1$被称为第一阶矩，一般取0.9，$\beta\_2$被称为第二阶矩，一般取0.999，$\epsilon$一般取$10^{-8}$。

#### 学习率衰减

随着时间推移，慢慢减少学习率$\alpha$的大小。在初期$\alpha$较大时，迈出的步长较大，能以较快的速度进行梯度下降，而后期逐步减小$\alpha$的值，减小步长，有助于算法的收敛，更容易接近最优解。
常用到的几种学习率衰减方法有：$$\alpha = \frac{1}{1+\text{decay_rate }\* \text{epoch_num}} \* \alpha\_0$$ $$\alpha = 0.95^{\text{epoch_num}} \* \alpha\_0$$ $$\alpha = \frac{k}{\sqrt{\text{epoch_num}} }\* \alpha\_0$$
其中的decay_rate为衰减率，epoch_num为将所有的训练样本完整过一遍的次数。

### 批标准化

**批标准化（Batch Normalization，BN）**和之前的数据集标准化类似，是将分散的数据进行统一的一种做法。具有统一规格的数据，能让机器更容易学习到数据中的规律。

对于含有$m$个节点的某一层神经网络，对$z$进行操作的步骤为：$$\mu = \frac{1}{m} \sum\_{i=1}^m z^{(i)}$$ $$\sigma^2 = \frac{1}{m} \sum\_{i=1}^m (z^{(i)}-\mu)^2$$ $$z\_{norm}^{(i)} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$$ $$\tilde{z}^{(i)} = \gamma z\_{norm}^{(i)}+\beta$$
其中的$\gamma$、$\beta$并不是超参数，而是两个需要学习的参数，神经网络自己去学着使用和修改这两个扩展参数。这样神经网络就能自己慢慢琢磨出前面的标准化操作到底有没有起到优化的作用。如果没有起到作用，就使用 $\gamma$和$\beta$来抵消一些之前进行过的标准化的操作。例如当$\gamma = \sqrt{\sigma^2+\epsilon}, \beta = \mu$，就抵消掉了之前的正则化操作。

![Batch Norm](https://ws1.sinaimg.cn/large/82e16446ly1fk9r4pxdodj20ik05smxu.jpg)

将图中的神经网络中的$z^{[1]}$、$z^{[2]}$进行批标准化后，$z^{[1]}$、$z^{[2]}$将变成$\tilde{z}^{[1]}$、$\tilde{z}^{[2]}$。

当前的获得的经验无法适应新样本、新环境时，便会发生“Covariate Shift”现象。对于一个神经网络，前面权重值的不断变化就会带来后面权重值的不断变化，批标准化减缓了隐藏层权重分布变化的程度。采用批标准化之后，尽管每一层的z还是在不断变化，但是它们的均值和方差将基本保持不变，这就使得后面的数据及数据分布更加稳定，减少了前面层与后面层的耦合，使得每一层不过多依赖前面的网络层，最终加快整个神经网络的训练。

批标准化还有附带的有正则化的效果：当使用小批量梯度下降时，对每个小批量进行批标准化时，会给这个小批量中最后求得的$z$带来一些干扰，产生类似与DropOut的正则化效果，但效果不是很显著。当这个小批量的数量越大时，正则化的效果越弱。

需要注意的是，**批标准化并不是一种正则化的手段**，正则化效果只是其顺带的小副作用。另外，在训练时用到了批标准化，则在测试时也必须用到批标准化。

训练时，输入的是小批量的训练样本，而测试时，测试样本是一个一个输入的。这里就又要用到指数加权平均，在训练过程中，求得每个小批量的均值和方差的数加权平均值，之后将最终的结果保存并应用到测试过程中。

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
3. [梯度下降法的三种形式-博客园](http://www.cnblogs.com/maybe2030/p/5089753.html)
4. [什么是批标准化-知乎专栏](https://zhuanlan.zhihu.com/p/24810318)
5. [Softmax回归-Ufldl](http://ufldl.stanford.edu/wiki/index.php/Softmax回归)
6. [神经网络权重初始化-知乎专栏](https://zhuanlan.zhihu.com/p/21560667?refer=intelligentunit)

#### 更新历史
* 2019.04.04 完成初稿
