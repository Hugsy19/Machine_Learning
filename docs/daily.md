### Faster R-CNN Paper
前面的R-CNN主要R作为分类器，并不能预测目标边界，最后的检测准确度也却绝育区域候选方法，而选择性搜索等区域候选方法通常依赖廉价的特征和简练的推断方案，使得区域候选成为Fast R-CNN在测试时的计算瓶颈。Faster R-CNN中在Fast R-CNN的基础上引入RPNs（Region Proposal Networks），它与进行目标检测的CNN共用卷积层而进行区域候选，大大减少计算成本。

![RPN](https://i.loli.net/2019/05/07/5cd143dbc03e7.png)

在Fast R-CNN中卷积层提取到的卷积特征之上，可通过附加一些卷积层来构建RPN，它们同时在规则的网格的每个位置上回归区域边界和目标分数。RPN是一种全卷积网络，可针对特定的区域候选任务进行端到端训练。RPN中引入anchor boxes作为多种尺度和长宽比的参考，它可以有效候选出多种尺度和长宽比的区域，而避免像之前的目标检测方法那样枚举多种尺度和长宽比的卷积核或图片。

![Faster R-CNN](https://i.loli.net/2019/05/07/5cd14e5c6f6e9.png)

Faster R-CNN由两个模块组成，第一个模块是用来进行区域候选的深度全连接CNN，第二个则是使用候选区域的Fast R-CNN检测器，两个模块共用卷积层。RPN中，以一张任意大小的图片作为输入，最后输出一系列的带有目标评分的矩形候选区域。

![RPN](https://i.loli.net/2019/05/07/5cd15c0771edd.png)

为生成候选区域，在最后一个卷积层输出的feature map上滑动一个小的网络，该网络以feature map的$3\times3$空间窗口作为输入，每次滑动窗口时将它们映射为低维特征（ZFnet为$256$维，VGG为$512$维，后面用ReLU激活），得到的这些特征被送入两个子全连接层——一个box回归层（reg）和一个box分类层（cls），这种结构可简单地通过一个$3\times3$的卷积层和两个$1\times1$的卷积层来实现。

在每个滑动窗口经过的的位置上，同时预测$k$个候选区域（即anchor boxes），由此reg层将输出$k$个boxes的$4k$个位置信息，cls层则输出$2k$个候选区域是目标与否的评分。每个anchor box都以窗口的中心为中心，但有各自的尺度和长宽比。这里用了3个尺度和3个长宽比，在每个滑动位置产生$k=9$个anchor box。Anchors还具有平移不变性。

训练RPNs时，给每个anchor进行二分类，将与实际box有最高的IoU或IoU超过$0.7$的anchor作为正类，单个真实box可以为多个anchor分配正标签，如果对所有真实的box，anchor的IoU都低于$0.3$，则作为负类，处于中间的anchor都是无助于训练的。

由此，有多任务损失函数：$$\mathcal{L}(\lbrace p_i \rbrace, \lbrace t_i \rbrace) = \frac{1}{N_{cls}}\sum_i \mathcal{L}_{cls}(p_i, p^{\*}_i) + \lambda\frac{1}{N_{reg}}\sum_i p^{\*}_i \mathcal{L}_{reg}(t_i, t^{\*}_i)$$

其中，$i$是小批量中anchor box的索引，$p_i$是预测出来的第$i$个anchor为某个物体的概率，$p^{\*}_i$是对应的标签，对应的$t_i$则是预测出来的$4$个box位置信息，$t^{\*}_i$是对应的正类anchor标签，它们用下式进行参数化：$$t_{\textrm{x}} =  (x - x_{\textrm{a}})/w_{\textrm{a}},\quad t_{\textrm{y}} = (y - y_{\textrm{a}})/h_{\textrm{a}},\\ t_{\textrm{w}} = \log(w / w_{\textrm{a}}), \quad t_{\textrm{h}} = \log(h / h_{\textrm{a}}),\\ t^{\*}_{\textrm{x}} =  (x^{\*} - x_{\textrm{a}})/w_{\textrm{a}},\quad t^{\*}_{\textrm{y}} = (y^{\*} - y_{\textrm{a}})/h_{\textrm{a}},\\ t^{\*}_{\textrm{w}} = \log(w^{\*} / w_{\textrm{a}}),\quad t^{\*}_{\textrm{h}} = \log(h^{\*} / h_{\textrm{a}})$$

$L_{cls}$是log损失函数，$L_{reg}$则是Fast R-CNN中用过的smooth L1损失函数；$N_{cls}$、$N_{reg}$分别为批量大小和anchor数量，用以标准化，$\lambda$则用于均衡两个任务的损失。

