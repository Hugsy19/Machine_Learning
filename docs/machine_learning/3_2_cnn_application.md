### 目标检测

图像识别中，目标检测的任务，是对输入图像样本准确进行分类的基础上，检测其中包含的某些目标，并对它们准确定位并标识。

#### 目标定位

图像分类问题一般都采用Softmax回归来解决，最后输出的结果是一个多维列向量，且向量的维数与假定的分类类别数一致。在此基础上希望检测其中的包含的各种目标并对它们进行定位，这里对这个监督学习任务的标签表示形式作出定义。

![定位表示](https://ws1.sinaimg.cn/large/82e16446ly1fnxiefqv4uj214e0j7jxa.jpg)

如上图所示，分类器将输入的图片分成行人、汽车、摩托车、背景四类，最后输出的就会是一个四维列向量，四个值分别代表四种类别存在的概率。

加入目标检测任务后，用$p_c$表示存在目标的概率；以图片的左上角为顶点建立平面坐标系，用$b_x$、$b_y$组成图像中某个目标的中点位置的二维坐标，它们的值都进行了归一化，相当于把图片右下角坐标设为(1,1)；$b_h$、$b_w$表示图中用以标识目标的红色**包围盒（Bounding Box）**的长度和宽度；$c_n$表示存在第n个种类的概率。不存在目标时，$p_c = 0$，此时剩下的其他值都是无效的。整个标签的表示形式如下：
$$y = \begin{bmatrix} p_c \\\ b_x \\\ b_y \\\ b_h \\\ b_w \\\ c_1 \\\ c_2 \\\ ... \end{bmatrix}$$

在计算损失时将有：
$$\mathcal{L}(\hat y,y) = \begin{cases} (\hat{y_1}-y_1)^2 + (\hat{y_2}-y_2)^2 + ... +(\hat{y_n}-y_n)^2 ,  & \text{$(y_1(p_c)=1)$} \\\ (\hat{y_1}-y_1)^2, & \text{$(y_1(p_c)=0)$} \end{cases}$$

其中，对不同的值，可采用不同的损失函数。

另外，如需检测某幅图像中的某些特征点，比如一张人脸图像中五官的各个位置，可以像标识目标的中点位置那样，在标签中，将这些特征点以多个二维坐标的形式表示。

#### 滑窗检测

用以实现目标检测的算法之一叫做**滑窗检测（Sliding Windows Detection）**。

滑动窗口检测算法的实现，首先需要用卷积网络训练出一个能够准确识别目标的分类器，且这个分类器要尽可能采用仅包含该目标的训练样本进行训练。随后选定一个特定大小（小于输入图像大小）的窗口，在要检测目标的样本图像中以固定的步幅滑动这个窗口，从上到下，从左到右依次遍历整张图像，同时把窗口经过的区域依次输入之前训练好的分类器中，以此实现对目标及其位置的检测。

![滑动窗口](https://ws1.sinaimg.cn/large/82e16446ly1fo10q4530uj213d0gntek.jpg)

选用窗口的大小、进行遍历的步长决定了每次截取的图片大小及截取的次数，如上图所示，这也关系到了检测性能的好坏以及计算成本的高低。然而这种方法实现滑窗的计算成本往往很大，效率也较低。

上述的滑窗检测的过程类似于前面进行卷积运算过程，而滑窗检测算法其实就可以用一个完整的卷积网络较为高效地实现，此时需要将卷积网络中原来的全连接层转化为卷积层。

![卷积网络](https://ws1.sinaimg.cn/large/82e16446gy1fo24vjkou5j21410hd0tv.jpg)

上图所示为一个卷积神经网络，经过卷积、池化后，全连接过程可以看作是将池化后得到的大小为5×5×16的结果与400个大小也为5×5×16的卷积核分别进行卷积，输出的结果大小为1×1×400，进一步全连接再采用Softmax后，最后输出的结果大小为1×1×4。由此，全连接过程本质上还是一个卷积过程。

![滑动检测](https://ws1.sinaimg.cn/large/82e16446gy1fo275wzl7dj21340dg0zd.jpg)

在卷积网络中实现滑窗检验的过程如上图。向之前的示例中输入一个更大的图像，而保持各层卷积核的大小不变，最后的输出结果大小为2×2×4，也就相当于用一个大小为14×14的窗口，以2个单位的步长，在输入的图像中进行滑窗检测后得到的结果，图中对此用不同的颜色进行了标识。

其实，在滑动窗口的过程中可以发现，卷积过程的很多很多计算都是重复的。用卷积网络实现滑动窗口检验，减少了重复的计算，从而提高了效率。这样一个方法，是Sermanet等人2014年在论文[[OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks]](https://arxiv.org/pdf/1312.6229.pdf)中提出来的。

#### YOLO算法

采用滑窗检测进行目标检测，难以选取到一个可以完美匹配目标位置的，大小合适的窗口。

**YOLO（You Only Look Once）算法**是Redmon等人2015年在论文[[You Only Look Once: Unified, Real-Time Object Detection]](https://arxiv.org/pdf/1506.02640.pdf)中提出的另一种用于目标检测的算法。

YOLO算法中，将输入的图像划分为S×S个网格（Grid Cell)，对这S×S个网格分别指定一个标签，标签的形式如前面所述：
* $p_c$标识该网格中的目标存在与否。为“1”则表示存在；“0”则表示不存在，且标签中其他值都无效。
* $b_x$、$b_y$表示包围盒的中心坐标值，它们相对于该网格进行了归一化，也就是它们的取值范围在0到1之间；
* $b_h$、$b_w$表示包围盒的长度和宽度；
* $c_n$表示第n个假定类别存在的概率。

某个目标的中点落入某个网格，该网格就负责检测该对象。

![YOLO](https://ws1.sinaimg.cn/large/82e16446gy1fo3c0ss4nnj218n0l4h31.jpg)

如上面的示例中，如果将输入的图片划分为3×3个网格、需要检测的类别有3类，则每一网格部分图片的标签会是一个8维的列矩阵，最后输出结果的大小就是3×3×8。要得到这个结果，就要训练一个输入大小为100×100×3，输出大小为3×3×8的卷积神经网络。

预测出的目标位置的准确程度用**IOU（Intersection Over Union）**来衡量，它表示预测出的包围盒（Bounding Box）与实际边界（Ground Truth）的重叠度，也就是两个不同包围盒的交并比。如下图中所示，IOU就等于两个包围盒的交集面积（黄色部分）占两个包围盒的并集面积（绿色部分）的比率。一般可以约定一个阈值，以此判断预测的包围盒的准确与否。

![IOU值](https://ws1.sinaimg.cn/large/82e16446gy1fo3diqi5z2j211d0fftfb.jpg)

使用YOLO算法进行目标检测，因为是多个网格对某些目标同时进行检测，很可能会出现同一目标被多个网格检测到，并生成多个不同的包围盒的情况，这时需要通过**非极大值抑制（Non-max Suppression）**来筛选出其中最佳的那个。

对于每一个网格，将通过一个**置信度评分（Confidence Scores）**来评判该网格检测某个目标的准确性，这个评分值为$p_c$值与各$c_n$值的乘积中的最大值，也就是说每个包围盒将分别对应一个评分值。进行非极大值抑制的步骤如下：
1. 选取拥有最大评分值的包围盒；
2. 分别计算该包围盒与所有其他包围盒的IOU值，将所有IOU超过预设阈值的包围盒丢弃；
3. 重复以上过程直到不存在比当前评分值更小的包围盒。

上述算法只适用于单目标检测，也就是每个网格只能检测一个对象。要将该算法运用在多目标检测上，需要用到**Anchor Boxes**。在原单目标检测所用的标签中加入其他目标的标签，每个目标的标签表示形式都如上所述，一组标签即标明一个Anchor Box，则一个网格的标签中将包含多个Anchor Box，相当于存在多个用以标识不同目标的包围盒。

![Anchor Box](https://ws1.sinaimg.cn/large/82e16446gy1fo6t2zsxcqj213p0luqdv.jpg)

如上面的例子中，还是将输入的图片划分为3×3个网格且检测的类别为3类。希望同时检测人和汽车，则每个网格的标签中要含有两个Anchor Box，每一网格部分图片的标签会是一个16维的列矩阵，最后输出结果的大小就是3×3×16。

单目标检测中，图像中的目标被分配给了包含该目标的中点的那个网格；引入Anchor Box进行多目标检测，图像中的目标则被分配到了包含该目标的中点的那个网格以及具有最高IOU值的网格的Anchor Box。

#### R-CNN

**R-CNN（Region CNN)**是Girshick等人2013年在论文[[Rich feature hierarchies for accurate object detection and semantic segmentation]](https://arxiv.org/pdf/1311.2524.pdf)中提出的一种目标检测算法，其中提出的**候选区域（Region Proposal）**概念在计算机视觉领域有很大的影响力，它可以说是利用深度学习进行目标检测的开山之作。

R-CNN意为带区域的卷积网络，类似之前所述的滑窗检测算法，先用卷积网络训练一个能够准确识别目标的分类器，但这个算法试图选出一些区域为候选区域，只在这些区域也就是只在少数的窗口上运行分类器。候选区域的选取采用的是一种称为图像分割的算法。

后续有一系列的研究工作，试图改进这个算法，而出现了Fast R-CNN、Faster R-CNN算法，不过（Andrew Ng认为）这些算法在运行速度方面还是不如YOLO算法。

### 人脸识别

**人脸验证（Face Verification）**和**人脸识别（Face Recognition）**是两个在人脸识别相关文献中被普遍提及的术语，前者一般指一个一对一问题，只需要验证输入的人脸图像等信息是否与某个已有的身份信息对应，而后者需要验证输入的人脸图像是否与多个已有的信息中的某一个匹配，是一个更为复杂的一对多问题。

在真实的应用场景中，人脸识别是一个One—Shot学习的过程，要求人脸识别系统只采集某人的一个面部样本，就能对这个人做出快速准确的识别，也就是说只用一个训练样本训练而获得准确的预测结果，这是对人脸识别的研究上所面临的挑战。

这里，One-Shot学习过程通过学习一个Similarity函数来实现。Similarity函数的表达式为：
$$Similarity = d(img1, img2)$$

它定义了输入的两幅图片之间的差异度。设置一个超参数$\tau$，当$d(img1, img2) \le \tau$，则两幅图片为同一人，否则为不同。

#### Siamese网络

![Siamese网络](https://ws1.sinaimg.cn/large/82e16446gy1focbtfiaqqj211g0hk0vf.jpg)

上图的示例中，将图片$x^{(1)}$、$x^{(2)}$分别输入两个相同的卷积网络中，经过全连接后不再进行Softmax，得到它们的特征向量$f(x^{(1)})$、$f(x^{(2)})$。此时Similarity函数就被定义为这两个特征向量之差的2范数：
$$d(x^{(1)}, x^{(2)}) = \mid \mid f(x^{(1)}) - f(x^{(2)}) \mid \mid^2_2$$

这种对两个不同输入运行相同的卷积网络，然后对它们的结果进行比较的神经网络，叫做**Siamese网络**。

![二分类](https://ws1.sinaimg.cn/large/82e16446gy1focjds277uj20yc0anmyz.jpg)

利用一对相同的Siamese网络，可以将人脸验证看作二分类问题。如上图中，输入的两张图片$x^{(i)}$、$x^{(j)}$，经过卷积网络后分别得到m维的特征向量$f(x^{(i)})$、$f(x^{(j)})$，将它们输入一个逻辑回归单元，最后输出的预测结果中用1和0表示相同或不同的人。

其中对最后的输出结果$\hat{y}$，如果使用的逻辑回归单元是sigmoid函数，表达式就会是：
$$\hat{y} = \sigma(\sum_{k=1}^m w_i \mid f(x^{(i)})_k - f(x^{(j)})_k \mid + b)$$

以上所述内容，都来自Taigman等人2014年发表的论文[[DeepFace closing the gap to human level performance]](http://www.cs.wayne.edu/~mdong/taigman_cvpr14.pdf)中提出的DeepFace。

#### Triplet损失

利用神经网络实现人脸识别，想要训练出合适的参数以获得优质的人脸图像编码，需要在每次正向传播后计算Triplet损失。

Triplet损失函数的定义基于三张图片--两张同一人的不同人脸图像和一张其他人的人脸图像，它们的特征向量分别用符号A（Anchor）、P（Positive）、N（Negative）表示，如下图。
![Anchor、Positive、Negative](https://ws1.sinaimg.cn/large/82e16446gy1foch0gg5vwj20kc0a6wic.jpg)
对于这三张图片，想要实现：
$$\mid \mid f(A) - f(P) \mid \mid_2^2 + \alpha < \mid \mid f(A) - f(N) \mid \mid_2^2$$

其中的$\alpha$为间隔参数，用以防止产生无用的结果。则Triplet损失函数被定义为：
$$\mathcal{L}(A, P, N) = max(\mid \mid f(A) - f(P) \mid \mid_2^2 - \mid \mid f(A)- f(N) \mid \mid_2^2 + \alpha, 0)$$

式中的主要部分为A与P之差的范数减去A与N之差的范数后，再加上间隔参数$\alpha$，因为它的值需要小于等于0，所以直接取它和0的max。

这样，训练这个神经网络就需要有大量经过特定组合的包含Anchor、Postive、Negative的图片组。且使用m个训练样本，代价函数将是：
$$\mathcal{J} = \sum^{m}_{i=1} [\mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2 - \mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2 + \alpha ] $$

Triplet损失的相关内容来自Schroff等人2015年在论文[[FaceNet: A unified embedding for face recognition and clustering]](https://arxiv.org/pdf/1503.03832.pdf)中提出的FaceNet，更细节可以参考论文内容。

### 神经风格转换

**神经风格迁移（Neural Style Tranfer）**是将参考风格图像的风格转换到另一个输入图像中，如下图所示。

![神经风格迁移](https://ws1.sinaimg.cn/large/82e16446gy1focllhe7ecj217a0ji1kx.jpg)

其中待转换的图片标识为C（Content），某种风格的图片为S（Style），转换后的图片为G（Generated）。

#### 理解CNN

要理解利用卷积网络实现神经风格转换的原理，首先要理解在输入图像数据后，一个深度卷积网络从中都学到了些什么。

2013年Zeiler和Fergus在论文[[Visualizing and understanding convolutional networks]](https://arxiv.org/pdf/1311.2901.pdf)中提出了一种将卷积神经网络的隐藏层特征进行可视化的方法。

![AlexNet](https://ws1.sinaimg.cn/large/82e16446gy1focon19xrrj20i30g10wh.jpg)

上图展示是一个AlexNet中的卷积、池化以及最后的归一化过程，以及实现隐藏层可视化的反卷积网络中的Unpooling、矫正以及反卷积过程。论文中将ImageNet 2012中的130万张图片作为训练集，训练结束后提取到的各个隐藏层特征如下图：

![特征图](https://ws1.sinaimg.cn/large/82e16446gy1focp4uk3v0j20f30kqwhi.jpg)

从中可以看出，浅层的隐藏单元通常学习到的是边缘、颜色等简单特征，越往深层，隐藏单元学习到的特征也越来越复杂。

#### 实现过程

实现神经风格转换，需要定义一个关于生成的图像$G$的代价函数$J(G)$，以此评判生成图像的好坏的同时，用梯度下降法最小化这个代价函数，而生成最终的图像。

$J(G)$由两个部分组成：
$$J(G) = \alpha J\_{content}(C,G) + \beta J\_{style}(S,G)$$

其中**内容代价函数**$J\_{content}(C,G)$度量待转换的C和生成的G的相似度，风格代价函数$J_{style}(S,G)$则度量某风格的S和生成的G的相似度，用超参数$\alpha$和$\beta$来调整它们的权重。

将C、G分别输入一个预先训练好的卷积神经网络中，选择该网络中的某个中间层$l$，$a^{(C)[l]}$、$a^{(G)[l]}$表示C、G在该层的激活,则内容代价函数$J\_{content}(C,G)$的表达式为：
$$J\_{content}(C,G) =  \frac{1}{2} \mid \mid(a^{(C)[l]} - a^{(G)[l]})\mid \mid^2$$

定义**风格代价函数**$J_{style}(S,G)$前，首先提取出S的“风格”。通过之前的理解CNN内容，将S也输入那个预先训练好的卷积神经网络中，就可以将其所谓的“风格”定义为神经网络中某一层或者几个层中，各个通道的激活项之间的相关系数。如下图所示为网络中的某一层，假设其中前两个红、黄色通道分别检测出了下面对于颜色圈出的特征，则这两个通道的相关系数，就反映出了该图像所具有的“风格”。

![定义“风格”](https://ws1.sinaimg.cn/large/82e16446gy1fodrmwzl1cj20ej0ibaeu.jpg)

选定网络中的大小为$i \times j \times k$的第$l$层，$a^{[l]}\_{ijk}$表示k个通道的激活，则相关系数以一个Gram矩阵的形式表示为：
$$\mathcal{G}^{(S)[l]}\_{kk'} = \sum\_{i=1}^{n\_H^{[l]}} \sum\_{j=1}^{n\_W^{[l]}} a^{(S)[l]}\_{ijk} a^{(S)[l]}\_{ijk'}$$

对G，也表示出其在网络层中的相关系数：
$$\mathcal{G}^{(G)[l]}\_{kk'} = \sum\_{i=1}^{n\_H^{[l]}} \sum\_{j=1}^{n\_W^{[l]}} a^{(G)[l]}\_{ijk} a^{(G)[l]}\_{ijk'}$$

这样，第$l$层的风格代价函数表达式将是：
$$J\_{style}^{[l]}(S,G) = \frac{1}{(2 n_C n_H n_W)^2} \sum\_k \sum\_{k'}(\mathcal{G}^{(S)}\_{kk'} - \mathcal{G}^{(G)}\_{kk'})^2$$

一般将这个风格代价函数运用到每一层，可以获得更好的效果，此时的表达式为：
$$J\_{style}(S,G) = \sum\_{l} \lambda^{[l]} J^{[l]}\_{style}(S,G)$$

其中$\lambda$是一个用来设置每一层所占权重的超参数。

这样一种神经风格转换的实现方法，来自2015年Gatys等人发表的论文[[A Neural Algorithm of Artistic Style]](https://arxiv.org/abs/1508.06576)。

***
#### 相关程序

#### 参考资料
1. [吴恩达-卷积神经网络-网易云课堂](http://mooc.study.163.com/course/2001281004#/info)
2. [Andrew Ng-Convolutional Neural Networks-Coursera](https://www.coursera.org/learn/convolutional-neural-networks/)
3.  [YOLO——基于回归的目标检测算法-csdn](http://blog.csdn.net/btbujhj/article/details/75020217)
4.  [非极大值抑制-csdn](http://blog.csdn.net/u011534057/article/details/51235718)
5.  [RCNN算法详解-csdn](http://blog.csdn.net/shenxiaolu1984/article/details/51066975)
6.  [“看懂”卷积神经网-csdn](http://blog.csdn.net/xjz18298268521/article/details/52381830)

#### 更新历史
* 2019.04.20 完成初稿
