目标检测是计算机视觉领域基本且重要的问题之一，其主要任务是赋予机器找出图像中包含的所有感兴趣的目标（物体），并确定它们大小和位置的能力。由于物体的外观、形状、姿态多种多样，外加成像过程很容易受到光照、遮挡等不确定外部因素的干扰，目标检测一直是计算机视觉领域最具挑战性的问题。

基于传统手工特征时期的目标检测算法，大都采用与检测目标相对应的人工设计特征，利用SVM等传统的机器学习算法，结合一些诸如滑动窗口这样的简单的策略，来实现对某些特定目标的检测。然而面对多种多样的目标时，手工特征的表达能力不足，滑动窗口策略会产生大量冗余，导致这些传统的目标检测的算法泛化能力比较弱的同时，时间复杂度又较高，鉴于2012年AlexNet在图像分类任务上的出色表现，各类目标检测算法的研究也纷纷将目光转移到CNN上。目标检测技术由此进入当前基于深度学习算法时期，开始以前所未有的速度发展。期间提出的各类算法，大致上可分为以**R-CNN（Region-CNN）**为代表的多步检测算法和以**YOLO（You Only Look Once）**为代表的单步检测算法。

### R-CNN系列

率先使用CNN，并在检测效果上远超传统算法的目标检测算法，是2013年加州大学伯克利分校的Girshick等人在论文[[Rich feature hierarchies for accurate object detection and semantic segmentation]](https://arxiv.org/pdf/1311.2524.pdf)提出的R-CNN算法。该算法是利用深度学习进行目标检测的开山之作，其中是**区域候选（Region Proposal）**思想虽然早在传统的手工特征时期就有人尝试过，但却由此衍生了SPPNet、Fast R-CNN等一大批基于此思想的目标检测算法。

最初的R-CNN采用AlexNet作为主干架构（backbone），整个系统被分成三个模块，如下图所示：

![R-CNN检测流程](https://i.loli.net/2019/06/14/5d03105ed73e414813.png)

其中，第一个是区域候选模块，该模块利用**选择性搜索（Selective Search）**算法，在原图中产生大约$2000$个候选区域作为CNN的输入数据集；第二个模块即为AlexNet，用它来提取各候选区域中包含的特征，各候选区域在输入前进行各向同性或异性缩放而变成固定尺寸（$227\times227$），经过CNN后则输出固定大小（$4096$维）的特征向量；第三个模块是一系列的线性支持向量机（Support Vector Machine）分类器，它们负责根据提取到的特征来筛选出包含目标的候选区域，并对各目标进行分类。

相比于传统的滑窗算法，虽然区域候选策略的时间复杂度要小很多，但由于从一张图像中筛选出来的各个候选区域都需要单独使用CNN进行特征提取，各候选区域之间又存在一定的重叠，而导致R-CNN算法中的存在大量的重复计算，运行速度极慢。另外，R-CNN中对候选区域进行特征提取时，各区域被强制放缩，有损检测准确度，整个检测过程被分成多个阶段，训练也十分繁琐耗时。

![SPP结构](https://i.loli.net/2019/06/14/5d0311c60146784126.png)

2014年MSAR的何恺明等人在论文[[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition Kaiming]](https://arxiv.org/pdf/1406.4729)中提出的SPP-Net中，引入了如上图所示的**空间金字塔池化结构（Spatial Pyramid Pooling，SPP）**来避免R-CNN中存在的重复计算以及必须强制放缩候选区域问题。在CNN中，卷积、池化层可以处理任意大小的输入，只有全连接层需要有固定大小的输入。为使任意大小的输入经过CNN的卷积层后得到的特征图都有相同的大小，SPP结构对最后一个卷积层输出的特征图进行三种步幅、窗口大小各不相同的金字塔形池化，分别得到三种固定大小的特征图。只要将这些特征图拆分重组为一个特征向量，后面的全连接层就会有固定大小的输入，由此实现用同一个CNN来处理多种尺寸的候选区域。

![Fast R-CNN](https://i.loli.net/2019/06/14/5d0313428e60021898.png)

在SPP结构的启发下，2015年MSAR的Ross B. Girshick又在论文[[Fast R-CNN]](https://arxiv.org/pdf/1504.08083)中提出了Fast R-CNN，将SPP结构进一步简化为效果更优的**ROI（Region of Interest）池化结构**。另外，还将R-CNN中的SVM分类器替换成了两个同级输出的全连接层，用它们分别完成分类以及Bounding Boxes回归任务。这些策略进一步提高了R-CNN的检测速度及精度。然而Fast R-CNN仍没有实现对图像的端到端处理，而需要先采用其他算法进行区域候选。在Fast R-CNN提出后不久，何恺明等人在论文[[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks]](https://arxiv.org/pdf/1506.01497)再次提出了Faster R-CNN算法，彻底去掉了R-CNN中需要独立进行区域候选阶段，而引入与CNN共用卷积层的**区域候选网络（Region Proposal Network， RPN）**来完成区域候选任务，大大减少了检测的计算成本，实现了第一个真正意义上的端到端的深度学习目标检测算法。

![Faster R-CNN](https://i.loli.net/2019/06/14/5d0313aa68dd856739.png)

Faster R-CNN的整体框架如上。将整张图像输入最底层的CNN后，提取的特征图被输入到RPN中，RPN中将使用三种不同尺寸、不同比例的Anchor Boxes在整个特征图上滑动，最后将生成约300个候选区域。后面的过程与Fast R-CNN基本一致：将得到候选区域输入ROI池化层，得到固定大小的特征图，最后由两个同级输出层输出检测结果。 

2017年，林宗毅等人在Faster R-CNN的基础上，又在论文[[Feature Pyramid Networks for Object Detection]](https://arxiv.org/pdf/1612.03144)中提出了一种多尺度的目标检测算法——**特征金字塔网络（Feature Pyramid Networks，FPN）**。考虑到低层特征的抽象层级较低，包含的语义信息较少，但各目标的位置更为精确，高层特征则与此相反，算法中将网络提取到的高层特征进行上采样（Up Sampled）后与多个低层特征进行融合，各个融合后的特征独立进行结果预测，这样能有效提高目标的检测精度。其间，也有Mask R-CNN等更为先进的多步检测算法先后被提出。

### YOLO系列
R-CNN为代表的目标检测算法均利用区域候选思想，将检测分成多个阶段，这使得这些模型在训练过程较为繁琐，运行效率也难以提高。2015年华盛顿大学的Joseph Redmon等人在论文[[You Only Look Once: Unified, Real-Time Object Detection]](https://arxiv.org/pdf/1506.02640)中提出的YOLO算法中，将目标检测直接作为回归（Regression）问题来解决，而使用一个一体化的CNN实现了端到端（End to End）检测，为目标检测问题提供了另外一种解决思路的同时，基于深度学习的目标检测算法也自此有了单步和多步之分。

在YOLO的v1版本中，一体化的CNN以GoogLeNet的变种作为主干架构，其中共包含24个卷积层和2个全连接层。该网络的检测流程如下图：

![YOLOv1](https://i.loli.net/2019/06/14/5d0315e789faa47989.png)

检测过程中，将输入的图像平均分成$S\times S$个网格，某个目标的中心位置所在的网格负责检测该目标。经过整个CNN后，每个网格需要预测B个包围盒的位置以它们对应的置信评分（Confidence Scores）。评分则反映了预测的包围盒位置以及包围盒中包含某个目标的可信度，包围盒的位置用$x$、$y$、$w$、$h$这四个值来表示，其中$(x,y)$表示包围盒的中心点相对于其对应网格的坐标，w、h则表示包围盒相对于整个图像的宽度和高度。如果某个网格中存在某个目标，则该网格还需要预测C个概率值，它们表示感兴趣的C种目标存在的概率。因此，整个模型的输出结果大小将为$S\times S \times(B\cdot5+C)$。最后根据模型的输出的结果进行**非极大值抑制（Non-Maximum Suppression）**，就能得到最后的检测结果。

YOLOv1的检测过程都在CNN上的各个层上完成，速度非常快，但在定位精确度上却不及Faster R-CNN这类采用区域候选策略的多步检测算法，对于图像上小目标的检测效果也不够好。针对这些问题，2016年北卡大学教堂山分校的刘伟等人结合Faster R-CNN中的锚盒机制及YOLO中的回归思想，在论文[[SSD: Single Shot MultiBox Detector]](https://arxiv.org/pdf/1512.02325)中提出了SSD算法。SSD利用了CNN主干架构的不同卷积层上提取到的多种特征图来预测多个比例的包围盒，使单步目标检测算法保持了高速度的同时兼具了较高的检测准确度。下面是SSD检测模型与YOLOv1的对比图：

![SSD与YOLOv1对比](https://i.loli.net/2019/06/14/5d03173050a9b17716.png)

在2017年的CVPR上，Joseph Redmon等人采用一系列方法对YOLOv1进行了全方面的改进，而在论文[[YOLO9000: Better, faster, stronger]](https://arxiv.org/pdf/1612.08242)中提出了YOLO算法的v2版本。YOLOv2以加入残差结构的Darknet-19作为主干架构，在网络种使用了批标准化策略，并采用了Network in Network中提出的使用$1\times1$卷积核来完成特征图升维、全局平均池化来完成特征压缩的思想，对输入图像进行特征提取。该算法也和SSD一样，引入了Faster R-CNN中使用的锚盒机制，但其利用了K-Means算法从在训练集中进行聚类计算，而得到检测效果更优的锚盒尺寸。为使加入锚盒机制后的YOLO算法更加稳定，YOLOv2中不再直接预测包围盒的真实坐标，而是借鉴了Faster R-CNN中处理方式，使用强约束方法对包围盒的位置进行编码。

到2018年，YOLO算法已经更新到了v3版本，检测效果进一步得到提高。其间，SSD算法衍生出了多种版本，Retina-Net、RefineDet等结合多种思想的单步检测算法也先后被提出。无论是多步还是单步目标检测算法，都朝着更快更强的目标，仍在不断发展之中。

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
* 2019.06.14 完成初稿
