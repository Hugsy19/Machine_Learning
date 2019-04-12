后面将反复用到的一些符号，在这里将它们表示的含义约定如下：
* $m$：训练样本（training example）的数量
* $n$：特征（feature）的数量
* $x^{(i)}$：第$i$个输入变量（即训练样本）
* $y^{(i)}$：第$i$个训练样本对应的标签（label）
* $x^{(i)}_j$：第$i$个训练样本的第$j$种特征的值
* $\hat{y}^{(i)}$：第$i$个输出变量，也就是模型对第$i$个训练样本的的预测结果
* $(x^{(i)},y^{(i)})$：第$i$个训练样本，$i = 1，\cdots , m$时为一个训练集（training set）
* $X$：输入空间（input space，也称“样本空间”）
* $Y$：输出空间（output space，也称“标记空间”）
* $w_j$：第$j$种特征的权重（weight）
* $b$：偏差（bias)
* $a^{[i]}_j$:第$i$层神经网络中第$j$个神经元产生的激活（activation）
  
tip：
前面的一些章节中引用的图片上，采用了一些另外的符号：
* $h_{\theta}(x)$：同$\hat{y}$
* $\theta_1, \cdots, \theta_n$：同$w_1, \cdots, w_n$
* $\theta_0$：同$b$