### 数据划分

想要建立一个神经网络模型，首先，就是要设置好整个数据集中的**训练集（Training Sets）**、**开发集（Development Sets）**和**测试集（Test Sets）**。

采用训练集进行训练时，通过改变几个超参数的值，将会得到几种不同的模型。开发集又称为**交叉验证集（Hold-out Cross Validation Sets）**，它用来找出建立的几个不同模型中表现最好的模型。之后将这个模型运用到测试集上进行测试，对算法的好坏程度做无偏估计。通常，会直接省去最后测试集，将开发集当做“测试集”。一个需要注意的问题是，需要保证训练集和测试集的来源一致，否则会导致最后的结果存在较大的偏差。