1959年，IBM科学家Arthur Samuel开发了一个跳棋程序。通过这个程序，塞缪尔驳倒了普罗维登斯提出的机器无法超越人类，像人类一样写代码和学习的模式。他创造了“机器学习（machine learning）”，并将它定义为**“可以提供计算机能力而无需显式编程的研究领域”**。

1998年，卡内基梅隆大学的Tom MitChell给出了一种更为形式化的定义：**假设用P来估计计算机程序在某任务类T上的性能，若一个程序通过利用经验E在任务T上获得了性能改善，我们则称关于T和P，该程序对E进行了学习。**

通常，大部分机器学习问题都可以划分为**监督学习（Supervised Learning）**和**无监督学习（Unsupervised Learning）**两类：

* **监督学习**：给定的数据集中已经包含了正确的输出结果，希望机器能够根据这些数据集学习一个模型，使模型能够对任意的输入，对其对应的输出做出一个好的预测。监督学习具体又可以分为：
   * **回归（Regression）**：将输入的变量映射到某个连续函数。
   
   例如，根据一些房子面积与其价格的对应数据，训练一个模型来预测某面积之下的房价：
   ![房价预测](http://ww1.sinaimg.cn/large/82e16446ly1fyk945jcgtj20jp09labx.jpg)
   * **分类（Classification）**：将输入变量映射成离散的类别。
   
   例如，根据一些肿瘤大小与年龄的对应数据，训练一个模型来对良性、恶性肿瘤进行判断：
   ![肿瘤判断](https://i.loli.net/2018/12/26/5c2345682ee7c.png)

* **无监督学习**：给定的数据集中不包含任何输出结果，希望机器通过算法自行分析而得出结果。无监督学习具体可以分为：
   * **聚类（Clusterng）**：将数据集归结为几个簇
   
   例如，将各种新闻聚合成一个个新闻专题。

   * **非聚类（Non-clustering)**
   
   例如，将鸡尾酒会上的音乐声和人声分离。

***
#### 参考资料
1. [Andrew Ng-Machine Learning-Coursera](https://www.coursera.org/learn/machine-learning/)
2. [吴恩达-机器学习-网易云课堂](https://study.163.com/course/introduction/1004570029.htm)
3. [机器学习简史-CSDN](https://blog.csdn.net/qq_14845119/article/details/51317160)