---
html:
   toc: true
---

# 参考
1. [本书代码网址](http://www.ituring.com.cn/book/1921) 
2. [latex公式对应](https://blog.csdn.net/yga_airspace/article/details/82560040)
3. [latex公式对应2](https://blog.csdn.net/weixin_43894041/article/details/104193938)
4. [latex公式图片转换](https://web.baimiaoapp.com/image-to-latex) 
5. [插入gif](https://www.cnblogs.com/AhuntSun-blog/p/12028808.html)
# 第一章 python语言和库的基本功能
## 语言介绍
1. 主要采用的语言：python
2. 采用的外部库 numpy 和 matplotlib
   1. 前者用于数值计算
   2. 后者用于画图和可视化
## 基本语法
1. *表示乘法，/表示除法，**表示乘方
2. 整数除以整数的结果是小数（浮点数）
3. 查看数据类型
   1. 函数type
   2. 语法type(要查看的数据)
4. 数据类型的定义：动态类型语言，不需要主动去定义数据类型
5. 列表
   1. 类似于数组
   2. 定义方法：列表名=[数据，之间用逗号隔开]
   3. 打印
      1. 打印方式：print(列表名)
      2. 打印格式：完整的列表（有方括号）
   4. 访问单一数据的方式：和数组一样，列表名[索引值]
   5. 访问子表的方式：
      1. 列表名[索引起始值：索引结束值] **特别注意：这边不包括子表最后一个元素**，
      2. 如果不给出索引的结束值，那么就默认访问到最后一个元素
      3. 如果在索引结束值给出的负数-k，那么就访问到最后一个元素前第k个的位置
   6. 访问列表的长度：len(列表名)
6. 字典
   1. 含义，就对应者字典的含义，可以查询和存储条目
   2. 定义字典（生成字典）：字典名 = {'条目名'：数据} **特别注意：条目要加' ',中间用 ： 隔开**
   3. 添加字典元素：字典名['索引条目'] = 数据
   4. 访问方式
      1. 访问其中某一个元素：字典名['索引条目'] **特别注意；这边是方括号，和定义的时候不一样** 
      2. 访问整个字典 print（字典名）
   5. 代码样例：字典名是me,其中有两条条目分别是weight和height
      ~~~python
      >>> me = {'height':180} # 生成字典
      >>> me['height'] # 访问元素
      180
      >>> me['weight'] = 70 # 添加新元素
      >>> print(me)
      {'height': 180, 'weight': 70}
      ~~~
7. 布尔值
   1. 取值：true 和 false 
   2. 运算 and、or、not 对应逻辑运算中的与或非
8. if 语句
   1. **注意执行代码部分的缩进**
   2. 判断部分**没有()**

9. for 语句
   1.  可以用来遍历列表元素：for i in []: **注意这里有一个冒号**
   2.  代码样例：
        ~~~python
        >>> for i in [1, 2, 3]:
        ... print(i)
        ...
        1
        2
        3
        ~~~
10. 函数
    1.函数的定义：def+函数名
    2.函数可以附加参数，同样的这边的参数同样**不需要给出类型**(定义了有参数的函数不给参数是否在python中可以调用)
11. 文件的创建和运行：见hello.py
12. 类(**这个部分需要进一步了解**)
    1.  含义：由编程者创建的特殊的数据类型
    2.  内容:初始化函数+成员函数
    3.  特点（和其他语言的不同）：self
        1. 感觉就是一个模板的实体，在类的定义里面使用，定义这个模板各个属性
    4. 代码样例：见class_show.py

## NumPy
1. 主要功能：数组和矩阵的运算
2. 调用方式：
   ~~~
   >>> import numpy as np
   ~~~
## 数组创建和运行
1. 一维数组
   1. 创建：可以接收列表类型进行数组的创建具体见create_arr.py
   **注意：numpy.ndarray和 list不是一个数据类型**
   待解决：numpy.ndarray 每个数据后面的点是什么
   2. 数组的基本运算
      1. 类型：加减乘除
      2. 方法：对应位置进行运算
2. 多维数组
   1. 创建方法，和一维数组的创建方式类似，两个列表之间用逗号隔开
   2. 打印数据：直接print
   3. 查看数组类型：（几行几列）数组名.shape **没有括号**
   4. 查看矩阵元素类型：dtype  **没有括号**
   5. 代码：见create_narr.py
   6. 2. 数组的基本运算
      1. 类型：加减乘除
      2. 方法：对应位置进行运算(注意这里的乘法运算依旧是对应位置上进行乘法运算，不是矩阵的乘法运算)
3. 多维数值转化为一维数组 flatten() **这边又是有括号的，我是真的麻了**
4. 广播；
   1. 作用：使得不同形状的矩阵也可以进行运算
   2. ==广播的规则？？？==
5. 查看数组中的元素
   1. 查看行和列都确定位置的元素
   2. 查看某一行的元素
   3. 按照索引查看（具体见find_ele.py,这边最后==大于查看原理？==）
6. ==待做：numpy中的函数==
   

## Matplotlib
1. 主要功能：绘制图形和数据可视化
2. 图像的绘制 ：代码见 first_curve.py
   1. 合成相关性 plt.plot(x,y,标签（可省略），线类型（可省略）)
   2. 展示图像 plt.show() 
3. 添加标题和坐标轴名称: 代码见 first_curve_up.py
   1. 曲线的标签
      1. 需要在合成相关性的时候就指明标签是什么
      2. **调用函数plot.legend()**
   2. 坐标轴的标签
      1. plot.xlabel()
      2. plot.ylabel()
4. 显示图像


---
# 第二章 感知机
1. 作用：信号的传递：多个输入，一个输出
2. 概念名词
   1. 输入信号
   2. 输出信号
   3. 权重：这条路上型号的重要程度
   4. 阈值：神经元激活的界限
3. 类比，可以将权重类比成电流中的电阻，“电阻”越小，权重越大。
4. 对感知机的理解：对于感知机的节点来讲，有两个输入，这个如果映射到二维坐标上，表示的一维线性空间下某一条直线下方的范围。
5. 单层感知机可以实现的逻辑功能：与门，与非门，或门
6. 单层感知机不能实现的逻辑功能：异或门
   1. 原因：单纯的一条直线无法分割异或门的范围
7. 感知机的权重等值都是由人设定的
8. 多层感知机在理论上可以表示计算机（因为所有的逻辑功能都可以实现



---
# 第三章 神经网络

## 神经网络的结构
1. 含义在神经网中的输入、中间的信号传递过程、输出层
2. 这对应这分别是第零层，第一层，第二层

## 激活函数
- 每个神经元节点的状态都都可以由一个激活函数进行表示
- 激活函数必须采用非线性函数，如果是线性的话，多层的神经网络将没有意义
- 本书重要介绍的激活函数有两个:阶跃函数、sigmoid函数
- 关于线性的激活函数是没有意义的解释
因为这样会让隐藏层失效，比如一个三层的隐藏层，激活函数是$cx$,最后的结果是$c^{3}x$,然而这和$c_{1}x$没有区别


### 阶跃函数
1. 阶跃函数的代码实现：
~~~ python
import numpy as np
import matplotlib.pylab as plt
def step_function(x):
   return np.array(x > 0, dtype=np.int) #该函数的返回值是一个np数组，并且将其传化为了0，1的形式
x = np.arange(-5.0, 5.0, 0.1)           #表示从-5到5，步长为0.1生成数据
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)                     # 指定y轴的范围
plt.show()
~~~
2. 函数图像
![图片](/deep%20learning/1.png)

### sigmoid 函数
- 由于np数组计算时广播的功能，标量会和数组中每一个数字进行计算
1. 公式表示
$h(x)=  \frac {1}{1+exp(-x)} $
2.代码实现
~~~python
def sigmoid(x):
   return 1 / (1 + np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.show()
~~~
2. 图像
![图片](/deep%20learning/2.png)


### 和阶跃函数的比较
1. 相同之处
   1. 当输入信号为重要信息时，输出较大的值；当输入信号为不重要的信息时，输出较小的值
   2. 输出信号的值都在0到1之间
   3. 两周都是非线性函数，因此可以作为激活函数
2. 不同之处
   1. sigmoid 函数是连续的会输出中间值

### ReLU函数
1. 公式表示
 $
h(x)=  \begin{cases}\\\end{cases} $ 
(x>0)
 $ \begin{cases}0(x\leqslant0)\\\end{cases} $ 
2. ReLU函数在输入大于0时，直接输出该值；在输入小于等于0时，输出0

## 多维数组的运算
1. 矩阵运算的知识，不赘述
2. 实现函数：
   1. 获取数组的维度: np.dim()
   2. 获取数组的形状: np.shape()
   3. 矩阵乘法（点积）: np.dot(矩阵1,矩阵2)
   4. 求和：np.sum()
   
## 神经网络信号传递
- 符号含义：第一个下标表示指向的位置，第二个下标表示数据来源的位置，上标表示指向层的层数
1. 图解：
2. 公式表示
3. 对于每一层的信号传递，因为矩阵运算，一个公式就可以表示
4. sigmoid函数不改变矩阵元素数量，而且每一层对应的矩阵中元素数应该和该层神经元的数量一致
5. 代码实现
      ~~~python
      def init_network():
         network = {}
         network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
         network['b1'] = np.array([0.1, 0.2, 0.3])
         network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
         network['b2'] = np.array([0.1, 0.2])
         network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
         network['b3'] = np.array([0.1, 0.2])
         return network
      def forward(network, x):
         W1, W2, W3 = network['W1'], network['W2'], network['W3']
         b1, b2, b3 = network['b1'], network['b2'], network['b3']
         a1 = np.dot(x, W1) + b1
         z1 = sigmoid(a1)           #第一层的激活函数
         a2 = np.dot(z1, W2) + b2
         z2 = sigmoid(a2)           #第二层的激活函数
         a3 = np.dot(z2, W3) + b3
         y = identity_function(a3)  #最后一层的激活函数，和之前的有所不同是，恒等函数
         return y
      network = init_network()
      x = np.array([1.0, 0.5])
      y = forward(network, x)
      print(y) # [ 0.31682708 0.69627909]
      ~~~
## 输出层的函数
- 输出层的函数和隐藏层的不同
- 本章主要的介绍函数有两个：恒等函数（用于回归问题）、softmax函数（用于分类问题）

### 恒等函数
1. 对于得到的数据直接输出
2. 该激活转化可以直接用一根箭头表示

### softmax函数
1. 公式表示
    $ y_ {k} $ = $ \frac {exp(a_ {k})}{\sum _ {i=1}^ {n}exp(a_ {i})} $ 
2. 图解：
- ![图片](/deep%20learning/3.png)

3. 由上图可知输出层的各个神经元都**受到所有输入信号的影响**
4. 实现函数
~~~python
def softmax(a):
   exp_a = np.exp(a)
   sum_exp_a = np.sum(exp_a)
   y = exp_a / sum_exp_a
    return y

~~~
5. softmax函数的作用
   该函数常用于概论的表示，但是因为e的指数函数是单调的，softmax函数并不会改变输出项中最大的一项，而且在神经网络学习过程中，直接会采用输出最大的值，同时该函数的计算也需要一定的计算量，所以在实际的问题解决中，该函数也常常被省略。(?)
6. softmax函数的问题和解决
   1. 问题：对于$e$的指数运算得到的超大之间进行除法运算，结果会出现“不确定”的情况，也就是“溢出”问题。
   2. 解决方式：减去减去输入信号中的最大值再进行指数运算
   3.推导过程：
 $ y_ {k} $ = $ \frac {exp(a_ {k)}}{\sum _ {i=1}^ {n}exp(a_ {i})} $ = $ \frac {Cexp(a_ {k})}{C\sum _ {i=1}^ {n}exp(a_ {i})} $ 
= $ \frac {exp(a_ {k}+\log C)}{\sum _ {i=1}^ {n}exp(a_ {i}+\log _ {C}} $ 
= $ \frac {exp(a_ {k}+C')}{\sum _ {i=1}^ {n}exp(a_ {i}+C')} $ 
   4. 修正后的softmax函数
   
   ~~~python  
   def softmax(a):
      c = np.max(a)
      exp_a = np.exp(a - c) # 溢出对策
      sum_exp_a = np.sum(exp_a)
      y = exp_a / sum_exp_a
      return y
   ~~~

## 手写数字识别
- 一个神经网络学习的例子
- 主要的过程分为两个部分，先由学习的过程中确定参数，再由这些参数去识别和推理
- 采用的数据集MNIST
1. 数据没下下下来，离谱
2. 输入数据的集合称为批。通过以批为单位进行推理处理，能够实现高速的运算。


---
# 第四章 神经网络的学习
- 内容：由数据自动决定权重参数的值
- 标准：寻找降低损失函数的权重值去、
- 方法：函数斜率的梯度法
## 从数据中学习
1. 学习方式
   1. 人：寻找方法解决问题
   2. 机器学习：人为设定特征量（根据具体问题），机器学习方法，解决问题
   3. 深度学习：神经网路一步到位，（中间没有人的介入）
2. 数据分类
   1. 训练数据
   2. 测试数据（获得数据的泛化能力，也成为监督数据）
   3. 要避免的问题：数据过拟合

## 损失函数
- 用来评价神经网络性能的“恶劣程度”的指标
- 在本章中介绍两个损失函数：均方误差、交叉熵误差
- [参考资料](https://zhuanlan.zhihu.com/p/35709485)

### mini-batch学习
1. 随机选取数据进行学习
2. 是对整体函数的近似


### 均方误差
1. 公式表示：
   E= $ \frac {1}{2} $ $ \sum $ $ (y_ {k}-t_ {k})^ {2} $ 
   eg.
   y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
   t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
   y表示的神经网络的输出
   t表示的监督数据，这里的处理方式是one-hot表示
2. one-hot表示：将正确解标签表示为1，其他标签表示为0
3. **这个前面的二分之一的系数的作用？**

### 多分类交叉熵误差
1. 公式表示
$ E=-  \sum \limits_{k}t_{k}  \log {y_ {k}}  $
2. 特点:仅由正确标签对应的输出有关（因为其他项的t为0）
3. 代码：
~~~python
def cross_entropy_error(y, t):
   delta = 1e-7
   return -np.sum(t * np.log(y + delta))  #np可以直接进行log运算
~~~
4. 和均方误差的结构有一致性

### 损失函数的意义
1. 在进行神经网络的学习时，**不能将识别精度作为指标**。因为如果以识别精度为指标，则参数的导数在绝大多数地方都会变为0。
2. 采用损失函数进行评估的时候，**需要一个连续的激活函数**（否则无法关注到参数的微小变化）

## 补充内容：信息论（理解熵）
- [主要参考文本](https://www.bilibili.com/read/cv15258489?spm_id_from=333.999.0.0)
- [主要参考视频](https://www.bilibili.com/video/BV15V411W7VB?spm_id_from=333.999.0.0)
- [最大似然估计](https://www.bilibili.com/read/cv14977249?spm_id_from=333.999.0.0)
- [JS散度](https://blog.csdn.net/Invokar/article/details/88917214)
- 定义本身是没有意义的，这个意义是他被定义了才被赋予的，他的关键是它定义了之后，整个体系是否能完成自洽
1. 信息的概念
![](notes_graph/mk2022-05-02-20-35-35.png)
- 而在信息论的观点之下，一个越小概率的的事情发生了，其实他的信息量就越大（因为前提更多）
2. 信息衡量的标尺
![](notes_graph/mk2022-05-02-20-41-11.png)
- 也就是说我们要同时表示满足这两个条件，概率上的，信息论上的
![](notes_graph/mk2022-05-02-20-43-09.png)
- 所以对于信息量中公式的计算，log前面的系数应该采用负数负数，因为概率小的事情结果应该更大，而log的底数选择是2是为了和二进制对应，
  
3. 一个系统的信息量
![](notes_graph/mk2022-05-02-20-37-07.png)
![](notes_graph/mk2022-05-02-20-56-14.png)

$$
\Eta(P):= E(P_{f})
$$
$$
\Eta(P)=-\sum ^{m}_{i=1} p_{i}\cdot log_{2} p_{i}
$$
4. KL散度（相对熵）
![](notes_graph/mk2022-05-02-21-02-25.png)
- kl散度是为了衡量两个概率系统的分布差距是多少（在不知道其中一个的概率分布的情况）
![](notes_graph/mk2022-05-02-21-04-51.png)
- 这个是kl散度的定义，表示的用s作为基准的散度
![](notes_graph/mk2022-05-03-01-16-37.png)
-而且根据吉布斯不等式，散度恒大于零
 
5. 交叉熵
![](notes_graph/mk2022-05-02-21-07-54.png)
- 通过变换kl散度的表达式，我们就可以得到交叉熵的表达

6. 交叉熵的损失函数
![](notes_graph/mk2022-05-03-01-24-39.png)

7. JS散度
   1. JS散度的提出：由于kl散度的不对称性容易在学习的过程中产生问题，所以就提出了js散度
   2. 公式
   $$
   M= \frac{P+Q}{2}
   $$  

   $$
   JSD(P||Q)=JSD(Q||P)=\frac{1}{2}KL(P||M)+\frac{1}{2}KL(Q||M)
   $$
   2. JS散度的问题：当两个概率模型不重叠的时候，JS散度会为一个定值$log_2$导致参数无法更新。
8. 互信息熵
   1. 可以把互信息看成由于知道 y 值而造成的 x 的不确定性的减小(反之亦然)（即Y的值透露了多少关于X 的信息量）
   2. 公式表示
   ![](notes_graph/mk2022-05-05-18-35-24.png)

## 聚类
1. 在无监督学习中的分类情况
![](notes_graph/mk2022-05-04-10-59-12.png)
2. 兰德指数
   1. 和accuracy是类似的 $RI=  \frac {TP+TN}{TP+TN+FP+FN} $
   
   |                 | same cluster | different cluster |
   | :-------------: | :----------: | :---------------: |
   |   same class    |    TP=50     |       FN=10       |
   | different class |     FP=5     |       TN=20       |

   2. 取值范围为0~1，越大表示这个聚类的效果越好
3. 调整兰德指数（Adjusted Rand index）
   1. 公式
   $$
     AMI=\frac{RI-E[RI]}{max(RI)-E[RI]}
   $$
   1. 取值范围为-1~1 越大表示这个聚类的效果越好
   

## 数值微分
- 微积分内容，略
- 求导的函数numerical_diff(函数模块,x的值)(这个函数要自己写？)
## 梯度
### 梯度的基本知识
1. 梯度的实现
~~~python
def numerical_gradient(f, x):
   h = 1e-4 # 0.0001
   grad = np.zeros_like(x) # 生成和x形状相同的数组
   for idx in range(x.size):
   tmp_val = x[idx]
   # f(x+h)的计算
   x[idx] = tmp_val + h
   fxh1 = f(x)
   # f(x-h)的计算
   x[idx] = tmp_val - h
   fxh2 = f(x)
   grad[idx] = (fxh1 - fxh2) / (2*h)
   x[idx] = tmp_val # 还原值
return grad
~~~
1. 梯度指示的方向是各点处的函数值**减小最多的方向**
2. 但是梯度指向的方向**不一定是最小值的方向**，也可能是鞍点
3. 当函数很复杂且呈扁平状时，梯度不再下降，可能会陷入学习高原
### 梯度学习法
1. 梯度学习中需要设定合适的学习率，类似于学习的步长
   1. 过大导致发散
   2. 过小很难收敛
- 学习率是一种超参数，超参数是指深度学习中需要人工设定的参数
2. 学习率如何选取？

## 深度学习的过程
epoch：每经过一个epoch，我们都会记录下训练数据和测试数据的识别精度。

# 第五章误差反向传播法
- 用来高效计算权重参数的梯度的方法
- 两种理解方法：基于数学式、计算图
  
## 计算图

### 使用方法
1. 从左到右进行计算
2. eg示意图
![图片](\notes_graph/calculation_chart.png)
- 在计算图的流程中给出每一步计算的结果
- 在每一个圆圈的节点标定这一步的数学计算
- 用箭头指向每一个圆圈节点给出这一步计算的数学含义
- 这种从左到右进行的计算叫做正向传播
1. 可以对于局部进行计算（类似于树中的一个节点可以做为子树的根节点）
### 反向传播
1. 在计算图中可以使用反向传播计算导数是计算图的最大的优点
2. eg示意图
![图片](\notes_graph/calculation_chart_cal.png)
- 采用反向的箭头，箭头下标定倍数
3. 方法：链式法则计算（微积分的求导的链式法则）

- 反向传播时，加法节点不影响节点的值（求导为1）
- 反向传播时，乘法节点乘以输入信号的翻转值

## 简单层的实现


### 乘法层的实现
~~~python
class MulLayer:
 def __init__(self):#初始化
 self.x = None
 self.y = None
 def forward(self, x, y):#前向传播
 self.x = x
 self.y = y
 out = x * y
 return out    
 def backward(self, dout):#反向传播 这里的dout 表示前一项传递的导数
 dx = dout * self.y  # 翻转x和y
 dy = dout * self.x
 return dx, dy

~~~
### 加法层的实现
~~~python
class AddLayer:
 def __init__(self): #初始化
 pass
 def forward(self, x, y):
 out = x + y
 return out
 def backward(self, dout):
 dx = dout * 1
 dy = dout * 1
 return dx, dy
~~~


## 激活函数的实现
###　ReLU层的实现
- ReLU函数：
1. 大于零不变
2. 小于零置零
3. 根据这样的性质正向传播时的x小于等于0，则反向传播中传给下游的信号将停在此处。
~~~python
class Relu:       #mask是一个numpy数组，储存的是真值
 def __init__(self):#初始化
 self.mask = None
 def forward(self, x):#前向传播
 self.mask = (x <= 0)#小于是TRUE 大于为FALSE
 out = x.copy()
 out[self.mask] = 0#把对应的位置变成0
 return out
 def backward(self, dout):
 dout[self.mask] = 0#把对应的位置变成0，信号将停在此处
 dx = dout
 return dx
~~~
4. 类比：ReLU层的作用就像电路中的开关一样。正向传播时，有电流通过的话，就将开关设为 ON；没有电流通过的话，就将开关设为 OFF。反向传播时，开关为ON的话，电流会直接通过；开关为OFF的话，则不会有电流通过。

### sigmoid层的实现
1. 计算图流程
![图片](notes_graph/sigmoid_%20calculation.png)
- 这里的\节点表示去倒数
经过化简
![图片](notes_graph/sigmoid_%20calculation_final.png)
2. 代码实现
~~~ python
class Sigmoid:
 def __init__(self):
 self.out = None
 def forward(self, x):
 out = 1 / (1 + np.exp(-x))
 self.out = out
 return out
 def backward(self, dout):
 dx = dout * (1.0 - self.out) * self.out
 return dx
~~~

## Affine/Softmax层的实现

### Affine层的实现
1. 神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿射变换”A。因此，这里将进行仿射变换的处理实现为“Affine层”，简而言之，Affine层就是矩阵的相乘
2. 正向传播：矩阵相乘（主要矩阵的维数要满足矩阵相乘的规则）
3. 反向传播：由于矩阵运算的特殊，这里要引入矩阵的转置这一概念
![图片](notes_graph/Affine_calculation.png)
- 注意转置矩阵乘的顺序
- X和形状其反向传播是导数的形状是相同，W也是同理
4. 批处理的affine: 输入x的对应位置变成n,反向传播时，各个数据的反向传播的值需要汇总为偏置的元素。
![图片]()
5. 代码实现
~~~python
class Affine:
 def __init__(self, W, b):
 self.W = W #表示权重
 self.b = b #表示偏置
 self.x = None
 self.dW = None
 self.db = None
 def forward(self, x):
 self.x = x
 out = np.dot(x, self.W) + self.b
 return out
 def backward(self, dout):
 dx = np.dot(dout, self.W.T)
 self.dW = np.dot(self.x.T, dout)
 self.db = np.sum(dout, axis=0) 
 #axis表示求sum的方式，具体方法见https://www.cnblogs.com/SupremeBoy/p/12955652.html
 return dx
~~~

### Softmax-with-Loss层的实现
- 在深度学习的学习阶段，需要softmax的正规化
- Softmax-with-Loss包含两个层次，softmax函数和损失函数部分
1. 图解
![图片](notes_graph/softmax_loss.png)
简化版
![图片](notes_graph/easy_softmax_loss.png)
- 神经网络学习的目的就是通过调整权重参数，使神经网络的输出（Softmax的输出）接近教师标签。因此，必须将神经网络的输出与教师标签的误差高效地传递给前面的层。刚刚的（y1 − t1, y2 − t2, y3 − t3）正是Softmax层的输出与教师标签的差，直截了当地表示了当前神经网络的输出与教师标签的误差。
- 使用“平方和误差”作为“恒等函数”的损失函数，反向传播才能得到（y1 −t1, y2 − t2, y3 − t3）这样“漂亮”的结果。
2. 代码实现
~~~ python
class SoftmaxWithLoss:
 def __init__(self):
 self.loss = None # 损失
 self.y = None # softmax的输出
 self.t = None # 监督数据（one-hot vector）
 def forward(self, x, t):
 self.t = t
 self.y = softmax(x)
 self.loss = cross_entropy_error(self.y, self.t)
 return self.loss
 def backward(self, dout=1):
 batch_size = self.t.shape[0] #批处理，
 dx = (self.y - self.t) / batch_size #传递给前面是单个数据的误差
 return dx
~~~

## 误差反向传播法的实现
### 深度学习的的结构
- **前提**
神经网络中有合适的权重和偏置，调整权重和偏置以便拟合训练数据的
过程称为学习。神经网络的学习分为下面4个步骤。
- **步骤1（mini-batch）**
从训练数据中随机选择一部分数据。
- **步骤2（计算梯度）**
计算损失函数关于各个权重参数的梯度。
- **步骤3（更新参数）**
将权重参数沿梯度方向进行微小的更新。
- **步骤4（重复）**
重复步骤1、步骤2、步骤3。

在学习了反向传播之后，我们可以更容易的求取梯度

### 对应误差反向传播法的神经网络的实现
** 一个两层神经网络的类**

1. 类中变量的说明

| 实例变量  | 说明                                                                                  |
| :-------: | :------------------------------------------------------------------------------------ |
|  params   | 保存神经网络的参数的字典型变量                                                        |
|           | params['W1']是第1层的权重，params['b1']是第1层的偏置                                  |
|           | params['W2']是第2层的权重，params['b2']是第2层的偏置                                  |
|  layers   | 保存神经网络的层的有序字典型变量。                                                    |
|           | 以layers['Affine1']、layers['ReLu1']、layers['Affine2']的形式，通过有序字典保存各个层 |
| lastLayer | 神经网络的最后一层。                                                                  |
|           | 本例中为SoftmaxWithLoss层                                                             |

2. 类中函数的说明

# 第六章：与学习相关的参数
1. 《深度学习入门》第6章
2. 正则化：L1正则化（Lasso回归）和L2正则化（岭回归）、**梯度截断**、Dropout
3. 参数初始化的方法：随机初始化、Xavier初始化、He初始化、**正交初始化**
4. 参数标准化方法及其适用模型类型：**batch normalization**、**layer normalization**
5. 信息熵、交叉熵、KL散度、JS散度、聚类的损失（兰德指数、互信息（信息增益））
## 参数的更新
|      名称      | 含义                                                                  |
| :------------: | :-------------------------------------------------------------------- |
|     最优化     | 使损失函数最小的过程                                                  |
| 随机梯度下降法 | 沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，简称SGD |
### 探险家的故事
- 有一个性情古怪的探险家。他在广袤的干旱地带旅行，坚持寻找幽深的山谷。他的目标是要到达最深的谷底（他称之为“至深之地”）。这也是他旅行的目的。并且，他给自己制定了两个严格的“规定”：一个是不看地图；另一个是把眼睛蒙上。因此，他并不知道最深的谷底在这个广袤的大地的何处，而且什么也看不见。在这么严苛的条件下，这位探险家如何前往“至深之地”呢？他要如何迈步，才能迅速找到“至深之地”呢？
- 至深之地对应着最优解
- 蒙眼表示学习的过程没有人为的参与，采用的是深度学习的方法
- 下降的过程也就是按照坡度（梯度）学习的过程

## 学习方法
- 采用的学习的例子$z=$
![](notes_graph/mk2022-05-01-07-16-22.png)
- 参考视频
### SGD
1. 公式表示
![图片](notes_graph/sgd.png)
2. 代码实现(这里是伪代码)
~~~python
network = TwoLayerNet(...)
optimizer = SGD()
for i in range(10000):
 ...
 x_batch, t_batch = get_mini_batch(...) # mini-batch
 grads = network.gradient(x_batch, t_batch)
 params = network.params
 optimizer.update(params, grads)
 ...
~~~
3. 学习曲线
![图片](notes_graph/sgd_learning.png)
- 缺点：如果函数的形状非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。因此，我们需要比单纯朝梯度方向前进的SGD更聪明的方法。**SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。****而且学习率也一直是一个定值，在终点出很容易导致震荡问题**


### Momentum
1. 公式表示
![图片](notes_graph/momentum.png)
2. 和SGD的相比，加上加速度这一个量，这样的作用是通过累加来削减在震荡方向上的运动
3. 摩擦系数
   1. 在上面的公式中可以看到在存在一个$\eta$ 的量，这个可以理解成“摩擦系数”，数学上叫做**指数加权平均**，也是说通过不断对久的数据乘上这样一个系数，来**削减越早数据的影响程度**，同时，**避免学习的步长过大**的问题。
   2. 0.9的选取
      1. 为 0 时，退化为未优化前的梯度更新
      2. 为 1 时， 表示完全没有摩擦，如前所述，这样会存在大的问题；
      3. 取 0.9 是一个较好的选择。可能是 0.9 的 60 次方约等于 0.001，相当仅考虑最近的60轮迭代所产生的的梯度，这个数值看起来相对适中合理。
4. 代码实现
~~~ python
class Momentum:
   def __init__(self, lr=0.01, momentum=0.9):
      self.lr = lr
      self.momentum = momentum
      self.v = None
 def update(self, params, grads):
   if self.v is None:
      self.v = {}
      for key, val in params.items():
         self.v[key] = np.zeros_like(val)
   for key in params.keys():
      self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
      params[key] += self.v[key]
 ~~~
4. 学习曲线
![图片](notes_graph/momentum_learning.png)

- 和SGD相比，我们发现“之”字形的“程度”减轻了。这是因为虽然**x轴方向上**受到的力非常小，但是一直在同一方向上受力，所以朝同一个方向**会有一定的加速**。反过来，虽然**y轴方向上**受到的力很大，但是因为**交互地受到正方向和反方向的力**，它们会**互相抵消**，所以y轴方向上的速度不稳定。因此，和SGD时的情形相比，可以更快地朝x轴方向靠近，减弱“之”字形的变动程度。

### AdaGrad
1. 公式表示
![图片](notes_graph/AdaGrad.png)
2. 在AdaGrad中，使用了学习率衰减（learning rate decay），即随着学习的进行，使学习率逐渐减小。实际上，一开始“多”学，然后逐渐“少”学的方法,不独AdaGrad算法应该如此，因为越靠近目标的位置，学习率的减少有助于逼近。

3. 代码实现
~~~ python
class AdaGrad:
 def __init__(self, lr=0.01):
   self.lr = lr
   self.h = None
 def update(self, params, grads):
   if self.h is None:
      self.h = {}
   for key, val in params.items():
      self.h[key] = np.zeros_like(val)
   for key in params.keys():
      self.h[key] += grads[key] * grads[key]
      params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
      #微小值1e-7。这是为了防止当self.h[key]中有0时，将0用作除数的情况)
~~~
4. 学习曲线
![图片](notes_graph/AdaGrad_learning.png)

- 可以观察到，学习的步长逐渐减小
5. AdaGrad的缺点
![](notes_graph/mk2022-05-02-16-18-10.png)
- 红色的线是AdaGrad算法
- 上图中我们可以明显的看到，当AdaGrad通过一个学习的平原之时，学习率已经衰减到很低的一个值，由于对之前所有的学习过程，AdaGrad有“记忆”所以就算通过了这个“平原”，AdaGrad也很难继续“加速”，所以对AdaGrad的一个基本的改进思路就是让它逐渐遗忘距离越远的位置的学习，再此基础上就得到了RMprop算法(用的也是指数加权平均法)

### Adam
1. 可以理解成以上两种方法的结合
- Adam会设置 3个超参数。一个是学习率（论文中以α出现），另外两个是一次momentum系数β1和二次momentum系数β2。根据论文，标准的设定值是β1为 0.9，β2 为 0.999。设置了这些值后，大多数情况下都能顺利运行
- 进行超参数的“偏置校正”是Adam的特征
2. 学习图像
![图片](notes_graph/adam.png)

### 四种方法的比较
![](notes_graph/mk2022-05-01-15-19-43.png)
- 这是四种方法对于如上问题的学习过程，SGD和adam为现在比较常用的学习方法
![](notes_graph/learning_compare.gif)


### nesterov


## 权重初始值的设置
1. 一开始把权重值设置的较小，有利于最终权重值的减小，从而抑制过拟合的问题
2. 但是把初始的权重值设置成0，会导致神经方向传播的时候出现对称的结果，所以必须采用随机生成得到的权重值

### 权重初始值的两个问题
#### 问题1：梯度消失
1. 采用一个五层的神经网络进行这个问题的说明
~~~python
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
   return 1 / (1 + np.exp(-x))
x = np.random.randn(1000, 100) # 1000个数据
node_num = 100 # 各隐藏层的节点（神经元）数
hidden_layer_size = 5 # 隐藏层有5层
activations = {} # 激活值的结果保存在这里
for i in range(hidden_layer_size):
   if i != 0:
      x = activations[i-1]
   w = np.random.randn(node_num, node_num) * 1#这里权重生成的都是1
   z = np.dot(x, w)
   a = sigmoid(z) # sigmoid函数
   activations[i] = a
# 绘制直方图
for i, a in activations.items():
   plt.subplot(1, len(activations), i+1)
   plt.title(str(i+1) + "-layer")
   plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
~~~
2. 图像
![图片](notes_graph/vanishing_gradient_problem.png)
- 各层的激活值呈偏向0和1的分布。这里使用的sigmoid函数是S型函数，随着输出不断地靠近0（或者靠近1），它的导数的值逐渐接近0。因此，偏向0和1的数据分布会造成反向传播中梯度的值不断变小，最后消失。这个问题称为梯度消失（gradient vanishing）。层次加深的深度学习中，梯度消失的问题可能会更加严重。

#### 问题2：表现力受限
1. 将初始权重值改成标准差为0.01的高斯分布后
2. 图像
![图片](notes_graph/limited_presentation.png)
- 这次呈集中在0.5附近的分布。因为不像刚才的例子那样偏向0和1，所以不会发生梯度消失的问题。但是，激活值的分布有所偏向，说明在表现力上会有很大问题。也就是说，100个神经元都输出几乎相同的值，那么也可以由1个神经元来表达基本相同的事情

### Xavier初始值
1. 改变随机的标准差，和上一层的节点数量有关
![图片](notes_graph/Xavier_init.png)
2. 改进代码
~~~ python
node_num = 100 # 前一层的节点数
w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
~~~
3. 图像
![图片](notes_graph/Xavier_gragh.png)

### He初始值
1. 和ReLU激活函数配合的很好
2. 标准差为$\sqrt \frac{2}{n}$的高斯分布
3. 因为ReLU的负值区域的值为0，为了使它更有广度，所以需要2倍的系数
4. 图像
![](notes_graph/mk2022-05-05-19-14-01.png)

## normalization
### batch normalization
1. 对数据分布进行正规化
2. batch normalization的作用
   1. 可以使学习快速进行（可以增大学习率）
   2. 不那么依赖初始值（对于初始值不用那么神经质）。
   3. 抑制过拟合（降低Dropout等的必要性）
3.
![](notes_graph/mk2022-05-05-19-21-38.png)
![](notes_graph/mk2022-05-05-19-23-20.png)
![](notes_graph/mk2022-05-05-19-24-12.png)
### max normalization
### layer normalization



## 正则化

（插入一个图片）
![](notes_graph/mk2022-05-03-23-45-36.png)
从一开始，这个函数的图像就变成了一个凸函数，而解决凸函数的问题是相对容易的
1. 正则化：
2. L1正则化(曼哈顿距离)
![](notes_graph/mk2022-05-03-23-54-54.png)
- 和L2正则化相比，可以看到在对权重约束条件之下，L1正则化的极值点更容易落到坐标轴上，比如现在的神经网络的权重有两个维度（对应着两个方向上的轴，现在一个轴是零，我只用考虑另外一个轴上的数据来找到极值点，这也就是L**1正则化可以带来稀疏性**的原因。

2. 和L2正则化(阿基米德距离)
![](notes_graph/mk2022-05-03-23-50-36.png)
![](notes_graph/mk2022-05-03-23-51-23.png)

3. 为什是L1和L2
- 首先这个数值要大于1保证这是**一个凸函数**，而如果其他的数值其实和这两个有着相似的性质，但是计算起来又比较麻烦的情况，就没有必要使用了
4. 采用正则化是否会带来问题
![](notes_graph/mk2022-05-04-00-06-00.png)
- 对于这样的情况，约束的条件本身是松弛的，这样不会带来额外的偏差
![](notes_graph/mk2022-05-04-00-08-11.png)
- 而对于这样的情况，我们看到，本来不加约束条件的情况下可以收敛到黄色圆点的位置，而在加了约束条件之后，被约束到了蓝色原点的位置，那这样是否会对损失函数的减少造成影响？可以有这样的处理
$$
minJ(W,b)=minJ(\alpha \cdot W,\alpha \cdot b)
$$
这样的话我们可以通过约束**得到一条路径**，而非一个点（实际情况这个系数的取值当然也是不一样的）
4. 贝叶斯角度
   1. 先验概率和后验概率
   ![](notes_graph/mk2022-05-04-02-49-53.png)
   ![](notes_graph/mk2022-05-04-02-45-10.png)
   ![](notes_graph/mk2022-05-04-02-46-04.png)
   
   1. L2正则化:高斯分布的$log(P(W))$
   ![](notes_graph/mk2022-05-04-02-15-25.png)
   1. L1正则化：拉普拉斯的$log(P(W))$
   ![](notes_graph/mk2022-05-04-02-41-08.png)
   ![](notes_graph/mk2022-05-04-02-42-37.png)
   - 从这个图中我们
   1. 从贝叶斯角度理解深度学习
 

## 过拟合以及解决方式
### 过拟合
1. 什么是过拟合
![](notes_graph/mk2022-05-04-00-24-00.png)
- 从图上我们就可以看到，这样训练出来的神经网络可以精确的识别训练集中的数据，但是面对测试集的数据呢就未必可以做的很好
![](notes_graph/mk2022-05-04-00-24-57.png)
2. 导致过拟合的原因
   1. 训练数据太小
   2. 神经网络过于复杂
### 正则化和权值衰减
- 解决方法1：引入L1,L2范数
1. 采用L2范数进行举例
![](notes_graph/mk2022-05-04-00-57-44.png)
改进结果：
![](notes_graph/mk2022-05-04-00-59-36.png)
![](notes_graph/mk2022-05-04-01-00-04.png)
- 可以看到，训练数据的识别精度下降了，而测试集的识别精度上升了
1. 岭回归与Lasso回归
- [参考](https://blog.csdn.net/hzw19920329/article/details/77200475)
- lesson回归：引入的是L1范数惩罚项
- ![](notes_graph/mk2022-05-04-01-34-53.png)
- 岭回归：引入的是L2范数惩罚项
- ![](notes_graph/mk2022-05-04-01-34-12.png)

1. 权值衰减和正则化
L2正则化和权值衰减正则化在**标准随机梯度下降算法中是等价的**(当被学习速率缩放时)，但正如我们演示的，这不是自适应梯度算法的情况，如Adam。 虽然这些算法的常见实现采用L2正则化(通常称其为“权值衰减”，这可能是由于我们暴露的不等价性造成的误导）
有待了解的问题
[adam的优化问题](https://arxiv.org/pdf/1711.05101.pdf)
[岭回归与Lasso回归和权值衰减的对比](https://towardsdatascience.com/weight-decay-l2-regularization-90a9e17713cd)
### Dropout
- 解决方法2：Dropout
- [参考](https://blog.csdn.net/stdcoutzyx/article/details/49022443)
- [参考2](https://blog.csdn.net/weixin_43953686/article/details/105978308)
- [参考文献（参考1最后的论文）](https://arxiv.org/pdf/1506.08700.pdf)
1. Dropout是一种在学习的过程中随机删除神经元来降低过拟合问题的方法
2. 示意图：
![](notes_graph/mk2022-05-04-16-15-10.png)
3. 关于dropout的一个比喻
在自然界中，在中大型动物中，一般是有性繁殖，**有性繁殖是指后代的基因从父母两方各继承一半**。但是从直观上看，似乎无性繁殖更加合理，因为**无性繁殖可以保留大段大段的优秀基因**。**而有性繁殖则将基因随机拆了又拆，破坏了大段基因的联合适应性**。但是自然选择中毕竟没有选择无性繁殖，而选择了有性繁殖，须知物竞天择，适者生存。我们先做一个假设，那就是**基因的力量在于混合的能力而非单个基因的能力**。不管是有性繁殖还是无性繁殖都得遵循这个假设。

- 为了证明有性繁殖的强大，我们先看一个概率学小知识。
- 比如要搞一次恐怖袭击，两种方式：
- 集中50人，让这50个人密切精准分工，搞一次大爆破。
- 将50人分成10组，每组5人，分头行事，去随便什么地方搞点动作，成功一次就算。
- 哪一个成功的概率比较大？ 显然是后者。因为将一个大团队作战变成了游击战。

那么，类比过来，有性繁殖的方式不仅仅可以将优秀的基因传下来，还可以降低基因之间的联合适应性，使得复杂的大段大段基因联合适应性变成比较小的一个一个小段基因的联合适应性。**dropout也能达到同样的效果，它强迫一个神经单元，和随机挑选出来的其他神经单元共同工作，达到好的效果。消除减弱了神经元节点间的联合适应性，增强了泛化能力。**

个人补充一点：那就是植物和微生物大多采用无性繁殖，因为他们的生存环境的变化很小，因而不需要太强的适应新环境的能力，所以保留大段大段优秀的基因适应当前环境就足够了。而高等动物却不一样，要准备随时适应新的环境，因而将基因之间的联合适应性变成一个一个小的，更能提高生存的概率。也就是说神经网络在不仅要面对训练集的数据，还要接受测试集的数据。

所以从这个角度理解dropout为什么可以降低泛化能力,dropout掉不同的隐藏神经元就类似在训练不同的网络，随机删掉一半隐藏神经元导致网络结构已经不同，**整个dropout过程就相当于对很多个不同的神经网络取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。**
4. 代码（注重原理的展示）
~~~ Python
class Dropout:
   def __init__(self, dropout_ratio=0.5):# 遗忘程度一般是0.5，因为经过数学计算最后生成的结构最多（也有使用0.3的时候）
      self.dropout_ratio = dropout_ratio
      self.mask = None
   def forward(self, x, train_flg=True):
      if train_flg:
         self.mask = np.random.rand(*x.shape) > self.dropout_ratio
         return x * self.mask
      else:
      return x * (1.0 - self.dropout_ratio)
   def backward(self, dout):
      return dout * self.mask
~~~
- 这个实现的过程是每次前向传播的时候生成一个和x大小一样的数组，将大于dropout_ratio的部分置false,在通过和$x$相乘来起到“遗忘的作用”
5. 学习的结果
![](notes_graph/mk2022-05-04-16-29-12.png)
- 可以看到，增加了dropout之后学习上升的过程变慢了，但是最后的结果上泛化能力增加了

## 超参数的确定
1. 确定超参数的过程
   1. 从数据集中选取验证集（大概百分之二十）
   2. 设定超参数的范围（用量级划定范围）
   3. 从设定的超参数范围中随机采样。
   4. 使用步骤1中采样到的超参数的值进行学习，通过验证数据评估识别精
   度（但是要将epoch设置得很小）
   5. 重复步骤2和步骤3（100次等），根据它们的识别精度的结果，缩小超参数的范围
   - 这个训练的过程要持续很久（几天
   - 可以用贝叶斯最优化缩短时间

# 第七章 卷积神经网络
- 主要应用：图像识别、语音识别
## 卷积神经网络的一般结构
1. 普通的神经网络结构
![](notes_graph/mk2022-05-09-13-43-10.png)
>相邻层之间的神经元都有连接，称之为"**全连接（full-connected）**"
2. 卷积神经网络
3. ![](notes_graph/mk2022-05-09-14-04-31.png)
>可以看到在卷积神经网络中，加入了convolution和pooling层，连接顺序是“Convolution - ReLU -（Pooling），但是靠近输出的层中依旧使用了之前的“Affine - ReLU”组合。此外，最后的输出层中使用了之前的“Affine - Softmax”组合。这些都是一般的CNN中比较常见的结构。






































   





   






   

