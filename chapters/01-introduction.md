# 第一章：引言与历史背景

## 1.1 神经网络的起源与发展历程

### 1.1.1 早期神经网络（1943-1969）

神经网络的历史可以追溯到1943年，当Warren McCulloch和Walter Pitts提出了第一个数学化的神经元模型——**M-P模型**。这个模型将神经元抽象为一个简单的二值逻辑单元：

$$y = \sigma\left(\sum_{i=1}^{n} w_i x_i - \theta\right)$$

其中，$\sigma$是阶跃函数，$w_i$是权重，$\theta$是阈值。

1958年，Frank Rosenblatt提出了**感知机(Perceptron)**，这是第一个可以自动学习权重的神经网络模型。感知机能够解决线性可分问题，一度引发了人们对人工智能的巨大热情。

然而，1969年Minsky和Papert在《Perceptrons》一书中指出，单层感知机无法解决**XOR问题**这样的非线性可分问题，这直接导致了神经网络研究的第一次寒冬。

### 1.1.2 反向传播与多层网络（1986-1995）

1986年，Rumelhart、Hinton和Williams发表的**反向传播算法(Backpropagation)**论文重新点燃了神经网络研究的热情。反向传播允许有效地训练多层网络，从而解决了XOR问题以及更复杂的非线性问题。

在这一时期，出现了许多重要的理论成果：
- **万能近似定理**(Universal Approximation Theorem)：Cybenko(1989)和Hornik(1991)证明，具有单隐藏层的前馈神经网络可以以任意精度近似任何连续函数。
- **卷积神经网络**(CNN)：LeCun等人(1989)将反向传播应用于卷积网络，成功用于手写数字识别。

### 1.1.3 第二次寒冬与SVM的崛起（1995-2006）

尽管有了这些进展，90年代中期神经网络研究再次陷入低谷：

1. **计算资源限制**：当时的硬件难以支持大规模网络训练
2. **梯度消失问题**：深层网络难以训练
3. **过拟合问题**：缺乏有效的正则化方法
4. **理论理解不足**：神经网络被视为"黑箱"

与此同时，**支持向量机(SVM)**因其优雅的数学理论和强大的泛化保证而成为主流。SVM有：
- 凸优化保证全局最优解
- 结构风险最小化的理论基础
- 核方法提供的灵活性

### 1.1.4 深度学习的复兴（2006至今）

2006年，Hinton等人提出了**深度信念网络(DBN)**和逐层预训练方法，标志着深度学习的复兴。关键突破包括：

| 年份 | 突破 | 意义 |
|------|------|------|
| 2012 | AlexNet | 在ImageNet上大幅超越传统方法 |
| 2014 | GAN | 生成模型的革命 |
| 2015 | ResNet | 解决了深度网络的训练问题 |
| 2017 | Transformer | 统一了NLP和CV的架构 |
| 2020 | GPT-3 | 展示了大规模模型的涌现能力 |

## 1.2 从感知机到深度学习的理论演进

### 1.2.1 表达能力的理论发展

神经网络理论研究的一个核心问题是**表达能力(Expressiveness)**：什么样的函数类可以被神经网络表示？

**定理1.1（万能近似定理）**：设$\sigma:\mathbb{R} \to \mathbb{R}$是非多项式的连续激活函数。对于任意紧集$K \subset \mathbb{R}^d$上的连续函数$f$和任意$\epsilon > 0$，存在一个单隐藏层神经网络$N$使得：
$$\sup_{x \in K} |N(x) - f(x)| < \epsilon$$

这个定理告诉我们，**宽度足够大**的浅层网络理论上可以近似任何连续函数。但它没有告诉我们：
1. 需要多少神经元？
2. 如何找到这样的网络？
3. 深度是否有帮助？

### 1.2.2 深度的力量

后续研究表明，**深度**在表达能力上有本质优势：

**定理1.2（深度分离）**：存在一类函数，它们可以被深度为$d$、宽度为$O(\text{poly}(n))$的网络表示，但任何深度小于$d$的网络需要指数级宽度才能近似。

这解释了为什么深层网络在实践中优于浅层网络——它们可以用更少的参数表达更复杂的函数。

### 1.2.3 优化理论的演进

训练神经网络本质上是一个非凸优化问题。早期理论担心：
- 局部最小值的存在会导致训练失败
- 鞍点会使梯度下降停滞

现代理论揭示了一些意外的发现：
1. **过参数化**的网络几乎没有坏的局部最小值
2. **损失景观**在高维空间中具有特殊性质
3. **隐式正则化**使SGD偏好简单解

## 1.3 为什么深度学习在21世纪崛起

深度学习的成功是多种因素协同作用的结果：

### 1.3.1 数据规模

互联网时代带来了海量数据：
- ImageNet：1400万张标注图像
- Common Crawl：数万亿词的文本语料
- YouTube：每天上传数百万小时视频

### 1.3.2 计算能力

GPU和专用硬件的发展：
- 2012年：NVIDIA GPU使AlexNet训练成为可能
- 2016年：TPU专为深度学习设计
- 2020年：单模型训练成本达数百万美元

### 1.3.3 算法创新

关键的算法改进：
- **ReLU激活函数**：缓解梯度消失
- **Batch Normalization**：稳定训练
- **Dropout**：有效正则化
- **残差连接**：训练极深网络
- **注意力机制**：建模长程依赖

### 1.3.4 理论理解的深化

我们现在更好地理解了：
- 为什么过参数化有助于泛化
- 为什么SGD能找到好的解
- 为什么深度网络能学到层次化表示

## 1.4 本系列笔记的目标与范围

### 1.4.1 目标

本系列笔记旨在回答以下问题：

1. **架构有效性**：为什么Transformer等架构如此成功？
2. **理论基础**：这些架构背后有什么数学原理支撑？
3. **未来方向**：还有哪些潜在的架构值得探索？

### 1.4.2 范围

我们将重点讨论：
- 宽度与深度的理论分析
- NTK(Neural Tangent Kernel)理论
- JL引理与维度嵌入
- SVM与Transformer的理论联系
- 大模型的规模定律

### 1.4.3 不包含的内容

- 深度学习的数学基础（线性代数、微积分、概率论）
- 具体的实现细节和代码
- 工程优化技巧

### 1.4.4 阅读建议

建议读者：
1. 具备基本的机器学习背景
2. 熟悉神经网络的基本概念
3. 有一定的数学成熟度

---

## 参考文献

1. McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity.
2. Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain.
3. Minsky, M., & Papert, S. (1969). Perceptrons: An Introduction to Computational Geometry.
4. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.
5. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function.
6. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks.
7. Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A fast learning algorithm for deep belief nets.
8. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.
9. Vaswani, A., et al. (2017). Attention is all you need.

---

[下一章：宽度与深度理论 →](02-width-depth.md)
