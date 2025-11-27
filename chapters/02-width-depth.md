# 第二章：宽度与深度理论

## 2.1 万能近似定理与宽度

### 2.1.1 经典万能近似定理

万能近似定理是神经网络理论的基石之一。它告诉我们：任何连续函数都可以被单隐藏层神经网络以任意精度近似。

**定理2.1（Cybenko, 1989; Hornik, 1991）**：设$\sigma$是一个非常数、有界、单调递增的连续函数。对于任意紧集$K \subset \mathbb{R}^n$上的连续函数$f$和任意$\epsilon > 0$，存在整数$m$、实数$v_i, b_i$和向量$w_i \in \mathbb{R}^n$使得：

$$F(x) = \sum_{i=1}^{m} v_i \sigma(w_i^T x + b_i)$$

满足$|F(x) - f(x)| < \epsilon$对所有$x \in K$成立。

### 2.1.2 近似的代价：宽度需求

虽然万能近似定理保证了存在性，但没有告诉我们需要多少神经元。实际上，达到精度$\epsilon$所需的神经元数量可能非常大：

**定理2.2（宽度下界）**：对于某些连续函数$f$，要使单隐藏层网络达到精度$\epsilon$，需要的神经元数量$m$满足：

$$m = \Omega\left(\frac{1}{\epsilon^{d/r}}\right)$$

其中$d$是输入维度，$r$是函数的光滑度。这被称为**维度灾难(Curse of Dimensionality)**。

### 2.1.3 ReLU网络的近似理论

对于ReLU激活函数，有更精确的刻画：

**定理2.3（ReLU近似）**：设$f: [0,1]^d \to \mathbb{R}$是Lipschitz连续的，Lipschitz常数为$L$。则存在一个宽度为$O(L^d/\epsilon^d)$的ReLU网络$N$使得：

$$\|N - f\|_\infty \leq \epsilon$$

### 2.1.4 宽度的局限性

尽管增加宽度可以提高近似精度，但存在根本性限制：

1. **计算效率**：参数数量随宽度线性增长，计算成本相应增加
2. **样本复杂度**：更多参数通常需要更多训练数据
3. **表达效率**：某些函数用宽网络表达效率极低

这自然引出一个问题：**深度能否帮助克服这些限制？**

## 2.2 深度的表达能力优势

### 2.2.1 深度分离定理

**定理2.4（Telgarsky, 2016）**：存在一类函数$f_k$可以被深度为$O(k^3)$、宽度为$O(1)$的ReLU网络精确表示，但任何深度为$O(k)$的网络需要宽度$\Omega(2^k)$才能近似到常数精度。

这说明深度可以**指数级地**减少所需宽度。

### 2.2.2 直观理解：函数组合

深度网络的力量来自**函数组合**。考虑一个简单例子：计算$x^{2^n}$。

**浅层方法**：直接多项式展开需要$2^n$项

**深层方法**：通过$n$次平方运算
$$x \to x^2 \to x^4 \to \cdots \to x^{2^n}$$
只需要$n$层，每层$O(1)$宽度。

### 2.2.3 深度与层次化表示

深度网络自然地学习**层次化特征**：

```
输入层：像素
   ↓
第1层：边缘
   ↓
第2层：纹理
   ↓
第3层：部件
   ↓
第4层：物体
   ↓
输出层：类别
```

这种层次结构与人类视觉系统和语言的层次结构相匹配，这可能解释了深度学习在感知任务上的成功。

### 2.2.4 表达复杂度的度量

**定义2.1（轨迹复杂度）**：对于网络$N: \mathbb{R}^d \to \mathbb{R}$，其**线性区域数量**$R(N)$定义为：
$$R(N) = |\{r : \exists x \in r, N|_r \text{是线性的}\}|$$

对于ReLU网络，有以下结果：

**定理2.5**：深度为$L$、每层宽度为$n$的ReLU网络，其线性区域数量上界为：
$$R(N) \leq \left(\prod_{l=1}^{L-1} \sum_{j=0}^{n} \binom{n}{j}\right) \cdot \sum_{j=0}^{n}\binom{n}{j}$$

这可以简化为$O\left(\left(\frac{n}{d}\right)^{dL}\right)$，表明线性区域数量随深度指数增长。

## 2.3 宽度与深度的权衡

### 2.3.1 宽度-深度等价性

在某种意义上，宽度和深度可以相互替换：

**定理2.6（Lu et al., 2017）**：对于任意深度为$L$、最大宽度为$W$的ReLU网络$N$，存在一个深度为2、宽度为$O(WL)$的网络$N'$使得$N' = N$。

然而，这种等价是有代价的——宽度需求可能变得很大。

### 2.3.2 实践中的权衡

在实践中，最优的宽度-深度配置取决于：

| 因素 | 偏好宽度 | 偏好深度 |
|------|----------|----------|
| 硬件 | GPU并行优化 | 内存限制 |
| 数据 | 小数据集 | 大数据集 |
| 任务 | 简单模式 | 层次化特征 |
| 正则化 | Dropout有效 | 残差连接 |

### 2.3.3 ResNet的启示

ResNet的成功表明，通过**残差连接**可以有效训练极深网络：

$$y = F(x) + x$$

这解决了深度网络的两个关键问题：
1. **梯度消失**：恒等映射确保梯度可以直接传播
2. **优化难度**：网络只需学习残差$F(x)$，而非完整映射

### 2.3.4 深度的边际效益递减

实验表明，增加深度的收益存在边际递减：

- ResNet-152相比ResNet-101提升有限
- 非常深的网络可能需要特殊技巧（如随机深度）
- 计算预算固定时，宽度和深度需要平衡

## 2.4 无限宽度极限与均场理论

### 2.4.1 无限宽度的惊人简化

当网络宽度趋向无穷时，会出现一些意外的简化：

**观察2.1（Neal, 1996）**：单隐藏层网络在无限宽度极限下，其输出分布趋近于**高斯过程**。

设网络为：
$$f(x) = \frac{1}{\sqrt{m}} \sum_{i=1}^{m} v_i \sigma(w_i^T x + b_i)$$

当$m \to \infty$且权重独立同分布初始化时，根据中心极限定理：
$$f(x) \xrightarrow{d} \mathcal{GP}(0, K)$$

其中$K$是由激活函数和初始化确定的核函数。

### 2.4.2 神经网络高斯过程(NNGP)

**定义2.2（NNGP核）**：对于深度为$L$的网络，NNGP核递归定义为：

$$K^{(0)}(x, x') = x^T x' + \beta^2$$

$$K^{(l)}(x, x') = \sigma_w^2 \cdot \mathbb{E}_{(u,v) \sim \mathcal{N}(0, \Sigma^{(l-1)})}[\sigma(u)\sigma(v)] + \beta^2$$

其中$\Sigma^{(l-1)} = \begin{pmatrix} K^{(l-1)}(x,x) & K^{(l-1)}(x,x') \\ K^{(l-1)}(x',x) & K^{(l-1)}(x',x') \end{pmatrix}$。

### 2.4.3 均场理论视角

**均场理论(Mean Field Theory)**将神经网络视为相互作用粒子系统的极限：

在无限宽度下，每个神经元可以视为从某个**概率分布**中采样，整体行为由这个分布的演化描述。

这给出了另一种理解训练动力学的方式：优化不再是在参数空间进行，而是在**概率测度空间**进行。

### 2.4.4 无限宽度的局限

尽管无限宽度理论提供了有价值的洞察，它有重要局限：

1. **与有限宽度的差距**：实际网络是有限宽度的
2. **特征学习**：NNGP核是固定的，不能学习特征
3. **训练动力学**：实际训练可能偏离无限宽度预测

这些局限在NTK理论中得到部分解决，我们将在下一章详细讨论。

## 2.5 本章小结

本章的核心要点：

1. **万能近似定理**保证了宽度足够大的浅层网络的表达能力，但代价可能是指数级的宽度
2. **深度分离定理**表明深度可以指数级地减少所需宽度
3. 深度网络的优势来自**函数组合**和**层次化表示**
4. 实践中需要权衡宽度和深度，考虑计算资源和任务特性
5. **无限宽度极限**提供了理论分析的简化框架

---

## 参考文献

1. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*.
2. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*.
3. Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
4. Lu, Z., et al. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.
5. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
6. Neal, R. M. (1996). Priors for infinite networks. *Bayesian Learning for Neural Networks*.
7. Lee, J., et al. (2018). Deep neural networks as gaussian processes. *ICLR*.
8. Poole, B., et al. (2016). Exponential expressivity in deep neural networks through transient chaos. *NeurIPS*.

---

[← 上一章：引言与历史背景](01-introduction.md) | [下一章：神经切线核(NTK)理论 →](03-ntk.md)
