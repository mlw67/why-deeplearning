# 第五章：SVM与Transformer的等价性

## 5.1 支持向量机回顾

### 5.1.1 线性SVM

**支持向量机(Support Vector Machine, SVM)**是经典机器学习中最优雅的算法之一。其核心思想是寻找**最大间隔超平面**。

对于线性可分数据$\{(x_i, y_i)\}_{i=1}^n$，$y_i \in \{-1, +1\}$，SVM求解：

$$\min_{w, b} \frac{1}{2}\|w\|^2$$
$$\text{s.t. } y_i(w^T x_i + b) \geq 1, \quad \forall i$$

### 5.1.2 对偶形式与支持向量

通过拉格朗日对偶，得到对偶问题：

$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle$$
$$\text{s.t. } \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

关键观察：
- 解仅依赖于**内积**$\langle x_i, x_j \rangle$
- 只有边界上的点（**支持向量**）对应非零$\alpha_i$
- 决策函数：$f(x) = \sum_{i \in SV} \alpha_i y_i \langle x_i, x \rangle + b$

### 5.1.3 核方法

**核技巧(Kernel Trick)**允许隐式地在高维空间工作：

将内积$\langle x_i, x_j \rangle$替换为核函数$K(x_i, x_j)$，其中$K(x, x') = \langle \phi(x), \phi(x') \rangle$对应某个特征映射$\phi$。

常用核函数：

| 核 | 公式 | 特点 |
|---|------|------|
| 多项式 | $(x^T x' + c)^d$ | 捕捉高阶交互 |
| RBF/高斯 | $\exp(-\gamma\|x-x'\|^2)$ | 无限维特征空间 |
| 拉普拉斯 | $\exp(-\gamma\|x-x'\|_1)$ | 更稀疏 |

### 5.1.4 软间隔与正则化

对于非线性可分数据，引入松弛变量$\xi_i$：

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i$$
$$\text{s.t. } y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

这等价于正则化的hinge损失：
$$\min_{w, b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \max(0, 1 - y_i(w^T x_i + b))$$

## 5.2 注意力机制的核方法解释

### 5.2.1 注意力回顾

Transformer中的缩放点积注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：查询矩阵
- $K \in \mathbb{R}^{m \times d_k}$：键矩阵
- $V \in \mathbb{R}^{m \times d_v}$：值矩阵

### 5.2.2 Softmax作为核

观察到softmax可以写成核的形式：

$$\text{softmax}(QK^T)_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{l=1}^m \exp(q_i^T k_l / \sqrt{d_k})}$$

定义**softmax核**：
$$K_{\text{sm}}(q, k) = \exp(q^T k / \sqrt{d_k})$$

这是一个**正定核**，对应某个隐式特征映射$\phi$。

### 5.2.3 指数核的特征映射

对于指数核$K(x, y) = \exp(x^T y)$，有显式特征映射：

$$\phi(x) = e^{-\|x\|^2/2} \left[1, x_1, \ldots, x_d, \frac{x_1^2}{\sqrt{2!}}, \frac{x_1 x_2}{1!}, \ldots \right]$$

这是**无限维**的！特征空间包含所有阶多项式。

### 5.2.4 随机特征近似

**定理5.1（Random Fourier Features, Rahimi & Recht, 2007）**：对于移位不变核，可以用随机特征近似：

$$K(x, y) \approx \phi(x)^T \phi(y)$$

对于softmax核，可以使用**正随机特征(Positive Random Features)**：

$$\exp(q^T k) \approx \mathbb{E}_{\omega}[\exp(\omega^T q - \|q\|^2/2) \cdot \exp(\omega^T k - \|k\|^2/2)]$$

这是**Performer**架构的理论基础。

## 5.3 Transformer作为核机器

### 5.3.1 注意力作为核回归

将注意力重写为：

$$\text{Attention}(q, K, V) = \frac{\sum_{j=1}^m K_{\text{sm}}(q, k_j) v_j}{\sum_{j=1}^m K_{\text{sm}}(q, k_j)}$$

这正是**Nadaraya-Watson核回归**的形式！

**定义5.1（Nadaraya-Watson估计器）**：
$$\hat{f}(x) = \frac{\sum_{i=1}^n K(x, x_i) y_i}{\sum_{i=1}^n K(x, x_i)}$$

### 5.3.2 注意力与SVM的联系

**定理5.2（Attention as Support Vector Expansion）**：注意力输出可以写成：

$$\text{Attention}(q) = \sum_{j=1}^m \alpha_j(q) v_j$$

其中$\alpha_j(q) = \text{softmax}(q^T k_j / \sqrt{d_k})_j$是**数据依赖**的权重。

与SVM对比：
| | SVM | Attention |
|---|-----|-----------|
| 权重 | $\alpha_i$固定，由训练确定 | $\alpha_j(q)$随查询变化 |
| 基函数 | 核$K(x, x_i)$ | 值向量$v_j$ |
| 稀疏性 | 仅支持向量 | 通常非稀疏(softmax) |

### 5.3.3 硬注意力与SVM

如果将softmax替换为硬选择（argmax），注意力变成：

$$\text{HardAttention}(q) = v_{j^*}, \quad j^* = \argmax_j q^T k_j$$

这与最近邻分类器相似，也与**SVM的支持向量概念**相关——只有"最相关"的数据点参与决策。

### 5.3.4 Transformer层作为核变换

整个Transformer层可以视为：

1. **自注意力**：用核方法聚合信息
2. **FFN**：非线性特征变换
3. **残差连接**：保持恒等映射通道

这与深度核学习的框架一致：逐层构建越来越复杂的核。

## 5.4 理论等价性的实践意义

### 5.4.1 计算复杂度的改进

基于核视角，可以设计更高效的注意力：

**定理5.3（线性注意力）**：如果可以将$K_{\text{sm}}(q, k) = \phi(q)^T \psi(k)$分解为有限维特征，则：

$$\text{Attention}(Q, K, V) = \frac{\Phi(Q)(\Psi(K)^T V)}{\Phi(Q)\Psi(K)^T \mathbf{1}}$$

复杂度从$O(n^2 d)$降至$O(n d^2)$或$O(n d D)$（$D$是特征维度）。

### 5.4.2 Performer与线性注意力

**Performer**使用正随机特征(FAVOR+)来近似softmax注意力核（注意：这与一般指数核的无限维特征映射不同，是一种有限维近似）：

$$\phi(x) = \frac{\exp(-\|x\|^2/2)}{\sqrt{D}} [\exp(\omega_1^T x), \ldots, \exp(\omega_D^T x)]$$

其中$\omega_i \sim \mathcal{N}(0, I)$。这满足$\mathbb{E}[\phi(x)^T\phi(y)] = \exp(x^T y)$。

这给出：
- 理论上无偏近似softmax注意力
- 线性时间复杂度$O(n d D)$
- 保持了核方法的表达能力

### 5.4.3 稀疏注意力的核解释

稀疏注意力模式可以视为使用**局部核**：

$$K_{\text{local}}(i, j) = \begin{cases} K(q_i, k_j) & |i-j| \leq w \\ 0 & \text{otherwise} \end{cases}$$

这减少了计算量，但可能丢失长程依赖。

### 5.4.4 泛化理论的启示

核方法有成熟的泛化理论。将这些理论迁移到Transformer：

**命题5.1**：基于核视角，Transformer的泛化界可以用RKHS范数刻画：

$$\text{泛化误差} \leq O\left(\frac{\|f\|_{\mathcal{H}_K}}{\sqrt{n}}\right)$$

这解释了为什么预训练（学习好的核）有助于泛化。

### 5.4.5 架构设计的指导

核视角为架构设计提供理论指导：

1. **核选择**：不同的注意力变体对应不同的核
2. **深度**：更多层 = 更复杂的核组合
3. **宽度**：更宽的FFN = 更丰富的特征空间
4. **位置编码**：在核中注入位置信息

## 5.5 超越等价性：Transformer的独特之处

### 5.5.1 数据依赖的核

传统核方法使用**固定**核函数。Transformer的注意力是**数据依赖**的——核参数由输入本身决定。

这是关键区别：Transformer可以动态调整"相似性度量"。

### 5.5.2 参数化的特征映射

SVM使用固定特征映射$\phi$（由核隐式定义）。Transformer**学习**特征映射：

$$\phi_\theta(x) = \text{FFN}_\theta(\text{Attention}_\theta(x))$$

参数$\theta$通过训练优化。

### 5.5.3 组合多个核

多头注意力可以视为**多核学习(Multiple Kernel Learning)**：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

每个头学习不同的核，最终组合。

### 5.5.4 上下文依赖

Transformer最独特的能力是**上下文学习(In-Context Learning)**：

- 可以在推理时适应新任务
- 无需更新参数
- 通过注意力机制实现

这超越了传统核方法的范畴，是Transformer的独特优势。

## 5.6 本章小结

本章的核心要点：

1. **SVM**是基于核方法的经典机器学习算法
2. **注意力机制**可以解释为使用softmax核的核回归
3. 这种联系启发了**高效注意力**的设计（如Performer）
4. 核方法的**泛化理论**可以部分迁移到Transformer
5. Transformer超越传统核方法的关键：**数据依赖核**和**学习特征映射**

---

## 参考文献

1. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*.
2. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
3. Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *NeurIPS*.
4. Choromanski, K., et al. (2021). Rethinking attention with performers. *ICLR*.
5. Tsai, Y. H., et al. (2019). Transformer dissection: An unified understanding for transformer's attention via the lens of kernel. *EMNLP*.
6. Wright, M. A., & Gonzalez, J. E. (2021). Transformers are deep infinite-dimensional non-mercer binary kernel machines. *arXiv*.

---

[← 上一章：JL引理与维度嵌入理论](04-jl-embedding.md) | [下一章：大模型架构的理论基础 →](06-large-models.md)
