# 第三章：神经切线核(NTK)理论

## 3.1 NTK的起源与基本概念

### 3.1.1 从核方法到NTK

2018年，Jacot等人提出了**神经切线核(Neural Tangent Kernel, NTK)**理论，这是连接神经网络与核方法的重要桥梁。

**核心洞察**：在无限宽度极限下，神经网络的训练动力学可以用一个确定性的核函数来描述。

### 3.1.2 NTK的定义

对于参数为$\theta$的神经网络$f(x; \theta): \mathbb{R}^d \to \mathbb{R}$，其**神经切线核**定义为：

**定义3.1（NTK）**：
$$\Theta(x, x') = \left\langle \frac{\partial f(x; \theta)}{\partial \theta}, \frac{\partial f(x; \theta')}{\partial \theta} \right\rangle = \sum_{i=1}^{P} \frac{\partial f(x; \theta)}{\partial \theta_i} \cdot \frac{\partial f(x'; \theta)}{\partial \theta_i}$$

其中$P$是参数总数。

### 3.1.3 NTK与梯度流

考虑用均方损失训练网络：
$$\mathcal{L}(\theta) = \frac{1}{2} \sum_{i=1}^{n} (f(x_i; \theta) - y_i)^2$$

梯度流(连续时间梯度下降)的动力学为：
$$\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L}(\theta)$$

这导致网络输出的演化：
$$\frac{df(x; \theta)}{dt} = -\sum_{j=1}^{n} \Theta(x, x_j) (f(x_j; \theta) - y_j)$$

用矩阵形式表示，设$\mathbf{f} = (f(x_1), \ldots, f(x_n))^T$，$\mathbf{y} = (y_1, \ldots, y_n)^T$：
$$\frac{d\mathbf{f}}{dt} = -\Theta (\mathbf{f} - \mathbf{y})$$

其中$\Theta_{ij} = \Theta(x_i, x_j)$是NTK的Gram矩阵。

### 3.1.4 NTK的关键性质

**定理3.1（NTK收敛，Jacot et al., 2018）**：在适当的初始化条件下，当网络宽度趋向无穷时：
1. NTK $\Theta(x, x')$收敛到一个确定性极限$\Theta^{\infty}(x, x')$
2. 在训练过程中，NTK保持（近似）不变

这意味着无限宽度网络的训练动力学变成了**线性**的！

## 3.2 无限宽度网络的核方法等价

### 3.2.1 线性化动力学

当NTK在训练过程中保持不变时，动力学方程：
$$\frac{d\mathbf{f}}{dt} = -\Theta^{\infty} (\mathbf{f} - \mathbf{y})$$

有闭式解：
$$\mathbf{f}(t) = (I - e^{-\Theta^{\infty} t}) \mathbf{y} + e^{-\Theta^{\infty} t} \mathbf{f}(0)$$

当$t \to \infty$时：
$$\mathbf{f}(\infty) = \mathbf{y}$$

这表明网络可以完美拟合训练数据（在正定核条件下）。

### 3.2.2 核回归的联系

对于新样本$x^*$，预测为：
$$f(x^*; \infty) = f(x^*; 0) + \Theta^{\infty}(x^*, X) (\Theta^{\infty})^{-1} (\mathbf{y} - \mathbf{f}(0))$$

这正是**核回归**的形式！神经网络训练等价于用NTK进行核回归。

### 3.2.3 显式NTK公式

对于全连接网络，可以递归计算NTK。设深度为$L$的网络：

$$f^{(0)}(x) = x$$
$$g^{(l)}(x) = W^{(l)} f^{(l-1)}(x) + b^{(l)}$$
$$f^{(l)}(x) = \sigma(g^{(l)}(x))$$

NTK递归公式为：

$$\Sigma^{(0)}(x, x') = x^T x'$$

$$\Sigma^{(l)}(x, x') = \sigma_w^2 \cdot F_\sigma(\Sigma^{(l-1)}) + \sigma_b^2$$

$$\dot{\Sigma}^{(l)}(x, x') = \sigma_w^2 \cdot \dot{F}_\sigma(\Sigma^{(l-1)})$$

$$\Theta^{(l)}(x, x') = \Theta^{(l-1)}(x, x') \cdot \dot{\Sigma}^{(l)}(x, x') + \Sigma^{(l)}(x, x')$$

其中$F_\sigma$和$\dot{F}_\sigma$是与激活函数相关的函数。

### 3.2.4 ReLU网络的NTK

对于ReLU激活函数，有闭式公式：

$$F_{\text{ReLU}}(u, v, \rho) = \frac{1}{2\pi}\sqrt{uv}(\sin\theta + (\pi - \theta)\cos\theta)$$

其中$\cos\theta = \rho / \sqrt{uv}$，$\rho = \Sigma^{(l-1)}(x, x')$。

## 3.3 NTK的训练动力学

### 3.3.1 收敛保证

**定理3.2（全局收敛）**：设$\Theta^{\infty}$是NTK的Gram矩阵，$\lambda_{\min}$是其最小特征值。如果$\lambda_{\min} > 0$，则梯度下降的训练损失以指数速率收敛：
$$\mathcal{L}(t) \leq e^{-2\lambda_{\min} t} \mathcal{L}(0)$$

### 3.3.2 过参数化与NTK稳定性

为什么无限宽度下NTK保持不变？直观解释：

- 参数变化$\Delta\theta$与初始化$\theta_0$相比很小
- 这使得网络在初始化附近线性化
- 等价于在参数空间做一阶Taylor展开

**定理3.3（懒惰训练，Lazy Training）**：当宽度$m$足够大时，训练过程中：
$$\|\theta(t) - \theta(0)\| = O(1/\sqrt{m})$$

参数变化相对于初始化是微小的。

### 3.3.3 泛化理论

NTK框架也给出了泛化界：

**定理3.4（NTK泛化）**：在NTK regime下，泛化误差满足：
$$\text{泛化误差} \leq O\left(\frac{\|f^*\|_{\mathcal{H}_\Theta}^2}{n}\right)$$

其中$\mathcal{H}_\Theta$是NTK诱导的RKHS，$f^*$是目标函数。

### 3.3.4 优化与泛化的平衡

NTK理论揭示了一个有趣的现象：

| 宽度 | 优化 | 泛化 |
|------|------|------|
| 较窄 | 可能困难 | 可能更好 |
| 较宽 | 更容易 | 可能过拟合 |
| 无限 | 线性化，易优化 | 等价于核回归 |

## 3.4 NTK的局限性与超越

### 3.4.1 特征学习的缺失

NTK regime的一个关键局限是**缺乏特征学习**：

- 权重变化很小意味着中间层表示几乎不变
- 网络只在输出层做线性组合
- 这与深度学习"学习表示"的直觉矛盾

**实验发现**：实际训练的网络通常不在NTK regime中，它们确实学习了有用的特征。

### 3.4.2 有限宽度效应

有限宽度网络表现出NTK理论未捕捉的行为：

1. **NTK的演化**：有限宽度下，NTK会在训练过程中变化
2. **特征学习**：中间层表示会显著变化
3. **非线性动力学**：训练不再是线性核回归

### 3.4.3 超越NTK：均场理论

**均场(Mean Field)**理论提供了另一个视角，允许特征学习：

- 宽度趋于无穷，但学习率也相应缩放
- 权重分布本身在演化，而非单个权重
- 可以描述更丰富的动力学

### 3.4.4 超越NTK：张量程序

**张量程序(Tensor Programs)**框架统一了多种无限宽度理论：

- NTK regime
- 均场 regime  
- 最大更新参数化(μP)

这个框架表明，通过不同的参数化和学习率缩放，可以得到不同的训练行为。

### 3.4.5 实践意义

尽管有局限，NTK理论仍有实践价值：

1. **初始化设计**：理解NTK有助于设计好的初始化
2. **架构分析**：可以用NTK分析不同架构的归纳偏置
3. **超参数选择**：NTK提供了学习率选择的理论指导
4. **迁移学习**：NTK解释了为什么预训练有效

## 3.5 NTK与其他核的联系

### 3.5.1 NTK vs NNGP核

- **NNGP核**：描述网络输出的先验分布（初始化时）
- **NTK**：描述训练动力学（梯度流）

两者是相关但不同的对象：
$$\text{NNGP}: K^{(L)}(x, x') \quad \text{NTK}: \Theta^{(L)}(x, x')$$

### 3.5.2 NTK与经典核

NTK可以视为某种意义上的"最优"核：

- 对于给定的架构，NTK是该架构能实现的核
- 不同激活函数给出不同的NTK
- ReLU NTK是弧余弦核的变体

### 3.5.3 深层NTK的特性

深层NTK有一些特殊性质：

**观察3.1**：随着深度增加，NTK趋向于一个"universal"核，失去了对输入的敏感性。

这解释了为什么极深网络难以训练——梯度信息在层间传播时逐渐丢失。

## 3.6 本章小结

本章的核心要点：

1. **NTK**将神经网络训练与核方法联系起来
2. 无限宽度下，训练动力学**线性化**，可以精确分析
3. NTK提供了**收敛保证**和**泛化界**
4. 主要局限是**缺乏特征学习**
5. 后续理论（均场、张量程序）超越了NTK的限制

---

## 参考文献

1. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS*.
2. Lee, J., et al. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. *NeurIPS*.
3. Du, S., et al. (2019). Gradient descent finds global minima of deep neural networks. *ICML*.
4. Arora, S., et al. (2019). On exact computation with an infinitely wide neural net. *NeurIPS*.
5. Chizat, L., Oyallon, E., & Bach, F. (2019). On lazy training in differentiable programming. *NeurIPS*.
6. Yang, G. (2020). Tensor programs I: Wide feedforward or recurrent neural networks of any architecture are gaussian processes. *NeurIPS*.
7. Yang, G., & Hu, E. J. (2021). Tensor programs IV: Feature learning in infinite-width neural networks. *ICML*.

---

[← 上一章：宽度与深度理论](02-width-depth.md) | [下一章：JL引理与维度嵌入理论 →](04-jl-embedding.md)
