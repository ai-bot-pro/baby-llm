# intro

**Normalization:**: 归一化是特征缩放的一种形式，归一化是把数据压缩到一个区间内

# reference

- https://en.wikipedia.org/wiki/Normalization_(statistics)
- https://bingqiangzhou.github.io/2020/08/26/DailySummary-NormalizationStandardizationRegularization.html
  - 归一化(Normalization): 归一化是特征缩放的一种形式，归一化是把数据压缩到一个区间内，用于训练权重, 推理时可以移除.
  - 标准化(Standardization): 标准化，与归一化一样，也是特征缩放的一种形式。标准化公式也称z-score公式，是一个数与平均数的差再除以标准差。比如:缩放点积注意力计算: 假设查询和键的所有元素都是独立的随机变量， 并且都满足零均值和单位方差(标准化数据)， 那么两个向量的点积的均值为$0$，方差为$d$。 为确保无论向量长度如何， 点积的方差在不考虑向量长度的情况下仍然是$1$， 我们再将点积除以$\sqrt{d}$。
  - 正则化(Regularization): 正则化是指为解决适定性问题或过拟合而加入额外信息的过程。比如机器学习(NN)中常用的有L1正则化，L2正则化; 深度学习中常用的 Dropout ([2012. Geoffrey Hinton: Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)), 计算机视觉模型通过 Label Smoothing ([2015. Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567))正则化来避免过拟合。

## Batch Normalization (BatchNorm)
批量归一化旨在减少内部协变量偏移，从而加速深度神经网络的训练。它通过修复层输入的均值和方差的标准化步骤来实现这一点。通过减少梯度对参数规模或其初始值的依赖性，批量归一化还对通过网络的梯度流产生有益的影响。这允许使用更高的学习率而不存在发散的风险。此外，批量归一化可以规范模型并减少Dropout的需要。

- [2015.BatchNorm: **Batch Normalization:** Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (for CNN RNN in cumpute visition task, in NLP task difference seq len not good)

## Layer Normalization (LayerNorm)
- [2016.LayerNorm: **Layer Normalization**](https://arxiv.org/abs/1607.06450) (for RNN, Transformer in NLP task difference seq len)
  - Benefits of Layer Normalization:
    - Reduces the impact of internal covariate shift: This improves the stability of training and helps the network learn faster.
    - Improves gradient flow: Layer normalization helps to alleviate the vanishing gradient problem, which can occur in deep networks.
    - Reduces the need for careful initialization: Layer normalization can make the network less sensitive to the initial values of its weights and biases.
    - Improves performance: Layer normalization has been shown to improve the performance of various deep learning models on a variety of tasks.
  - Drawbacks:
    - Increased computational cost: Normalizing each layer independently can be computationally expensive, especially for large models with many layers. This can lead to slower training times and higher inference costs.

### RMS Layer Normalization (RMSLayerNorm)
RMSNorm 的性能与 LayerNorm 相当，但运行时间减少, loss收敛快些
- [2019.RMSLayerNorm: Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

------
- How BatchNorm Works: https://www.youtube.com/watch?v=34PDIFvvESc&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=83
- BatchNorm in PyTorch: https://www.youtube.com/watch?v=8AUDn7iF2DY&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=84
- Why BatchNorm Works: https://www.youtube.com/watch?v=uI19wIdzh9M&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=85