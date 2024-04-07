# torch

- https://nn.labml.ai/
- https://github.com/labmlai/annotated_deep_learning_paper_implementations

## method_params

- cross_entropy
```python
def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor
```
```
这个函数是用来计算交叉熵损失的。交叉熵损失通常用于多分类问题中，用来衡量模型输出与真实标签之间的差异。

下面是这个函数的参数解释：

input (Tensor): 输入参数，表示模型的预测值，通常是未经过归一化的对数（logits）值。支持的形状包括 (C)、(N, C) 或者 (N, C, d_1, d_2, ..., d_K)，其中 C 是类别数，N 是批量大小，后面的维度 d_1, d_2, ..., d_K 表示可能的其他维度，至少有一维。

target (Tensor): 目标参数，表示真实的类别索引或者类别概率。支持的形状与 input 相同，如果是类别索引，则每个值应在 [0, C) 范围内，如果是类别概率，则每个值应在 [0, 1] 范围内。

weight (Tensor, optional): 权重参数，可以手动设置每个类别的权重，形状为 C。

size_average (bool, optional): 已弃用参数，指定损失是否在批次中的每个损失元素上平均。如果为 True，则将在批次中的每个损失元素上平均。默认为 True。

ignore_index (int, optional): 忽略索引参数，指定一个目标值，在计算梯度时被忽略，不会对梯度产生贡献。默认为 -100。

reduce (bool, optional): 已弃用参数，指定是否对每个批次的观察值进行平均或求和。默认为 True。

reduction (str, optional): 指定对输出的减少方式，可以是 'none'、'mean' 或者 'sum'。'none' 表示不进行减少，'mean' 表示输出之和将被输出中的元素数量除，'sum' 表示输出将被求和。默认为 'mean'。

label_smoothing (float, optional): 标签平滑参数，介于 [0.0, 1.0] 之间的浮点数，指定在计算损失时的平滑程度，其中 0.0 表示不进行平滑。标签将变为原始真实标签和均匀分布的混合。默认为 0.0。
```