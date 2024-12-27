# Transformer

- ❤[2017. **Attention Is All You Need**](https://arxiv.org/abs/1706.03762)❤ (Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. **self attention(Scaled Dot-Product Attention(SDPA), Multi-Head Attention)**)

![](https://arxiv.org/html/1706.03762v7/extracted/1706.03762v7/Figures/ModalNet-21.png)

## self-attention
在深度学习中，经常使用卷积神经网络（CNN）或循环神经网络（RNN）对序列进行编码。 想象一下，有了注意力机制之后，我们将词元序列输入注意力池化中， 以便同一组词元同时充当查询、键和值。 具体来说，每个查询都会关注所有的键－值对并生成一个注意力输出。 由于查询、键和值来自同一组输入，因此被称为 自注意力（self-attention), 也被称为内部注意力（intra-attention）。 

## positional-encoding

在处理token序列时，循环神经网络是逐个的重复地处理token， 而自注意力则因为并行计算而放弃了顺序操作。 为了使用序列的顺序信息，通过在输入表示中添加 位置编码（positional encoding）来注入绝对的或相对的位置信息。 位置编码可以通过学习得到也可以直接固定得到。 


## Model Architecture
### Positionwise Feed-Forward Networks
- [**Yoshua Bengio et al. 2003 MLP LM: A Neural Probabilistic Language Model**](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (MLP LM)
- https://en.wikipedia.org/wiki/Activation_function
- https://en.wikipedia.org/wiki/Rectifier_(neural_networks) (ReLU)

### Residual Connection, Layer Normalization and Dropout (for deep NN)
- **Residual connection(add)** frome [2015. Kaiming He ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
residual connection 残差连接（residual connection）是深度神经网络中的一种常见技术，它的作用是解决梯度消失和梯度爆炸问题，同时也可以帮助模型更快地收敛。
  - 在传统的神经网络中，每个层的输出都是通过对前一层输出的非线性变换得到的。但是，当网络的深度增加时，前一层的输出可能会被过度压缩或拉伸，导致信息丢失或重复。这种情况下，网络的性能可能会受到影响，同时也会出现梯度消失或梯度爆炸的问题。
  - 残差连接通过在每个层的输出与输入之间添加一个跨层连接来解决这个问题。更具体地说，残差连接将前一层的输出直接添加到当前层的输出中，从而提供了一种绕过非线性变换的路径。这样，网络就可以学习到在信息压缩或拉伸后保留重要信息的方法，同时也减轻了梯度消失或梯度爆炸的问题。
  - Residual connections alleviate unstable gradient problems and they help the model to converge faster (for deep)
  - “residual connections carry positional information to higher layers, among other information.” - One of the transformer authors, Ashish Vaswani


- [2016. Geoffrey Hinton: **Layer Normalization**](https://arxiv.org/abs/1607.06450) (for RNN, Transformer in NLP task difference seq len)
  - Benefits of Layer Normalization:
    - Reduces the impact of internal covariate shift: This improves the stability of training and helps the network learn faster.
    - Improves gradient flow: Layer normalization helps to alleviate the vanishing gradient problem, which can occur in deep networks.
    - Reduces the need for careful initialization: Layer normalization can make the network less sensitive to the initial values of its weights and biases.
    - Improves performance: Layer normalization has been shown to improve the performance of various deep learning models on a variety of tasks.
  - Drawbacks:
    - Increased computational cost: Normalizing each layer independently can be computationally expensive, especially for large models with many layers. This can lead to slower training times and higher inference costs.


- **Dropout** Regularization from [2012. Geoffrey Hinton: Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) 解决过拟合
  - 训练神经网络的时候经常会遇到过拟合的问题，过拟合具体表现在：模型在训练数据上损失函数较小，预测准确率较高；但是在测试数据上损失函数比较大，预测准确率较低
  - Dropout说的简单一点就是：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征，

### Encoder
Encoder self-attention: 编码器包含自注意力层。在自注意力层中，所有键、值和查询都来自同一位置，在本例中是编码器中前一层的输出。编码器中的每个位置可以关注编码器上一层中的所有位置.

### Decoder
MaskedDecoder self-attention: 类似地，解码器中的自注意力层允许解码器中的每个位置关注解码器中直到并包括该位置的所有位置。我们需要防止解码器中的左向信息流以保留自回归属性。我们通过屏蔽（设置为 -∞) softmax 输入中对应于非法连接的所有值。

### Encoder-Decoder
Encoder-Decoder attention: 在“编码器-解码器注意力”层中，查询来自前一个解码器层，内存键和值来自编码器的输出。这允许解码器中的每个位置都参与输入序列中的所有位置。这模仿了序列到序列模型中典型的编码器-解码器注意机制，例如 [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)

---

## Transformers

### [encoder-decoder](./Encoder-Decoder/)
NMT([Neural machine translation](https://en.wikipedia.org/wiki/Neural_machine_translation)):
- [Transformer](./Encoder-Decoder/model.py): 原始论文实现,使用 Multi30k 德语-英语翻译任务的真实示例。该任务比论文中考虑的 WMT 任务小得多，但是可以复现实现结果
- T5

### [encoder-only](./Encoder/)

### [decoder-only](./Decoder/)
- GPT
  - https://bbycroft.net/llm (可视化GPT)

---

# reference
- [Tensor2Tensor Transformers New Deep Models for NLP](https://nlp.stanford.edu/seminar/details/lkaiser.pdf) (from RNN CNN with attention (e.g.: WaveNet,ByteNet) -> transformers all attention)
- ⭐原始论文源码：https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
- ⭐[illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- ⭐⭐[**Transformers from Scratch**](https://e2eml.school/transformers.html) 
- ⭐⭐[**The Annotated Transformer**](https://nlp.seas.harvard.edu/annotated-transformer/)
- https://github.com/weedge/doraemon-nb/blob/main/AnnotatedTransformer.ipynb
- ⭐[3blue1brown: Visualizing transformers and attention](https://www.youtube.com/watch?v=KJtZARuO3JY&t=2124s)
- https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html