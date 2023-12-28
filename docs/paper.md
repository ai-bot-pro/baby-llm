
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) tokenizer 使用 BPE 算法，将一个大的词汇表分成多个小的词汇表，然后将每个词汇表中的词汇进行编码，最后将编码后的词汇进行拼接，形成一个新的词汇表。
  - https://github.com/google/sentencepiece
  - 训练的分词模型 用于 模型的训练和推断
  - 训练的词汇表 用于 词汇表的构建

- 定义一个神经网络Embedding嵌入权重层：torch.nn.Embedding(params.vocab_size, params.dim)
  - embedding 把数据集合映射到向量空间，进而把数据进行向量化的过程
  - 将稀疏向量转化为稠密向量，便于上层神经网络的处理
  - 在训练的过程中 找到一组合适的向量，来刻画现有的数据集合
  - 将客观世界中的物体不失真的映射到高维特征空间中，进而可以使用这些embedding向量 实现分类、回归和预测等操作
  - 详细介绍：
  - https://mp.weixin.qq.com/s/FsqCNPtDPMdH0WGI0niELw
  - https://vickiboykis.com/what_are_embeddings/

- 定义一个全连接神经网络线性权重层：torch.nn.Linear 特征维度转化，

$$y = xA^T + b$$

  - 训练权重和偏置bias
  - 可将输出维度变大，参数变多了，模型的拟合能力也变强
  - 多层组合为多层感知机 (Multilayer Perceptron, MLP)，后面接一个激活函数,

- [Root Mean Square Layer Normalization](https://openreview.net/pdf?id=SygkZ3MTJE) RMSNorm 归一化权重层,

$$ \begin{align} \begin{split} & \bar{a}i = \frac{a_i}{\text{RMS}(\mathbf{a})} g_i, \quad \text{where}~~ \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n} \sum{i=1}^{n} a_i^2}. \end{split}\nonumber \end{align} $$

  - https://github.com/bzhangGo/rmsnorm
  - RMSNorm 根据均方根 (RMS) 对一层神经元的输入求和进行正则化，从而赋予模型重新缩放不变性和隐式学习率自适应能力。
  - RMSNorm 计算更简单，因此比 [LayerNorm](https://arxiv.org/abs/1607.06450) 更高效。
- [attention is all u need](https://arxiv.org/pdf/1706.03762.pdf), transformer 自注意力机制 
  - MHA: multihead attention. iterate over all heads -> GQA: Grouped-Query Attention(llama2)
  - MHA简单来说就是通过多次线性投影linear projection得到原始输入的多个子空间，然后再每个子空间分别进行SDPA(Scaled Dot-Product Attention), 再把SDPA的结果进行聚合Concatenation,最后再做一个linear projection。
  - SDPA的全称为Scaled Dot-Product Attention, 属于点积注意力机制， 简单一句话来说就是，根据Query (Q)与Key之间的匹配度来对Value进行加权，而事实上不管是Query, Key还是Value都来自于输入，因此所谓的SDPA本质上是对输入信息信息进行重组。
  - 通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。
  - self-attention 机制用于计算序列中当前token关注与其他token的联系，就是一个序列内的token，互相看其他token对自己的影响力有多大
  
- [Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) RoPE relative positional embeddings
  
- [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) dropout 解决过拟合
  - 训练神经网络的时候经常会遇到过拟合的问题，过拟合具体表现在：模型在训练数据上损失函数较小，预测准确率较高；但是在测试数据上损失函数比较大，预测准确率较低
  - Dropout说的简单一点就是：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征，
  
- [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202.pdf) FFN 中激活函数SwiGLU/SiLU
  
- [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) 交叉熵 cross_entropy 
    - 交叉熵损失函数的计算公式：
    - $L = -\frac{1}{N}\sum_{i=1}^{N}y_i\log(p_i)$
    - $y_i$ 表示真实值，$p_i$ 表示预测值
    - 目的是获取输出概率（P）并测量与真值的距离
    - 使用softmax函数或者sigmoid函数将网络的输出转换为概率
    - PyTorch中它自带的命令torch.nn.functional.cross_entropy已经将转换概率值的操作整合了进去，所以不需要额外进行转换概率值的操作

- [backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
  - [3blue1brown-nn-bp-video](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) 优化器 AdamW: Adam + Weight Decay（权重衰减）; 
  - learning rate和weight decay 参数调整，固定一个调整另一个

- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf) 模型训练效率的评估工作 比如 MFU


简单概括:

- 训练一个神经网络(transformer结构)，数据通过tokenizer分词，输入token数据embedding, 首先要跑一遍前向Forward的过程，算算wx+b，算算激活函数，之后计算Loss function,利用损失函数进行Backward对参数求导得到梯度Grad，拿到Grad后扔给优化器Optimizer更新模型权重，反复迭代，直到损失函数收敛。

# 参考学习：
1. https://vickiboykis.com/what_are_embeddings/
3. http://neuralnetworksanddeeplearning.com/
2. https://www.3blue1brown.com/topics/neural-networks
4. https://karpathy.ai/zero-to-hero.html
5. https://courses.d2l.ai/zh-v2/ (看沐神视频学)


# LLMs:
- [GPT1: Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised)
- [GPT2: Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [InstructGPT: Aligning language models to follow instructions](https://openai.com/research/instruction-following)
- [ChatGPT](https://openai.com/blog/chatgpt)
- [GPT-4 Technical Report](https://openai.com/research/gpt-4)
- [LLaMA: Open and Efficient Foundation Language Models](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)