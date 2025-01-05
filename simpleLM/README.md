# Simple Language Model
简单的生成式语言模型(GLM), 用于了解GLM训练推理过程
## datasets
- tinyshakespeare.txt
```shell
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O datas/tinyshakespeare.txt
```
- 红楼梦.txt
```shell
wget https://raw.githubusercontent.com/shjwudp/shu/master/books/%E7%BA%A2%E6%A5%BC%E6%A2%A6.txt -O datas/红楼梦.txt
```
## tokenizer
这里英文以一个单个字符来分词，字典是数据集中的字母集合，进行排序后，对应token为字母对应的索引。
```python
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
```
tips: 这里用数据集里的字符集作为一个简单的tokenizer 词表用来分词, 简单实现； 未使用像[sentencepiece](https://github.com/google/sentencepiece)、[tiktoken](https://github.com/openai/tiktoken) 这些分词器库，对分词器进行训练，获取词表。

## model

1. Bigram LM: 一个包含vocab_size张量的嵌入权重层模块(nn.Embedding)，用来训练学习类似二元语言模型token的概率 $P(t_i\|t_{i-1})$ 权重参数 $W_e$

2. MLP(Multilayer Perceptron) LM: 每层为全连接线性权重层;  
   - 该模型定义了一个输入层，三个隐藏层(每层维度逐层指数减少)，一个输出层，除了输入层节点，每个节点都是一个带有非线性激活函数(sigmoid/relu(线性整流)/silu); 
   - 定义了一个特殊的编码permutation coding(排列编码);
   - 使用Batch Normalization算法,以进行学习时的mini-batch为单位，按mini-batch进归一化; 可以使学习快速进行（可以增大学习率）;不那么依赖初始值;抑制过拟合（降低Dropout等的必要性）
  ![](https://raw.githubusercontent.com/weedge/mypic/master/llm/llm-knowledge-point-all-u-need/3.jpg)
  

> [!TIP]
> Transformer Decoder-only 模型 (AR GLM)
> 为了使模型能够利用序列的顺序，必须注入一些有关序列中标记的相对或绝对位置的信息。在原始Transformer整体模型结构中， 采用“positional encodings”, 使用正弦版本(sinusoidal version), 因为它可以允许模型推断出比训练期间遇到的序列长度更长的序列长度。这里训练简单起见，使用学习的位置嵌入(learned positional embeddings)来代替。(原论文中提到正弦版本positional encodings和learned positional embeddings,产生几乎相同的结果)； 至于后续相关改进， 比如：RoPE 见[../modules/positional](../modules/positional/)

1. GPT(Generative Pre-trained Transformer) LM: 使用类似GPT2模型，加入位置embedding, block(attention机制, FFN(MLP)前馈层, 以及残差连接)， 以及对输入权重参数进行了初始化(如果初始化为0,反向传播时更新权重变的没有意义;为了防止"权重均一化"（严格地讲，是为了瓦解权重的对称结构），必须随机生成初始值;常采用定义标准差正太分布(高斯分布),这里标准差std=0.02)，
![](https://raw.githubusercontent.com/weedge/baby-llm/main/docs/simple-gpt.drawio.png)

2. layer/block wise scaling GPT LM: 使用类似GPT2模型，使用 layer/block wise scaling (scaling attention qkv heads, ffn intermediate sizes), 详情见:
   - [OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework](https://arxiv.org/abs/2404.14619)
   - [DeLighT: Deep and Light-weight Transformer](https://arxiv.org/abs/2008.00623)

3. MoE(mixture of experts) LM: 稀疏专家混合语言模型(SMoE) 
   - 稀疏专家混合而不是孤立的前馈神经网络。
   - 使用了top-k门控和noisy top-k门控实现。
   - 模型训练初始化 - 这里使用了Kaiming He初始化，
      - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) Kaiming He 
      - [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) Xavier Glorot
      
   ![](https://raw.githubusercontent.com/weedge/baby-llm/main/docs/simple-moe.drawio.png)

4. MoA(SMoE+MultiHeadAttention)-MoE(mixture of experts) LM: 模块化来源于稀疏专家混合语言模型(ModuleFormer) 
   - 引入专家容量（Expert Capacity factor）
   - Load Balancing Loss
   - 相关详情见:[**Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**](https://arxiv.org/abs/2101.03961)
   ![](https://raw.githubusercontent.com/weedge/baby-llm/main/docs/simple-moa-moe.drawio.png)

5. MLA(Multi-Head Latent Attention)-MoE(mixture of experts) LM: 模块化来源于deepseekv2 
   - 从低秩(low rank)投影的角度引入Multi-Head Latent Attention
   - 训练模型，实现中未引入kv caching
   - Load Balancing Loss (auxiliary loss) for MoEs (注：v3 则使用了auxiliary-loss-free load balancing strategy, 详情见： [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664))

## start
```shell
git clone https://github.com/weedge/baby-llm.git
cd baby-llm && make -p {datas,models}
# datas/tinyshakespeare.txt
#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O datas/tinyshakespeare.txt
```

### datasets prepare
prepare: download datasets -> tokenizer --encode--> tokenids (train.bin, val.bin)
```shell
# shakespeare_char
python3 simpleLM/datasets/shakespeare_char/prepare.py
```

### train & generate(sampling; no temperature,top-k)
```shell
# train 超参数可调整
python3 simpleLM/train.py --model_name=bigramLM
python3 simpleLM/train.py --model_name=mlpLM
python3 simpleLM/train.py --model_name=gptLM
python3 simpleLM/train.py --model_name=block_wise_scaling_gptLM
python3 simpleLM/train.py --model_name=moeLM
python3 simpleLM/train.py --model_name=moa_moeLM
python3 simpleLM/train.py --model_name=mla_moeLM # --model_config_file "mla_moe_config.json" model args from model_config_file 

# plot train/validation loss
ls loss_*.log | python3 simpleLM/plot.py 
# tips: 这里没有使用wandb来记录loss, 简单直接通过plot来绘制曲线图
```
附：
- [simpleLM训练笔记](https://colab.research.google.com/drive/1ArSBhdnET4-o6KpX6qP7VXhYrVTg0lKN?usp=sharing)
- [moa_moeLM训练笔记](https://github.com/weedge/doraemon-nb/blob/main/makeMoA_MoE_from_Scratch_with_Expert_Capacity_Aux_Loss_Balance.ipynb)

# 参考
- **https://lena-voita.github.io/nlp_course/language_modeling.html**
- [karpathy/min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086)
- https://en.wikipedia.org/wiki/Activation_function
- https://karpathy.ai/zero-to-hero.html | https://github.com/karpathy/nn-zero-to-hero
- https://github.com/karpathy/ng-video-lecture
- https://github.com/antirez/simple-language-model
- https://github.com/karpathy/makemore
- https://github.com/AviSoori1x/makeMoE/blob/main/makeMoE_from_Scratch.ipynb
- https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L384
- https://github.com/myshell-ai/JetMoE
- https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py
- https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
- https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

# paper
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
- [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/pdf/1808.04444.pdf)
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [**GPT1-Improving Language Understanding by Generative Pre-Training**](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [**Outrageosly Large Neural Networks: The Sparsely-Gated Mixture-Of-Experts layer**](https://arxiv.org/pdf/1701.06538.pdf)
- [**Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts](https://arxiv.org/pdf/2401.04088.pdf)
- [ModuleFormer: Modularity Emerges from Mixture-of-Experts](https://arxiv.org/pdf/2306.04640.pdf)
- [JetMoE: Reaching Llama2 Performance with 0.1M Dollars](https://arxiv.org/pdf/2404.07413.pdf)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)
- [Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models](https://arxiv.org/pdf/2404.05567)
- [**DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**](https://arxiv.org/pdf/2405.04434)(建模训练,模型结构优化,MAL+MoE)
- [**DeepSeek-V3 Technical Report**](https://arxiv.org/pdf/2412.19437)(模型结构优化：MoE with Auxiliary-Loss-Free Load Balancing 以及 Multi-Token Prediction; 训练推理工程优化：分布式训练，最大利用显存,零气泡(DualPipe),引入FP8进行混合精度训练； 分布式部署推理，将prefilling 和 decoding 拆分（和kimi mooncake类似），prilling 中的并行化：ttention(TP4,SP,DP8), MoE(EP32)/MLP(TP1),高负载的专家并进行冗余部署(10分钟检查监控统计数据检查是否高负载，进行扩容); decoding中的并行化： Attention(TP4,SP,DP80), MoE(EP320),结合硬件进行量化，算子融合操作(fused operation),利用局部性原理，提高吞吐)
