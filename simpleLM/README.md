# Simple Language Model
简单的生成式语言模型(GLM), 用于了解GLM训练推理过程
## datasets
- tinyshakespeare.txt
```shell
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O datas/tinyshakespeare.txt
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
1. Bigram LM: 一个包含vocab_size的vocab_size张量的嵌入权重层模块(nn.Embedding)，用来训练学习类似二元语言模型token的概率 $P(t_i\|t_{i-1})$权重参数

2. MLP(Multilayer Perceptron) LM: 每层为全连接线性权重层;  
   - 该模型定义了一个输入层，三个隐藏层(每层维度逐层指数减少)，一个输出层，除了输入层节点，每个节点都是一个带有非线性激活函数(sigmoid/relu(线性整流)/silu); 
   - 定义了一个特殊的编码permutation coding(排列编码);
   - 使用Batch Normalization算法,以进行学习时的mini-batch为单位，按mini-batch进行正规化; 可以使学习快速进行（可以增大学习率）;不那么依赖初始值（对于初始值不用那么神经质）;抑制过拟合（降低Dropout等的必要性）
  ![](https://raw.githubusercontent.com/weedge/mypic/master/llm/llm-knowledge-point-all-u-need/3.jpg)
  

3. GPT(Generative Pre-trained Transformer) LM: 使用类似GPT2模型，加入位置embedding, block(attention机制, FFN(MLP)前馈层, 以及残差连接)， 以及对输入权重参数进行了初始化(如果初始化为0,反向传播时更新权重变的没有意义;为了防止"权重均一化"（严格地讲，是为了瓦解权重的对称结构），必须随机生成初始值;常采用定义标准差正太分布(高斯分布),这里标准差std=0.02)，


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

### train & generate
```shell
# train
python3 simpleLM/train.py --model_name=bigramLM
python3 simpleLM/train.py --model_name=mlpLM
python3 simpleLM/train.py --model_name=gptLM

# plot train/validation loss
ls loss_*.txt | python3 simpleLM/plot.py 
```
附：[simpleLM训练笔记](https://github.com/weedge/doraemon-nb/blob/main/simple_lm.ipynb)


# 参考
- **https://lena-voita.github.io/nlp_course/language_modeling.html**
- https://en.wikipedia.org/wiki/Activation_function
- https://karpathy.ai/zero-to-hero.html
- https://github.com/karpathy/ng-video-lecture
- https://github.com/antirez/simple-language-model
- https://www.youtube.com/watch?v=EXbgUXt8fFU

# paper
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
- [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/pdf/1808.04444.pdf)
