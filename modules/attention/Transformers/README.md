# Transformer


## self-attention
在深度学习中，经常使用卷积神经网络（CNN）或循环神经网络（RNN）对序列进行编码。 想象一下，有了注意力机制之后，我们将词元序列输入注意力池化中， 以便同一组词元同时充当查询、键和值。 具体来说，每个查询都会关注所有的键－值对并生成一个注意力输出。 由于查询、键和值来自同一组输入，因此被称为 自注意力（self-attention), 也被称为内部注意力（intra-attention）。 

## positional-encoding

在处理token序列时，循环神经网络是逐个的重复地处理token， 而自注意力则因为并行计算而放弃了顺序操作。 为了使用序列的顺序信息，通过在输入表示中添加 位置编码（positional encoding）来注入绝对的或相对的位置信息。 位置编码可以通过学习得到也可以直接固定得到。 


## Model Architecture
### Positionwise Feed-Forward Networks
### Residual Connection and Layer Normalization
### Encoder
### Decoder

### encoder-decoder

### encoder-only

### decoder-only


# reference
- 
- https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html