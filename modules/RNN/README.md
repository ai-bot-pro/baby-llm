# intro
PR: [Feat: RNN (recurrent neural network) sequence model with hidden states, have time steps.](https://github.com/ai-bot-pro/baby-llm/pull/2)

seq model with hidden states
> [!TIP]
> It is noteworthy that hidden layers and hidden states refer to two very different concepts. 
> - **Hidden layers** are, as explained, layers that are hidden from view on the path from input to output. 
> - **Hidden states** are technically speaking inputs to whatever we do at a given step, and they can only be computed by looking at data at previous time steps. 


## RNN (Recurrent Neural Networks)
- [Recurrent neural network based language model](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- https://gist.github.com/karpathy/d4dee566867f8291f086 
- https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html
- **https://karpathy.github.io/2015/05/21/rnn-effectiveness/**

## GRU (Gated Recurrent Unit)
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259) (GRU)
- https://zh.d2l.ai/chapter_recurrent-modern/gru.html

## LSTM (Long Short-Term Memory)
- https://en.wikipedia.org/wiki/Long_short-term_memory
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) (LSTM)
- https://zh.d2l.ai/chapter_recurrent-modern/lstm.html
- [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) (RNN(LSTM)) [code](https://github.com/kevinzakka/recurrent-visual-attention)

# src view
## RNN
- https://github.com/pytorch/pytorch/blob/main/torch/backends/cudnn/rnn.py
- https://developer.nvidia.com/cudnn dnn kenerl
- https://docs.nvidia.com/deeplearning/cudnn/latest/api/overview.html (cudnn be) shim layer(libcudnn.so) -> (libcudnn_adv.so)
- https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-adv-library.html#cudnnrnnalgo-t
- https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/rnn.cpp#L400

> [!TODO] maybe impl a simple rnn.c with gate (just learn cuda->cudnn) like [llm.c](https://github.com/karpathy/llm.c) attention,mlp


## RNN局限
- RNN 的长距离依赖比较 tricky：RNN 很强大（可以作为 encoder 对长度任意的序列进行特征抽取，基于特征抽取的能力可以胜任分类任务, 另一方面可以作为Generative Language Model 生成序列），其实核心就是长距离依赖（gate(门控) architectures - 线性操作让信息(hidden states)可以保持并流动，并选择性地让信息通过），可以对长度任意的序列进行表达，但是这种方式还是比较 tricky。并且这种序列建模方式，无法对具有层次结构的信息进行很好的表达, 更深层的神经网络进行特征学习。
- RNN 由于递归的本质，导致无法并行(当前时刻的状态依赖上一时刻的状态,串行执行)。


so 引入attention机制，rnn+attention -> all attention (transformers) -> attention + gate (MoE,MoA-MoE) (具体实现可以看[simpleLM](https://github.com/ai-bot-pro/baby-llm/tree/main/simpleLM)中的实现~)