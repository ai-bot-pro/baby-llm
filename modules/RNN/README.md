# intro
seq model with hidden states
> [!TIP]
> It is noteworthy that hidden layers and hidden states refer to two very different concepts. 
> - **Hidden layers** are, as explained, layers that are hidden from view on the path from input to output. 
> - **Hidden states** are technically speaking inputs to whatever we do at a given step, and they can only be computed by looking at data at previous time steps. 


## RNN (Recurrent Neural Networks)
- [Recurrent neural network based language model](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- https://gist.github.com/karpathy/d4dee566867f8291f086 
- https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html
- **https://karpathy.github.io/2015/05/21/rnn-effectiveness/**(view rnn history -> attention(transformer))
- [**Recurrent Models of Visual Attention**](https://arxiv.org/abs/1406.6247) (RNN+attention)

## GRU (Gated Recurrent Unit)
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259) (GRU)
- https://zh.d2l.ai/chapter_recurrent-modern/gru.html

## LSTM (Long Short-Term Memory)
- https://en.wikipedia.org/wiki/Long_short-term_memory
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) (LSTM)
- https://zh.d2l.ai/chapter_recurrent-modern/lstm.html

# src view
## RNN
- https://github.com/pytorch/pytorch/blob/main/torch/backends/cudnn/rnn.py
- https://developer.nvidia.com/cudnn dnn kenerl
- https://docs.nvidia.com/deeplearning/cudnn/latest/api/overview.html (cudnn be) shim layer(libcudnn.so) -> (libcudnn_adv.so)
- https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-adv-library.html#cudnnrnnalgo-t
- https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/rnn.cpp#L400

> [!TODO] maybe impl a simple rnn.c (just learn cuda->cudnn) like [llm.c](https://github.com/karpathy/llm.c) attention,mlp