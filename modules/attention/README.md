# intro
encoder-decoder for seq2seq with attention (e.g.: NMT(Neural Machine Translation),TTS,text2image,image2image...etc)

# reference
- https://en.wikipedia.org/wiki/Attention_(machine_learning)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)(RNN(LSTM)+Additive Attention(Bahdanau Attention) in NMT)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) (RNN(LSTM)+(hard/soft)attetion)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)(RNN(LSTM)+(global/local)multiplicative attention(Luong Attention))
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (WaveNet TTS, CNN with attention)
- [Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825) (see Appendix WaveNet detail arch)
- [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099) (ByteNet in char-level NMT encoder-decoder arch, CNN with attention)
- [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) (ConvS2S in NMT with Multi-step Attention in decoder, CNN with attention)
- ❤[**Attention Is All You Need**](https://arxiv.org/abs/1706.03762)❤ (Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. **self attention(Scaled Dot-Product Attention(SDPA), Multi-Head Attention)**)
- [Tensor2Tensor Transformers New Deep Models for NLP](https://nlp.stanford.edu/seminar/details/lkaiser.pdf) (from RNN CNN with attention (e.g.: WaveNet,ByteNet) -> transformers all attention)


------
- [illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- ⭐⭐️[**Attention? Attention!**](https://lilianweng.github.io/posts/2018-06-24-attention/)
- ⭐[**Transformers from Scratch**](https://e2eml.school/transformers.html)
- ⭐[3blue1brown: Visualizing transformers and attention](https://www.youtube.com/watch?v=KJtZARuO3JY&t=2124s)

>Self Attention与传统的Attention机制非常的不同：传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。因此，self Attention Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系，相对比，self Attention可以不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系