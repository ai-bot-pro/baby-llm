# intro
encoder-decoder for seq2seq with attention (e.g.: NMT(Neural Machine Translation),TTS,text2image,image2image...etc)

# reference
- https://en.wikipedia.org/wiki/Attention_(machine_learning)

## RNN + attention
- [2014. Recurrent models of visual attention](https://arxiv.org/abs/1406.6247)(RNN + attetion(non-differentiable(hard), use RL to learn task-specific policies))
- [2014. Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v7)(RNN(LSTM)+Additive Attention(Bahdanau Attention) in NMT)
- [2015. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) (RNN(LSTM)+(hard/soft)attetion)
- [2015. Attention-Based Models for Speech Recognition](https://arxiv.org/abs/1506.07503) (RNN(LSTM/GRU) + Location Sensitive Attention, extends the additive attention)
- [2015. Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)(RNN(LSTM)+(global/local)multiplicative attention(Luong Attention))
- [2017. A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)(LSTM + self-attention)

>LSTM内部有Gate机制，其中input gate选择哪些当前信息进行输入，forget gate选择遗忘哪些过去信息，算是一定程度的Attention，而且号称可以解决长期依赖问题，实际上LSTM需要一步一步去捕捉序列信息，在长文本上的表现是会随着step增加而慢慢衰减，难以保留全部的有用信息。
>
>LSTM通常需要得到一个向量，再去做任务，常用方式有：
>
> - 直接使用最后的hidden state（可能会损失一定的前文信息，难以表达全文）
> - 对所有step下的hidden state进行等权平均（对所有step一视同仁）。
> - Attention机制，对所有step的hidden state进行加权，把注意力集中到整段文本中比较重要的hidden state信息。性能比前面两种要好一点，而方便可视化观察哪些step是重要的，但是要小心过拟合，而且也增加了计算量。


## CNN + attention
- [2015. ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/abs/1512.05193) (Attention-Based CNN: ABCNN-1(attention before Convolution), ABCNN-2(attention after Convolution), ABCNN-3(combines ABCNN-1 and ABCNN-2))
- [2016. WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (WaveNet TTS, CNN with attention)
- [2016. Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099) (ByteNet in char-level NMT encoder-decoder arch, CNN with attention)
- [2017. Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825) (see Appendix WaveNet detail arch)
- [2017. Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) (ConvS2S in NMT with Multi-step Attention in decoder, CNN with attention)

>CNN的卷积操作可以提取重要特征，也算是Attention的思想，但是CNN的卷积感受视野是局部的，需要通过叠加多层卷积区去扩大视野。另外，Max Pooling直接提取数值最大的特征，也像是hard attention的思想，直接选中某个特征。
>
>CNN上加Attention可以加在这几方面：
>
> - 在卷积操作前做attention，比如Attention-Based CNN-1，这个任务是文本蕴含任务需要处理两段文本，同时对两段输入的序列向量进行attention，计算出特征向量，再拼接到原始向量中，作为卷积层的输入。
> - 在卷积操作后做attention，比如Attention-Based CNN-2，对两段文本的卷积层的输出做attention，作为pooling层的输入。
> - 在pooling层做attention，代替max pooling。比如Attention pooling，首先我们用LSTM学到一个比较好的句向量，作为query，然后用CNN先学习到一个特征矩阵作为key，再用query对key产生权重，进行attention，得到最后的句向量。


## all attention
- ❤[2017. **Attention Is All You Need**](https://arxiv.org/abs/1706.03762)❤ (Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. **self attention(Scaled Dot-Product Attention(SDPA), Multi-Head Attention)**)
- [Tensor2Tensor Transformers New Deep Models for NLP](https://nlp.stanford.edu/seminar/details/lkaiser.pdf) (from RNN CNN with attention (e.g.: WaveNet,ByteNet) -> transformers all attention)
- ⭐⭐[Understanding and Coding Self-Attention, Multi-Head Attention, Cross-Attention, and Causal-Attention in LLMs](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)


------
- [illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- ⭐⭐[**Transformers from Scratch**](https://e2eml.school/transformers.html) 
- ⭐⭐[**The Annotated Transformer**](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- ⭐⭐️[**Attention? Attention!**](https://lilianweng.github.io/posts/2018-06-24-attention/)
- ⭐[3blue1brown: Visualizing transformers and attention](https://www.youtube.com/watch?v=KJtZARuO3JY&t=2124s)

>Self Attention与传统的Attention机制非常的不同：
> - 传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。
> - 但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。
> - 因此，self Attention Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系，相对比，self Attention不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系