# reference
- [2017. Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) (Absolute position encoding. ssee 5.4. Position Embeddings(train to learn), less effect)
- [2020. How Much Position Information Do Convolutional Neural Networks Encode](https://arxiv.org/abs/2001.08248)(Zero-Padding Driven Position Information: 提取的是当前位置与padding的边界的相对距离,so less effect with position embedding)
- [2017. **Attention Is All You Need**](https://arxiv.org/abs/1706.03762) (Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. self attention(Scaled Dot-Product Attention(SDPA), Multi-Head Attention)) (**fixed positional encoding based on sine and cosine functions (Sinusoidal Positional embeddings, Absolute position encoding)**)
- [2018. Self-attention with relative position representations](https://arxiv.org/abs/1803.02155) (Relative position encoding)
- [2019. Encoding word order in complex embeddings](https://arxiv.org/abs/1912.12333)(Complex embedding) [complex-order](https://github.com/FreedomIntelligence/complex-order)
- [2021. **RoFormer: Enhanced Transformer with Rotary Position Embedding**](https://arxiv.org/abs/2104.09864) (**RoPE** fusing Absolute and Relative position encoding)

------
- ⭐️ https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
- https://blog.eleuther.ai/rotary-embeddings/
- [苏剑林: 让研究人员绞尽脑汁的Transformer位置编码](https://spaces.ac.cn/archives/8130)
- [**苏剑林: Transformer升级之路：1、Sinusoidal位置编码追根溯源**](https://spaces.ac.cn/archives/8231)
- [苏剑林: Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)
- https://spaces.ac.cn/tag/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/
- https://huggingface.co/blog/designing-positional-encoding