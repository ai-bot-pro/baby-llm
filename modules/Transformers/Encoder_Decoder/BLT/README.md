# BLT (Byte Latent Transformer)
- 将视觉(vision)领域的patches引入Transformer中，无需单独的tokenizer操作；(Patches Scale Better Than Tokens)
- 模型结构:
  - Local Encoder: a small local model that encode sequences of bytes into patches 
  - Latent Global Transformer Model: a large global autoregressive language model that operates on patch representations
  - Local Decoder: a small local model that decode patch representations back into bytes
  - Local Encoder使用交叉注意力块(patch cross-attention block)，将patch作为查询，将byte作为键/值，将byte编码为patch。Local Decoder使用类似的块，但角色相反，即byte表示现在是查询，patch表示是键/值。

<img width="785" alt="image" src="https://github.com/user-attachments/assets/18af4c93-2202-4e6c-a313-f1a5c5be4636" />


# reference
- [2024. Byte Latent Transformer: Patches Scale Better Than Tokens](https://dl.fbaipublicfiles.com/blt/BLT__Patches_Scale_Better_Than_Tokens.pdf)
- https://github.com/facebookresearch/blt
- [2021. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/pdf/2103.14899)
- https://github.com/IBM/CrossViT