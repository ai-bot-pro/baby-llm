# reference

## CNN (Convolutional Neural Networks)
- [**A guide to convolution arithmetic for deep learning**](https://ar5iv.labs.arxiv.org/html/1603.07285)
- https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html (lenet)
- https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html (deep) [**cuda-convnet2 code**](https://github.com/akrizhevsky/cuda-convnet2)
- https://zh.d2l.ai/chapter_convolutional-modern/vgg.html (block)
- https://zh.d2l.ai/chapter_convolutional-modern/nin.html (mlp(linear->1x1 kernel conv2d))
- https://zh.d2l.ai/chapter_convolutional-modern/googlenet.html (Inception block filter,go to deeper)
- https://zh.d2l.ai/chapter_convolutional-modern/resnet.html (residual block)
------
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (WaveNet TTS)
- [Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825) (see Appendix WaveNet detail arch)
- [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099) (BitNet in char-level NMT encoder-decoder arch)
- [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) (ConvS2S in NMT with Multi-step Attention in decoder)
- [2022. A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (对ResNet-200进行改进，按照ViT(encoder)变体 Swin Transformer设计，替换MSA(multiheaded self-attention)-> 7x7 conv2d; MLP(linear->1x1 kernel conv2d,active function ReLU->GeLU), BN->LN, 纯 ConvNet 模型结构, 性能和Swin transformer相当， 但是模型结构轻巧，推理更快； 但是ConvNeXt 可能更适合某些任务，比如： 图像分类、对象检测、实例和语义分割任务； 而 Transformers 对于其他任务可能更灵活，泛化能力强，当用于需要离散、稀疏或结构化输出的任务时，Transformer 可能会更加灵活。所以架构选择应该满足手头任务的需求，同时力求简单。)


## CNN 局限
- 基于的假设是局部信息相互依赖 (常用于计算机视觉任务), 不能用于长距离的关联依赖
- CNN 具有 Hierarchical Receptive Field，使得任意任意两个位置之间的长度距离是对数级别的