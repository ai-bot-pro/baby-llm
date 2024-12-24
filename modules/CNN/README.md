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


## CNN 局限
- 基于的假设是局部信息相互依赖 (常用于计算机视觉任务), 不能用于长距离的关联依赖
- CNN 具有 Hierarchical Receptive Field，使得任意任意两个位置之间的长度距离是对数级别的