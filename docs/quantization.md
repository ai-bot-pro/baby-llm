# Quantization


## Transformers quantization with bitsandbytes

BitsAndBytesConfig:

https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
```
BitsAndBytesConfig {
  "_load_in_4bit": true,
  "_load_in_8bit": false,
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_quant_storage": "uint8",
  "bnb_4bit_quant_type": "nf4",
  "bnb_4bit_use_double_quant": true,
  "llm_int8_enable_fp32_cpu_offload": false,
  "llm_int8_has_fp16_weight": false,
  "llm_int8_skip_modules": null,
  "llm_int8_threshold": 6.0,
  "load_in_4bit": true,
  "load_in_8bit": false,
  "quant_method": "bitsandbytes"
}

```
这是一个名为 `BitsAndBytesConfig` 的配置，用于控制模型量化（quantization）过程中的一些选项。下面是每个参数的含义解释：
- `load_in_8bit`（bool，可选，默认为False）：启用8位量化的标志。使用LLM.int8()。
- `load_in_4bit`（bool，可选，默认为False）：启用4位量化的标志。通过用来自 bitsandbytes 的 FP4/NF4 层替换线性层实现。
- `llm_int8_threshold`（float，可选，默认为6.0）：对于异常值检测的离群值阈值。任何超过此阈值的隐藏状态值将被视为离群值，并对这些值进行fp16操作。通常情况下，值是正态分布的，即大多数值在范围[-3.5，3.5]内，但对于大型模型，有一些异常的系统离群值的分布可能会非常不同。这些离群值通常在区间[-60，-6]或[6，60]内。Int8量化对于幅度约为5的值效果很好，但超过该值会有显著的性能惩罚。一个良好的默认阈值是6，但对于更不稳定的模型（小型模型、微调），可能需要更低的阈值。
- `llm_int8_skip_modules`（List[str]，可选）：我们不希望转换为8位的模块的显式列表。这对于像Jukebox这样的模型非常有用，该模型在不同位置具有几个头部而不一定在最后位置。例如，对于CausalLM模型，最后的lm_head保持其原始数据类型。
- `llm_int8_enable_fp32_cpu_offload`（bool，可选，默认为False）：用于高级用例和知道此功能的用户。如果您想要将模型拆分为不同的部分，并在GPU上以int8运行某些部分并在CPU上以fp32运行某些部分，则可以使用此标志。这对于卸载诸如google/flan-t5-xxl之类的大型模型非常有用。请注意，int8操作不会在CPU上运行。
- `llm_int8_has_fp16_weight`（bool，可选，默认为False）：使用16位主要权重运行LLM.int8()。对于微调很有用，因为权重不必在反向传播过程中反复转换。
- `bnb_4bit_compute_dtype`（torch.dtype或str，可选，默认为torch.float32）：设置计算类型，可能与输入类型不同。例如，输入可能是fp32，但计算可以设置为bf16以提高速度。
- `bnb_4bit_quant_type`（str，可选，默认为"fp4"）：设置bnb.nn.Linear4Bit层中的量化数据类型。选项是FP4和NF4数据类型，由fp4或nf4指定。
- `bnb_4bit_use_double_quant`（bool，可选，默认为False）：用于嵌套量化，其中来自第一次量化的量化常数再次量化。
- `bnb_4bit_quant_storage`（torch.dtype或str，可选，默认为torch.uint8）：设置存储类型以打包量化的4位参数。
- `kwargs`（Dict[str, Any]，可选）：用于初始化配置对象的其他参数。

这是一个关于通过 `bitsandbytes` 加载的模型中所有可能属性和特性的包装类的说明。这替换了 `load_in_8bit` 或 `load_in_4bit`，因此这两个选项是互斥的。

目前仅支持 `LLM.int8()`、`FP4` 和 `NF4` 量化。如果将更多方法添加到 `bitsandbytes` 中，则将添加更多参数到此类中。

这些参数控制着量化过程中的各种设置，如量化类型、数据类型、阈值等，以及加载模型时使用的位数。量化是一种优化技术，可以将神经网络模型中的参数和激活值从高精度（如32位浮点数）转换为低精度（如8位整数），从而减少模型的存储空间和计算量，提高模型的效率和速度。