## Supervised Fine-Tuning

### Full Fine-Tuning (FFT)
![full-fine-tuning](https://raw.githubusercontent.com/weedge/mypic/master/llm/llm-sft/full-fine-tuning.png)
Adjusts all parameters of the LLM using task-specific data.
like pre-training, but resume ckpt.pt and use like prompt-text labels supervised datasets with tokenizer(sp bpe)


#### inference
juse like pre-training base model to inference

#### Notebooks



### Parameter-efficient Fine-Tuning (PEFT)
![PEFT](https://raw.githubusercontent.com/weedge/mypic/master/llm/llm-sft/PEFT.png)
Modifies select parameters for more efficient adaptation. for base pre-training parameters model (large params)
eg: LoRA; Prefix-tuning P-tuning v2, P-tuning; IA3
see hf peft: https://huggingface.co/docs/peft/conceptual_guides/adapter 
- LoRA (generate task，finetune embedding/attention,ffn layer/lm_head)
```python
config = LoraConfig(
    r=16, lora_alpha=128, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
)
```
```
`LoraConfig` 的配置，用于配置 LoRA（Low-Rank Adaptation）的相关选项。LoRA 是一种低秩适应技术，用于减少大型语言模型的参数量，并提高模型的效率和性能。以下是各个参数的含义解释：

1. `peft_type`: PEFT（Parameter-Efficient Fine-Tuning）类型。在这里，设置为 `PeftType.LORA`，表示使用 LoRA 进行微调。

2. `auto_mapping`: 自动映射。用于指定是否自动映射预训练模型的参数以适应新任务。

3. `base_model_name_or_path`: 基础模型名称或路径。用于指定要微调的基础语言模型的名称或路径。

4. `revision`: 模型修订版本。

5. `task_type`: 任务类型。在这里，未指定任务类型。

6. `inference_mode`: 推理模式。用于指定是否在推理时使用 LoRA。

7. `r`: LoRA 中的秩（rank）参数。用于控制低秩适应的参数量。

8. `target_modules`: 目标模块。指定了要应用低秩适应的模块名称集合。

9. `lora_alpha`: LoRA 中的 alpha 参数。用于控制低秩适应的强度。

10. `lora_dropout`: LoRA 中的 dropout 参数。用于控制低秩适应的正则化。

11. `fan_in_fan_out`: 是否使用扇入/扇出初始化。

12. `bias`: 偏置类型。指定了在低秩适应中使用的偏置类型。

13. `use_rslora`: 是否使用 RSLora。

14. `modules_to_save`: 要保存的模块列表。

15. `init_lora_weights`: 是否初始化 LoRA 权重。

16. `layers_to_transform`: 要转换的层列表。

17. `layers_pattern`: 层模式。

18. `rank_pattern`: 秩模式。

19. `alpha_pattern`: Alpha 模式。

20. `megatron_config`: Megatron 配置。

21. `megatron_core`: Megatron 核心。

22. `loftq_config`: LoFTQ 配置。

23. `use_dora`: 是否使用 DoRA。

24. `layer_replication`: 层复制。

这些参数用于配置 LoRA 过程中的各个方面，包括秩参数、模块选择、初始化方法等。通过调整这些参数，可以控制 LoRA 的行为，以满足特定任务和硬件环境的需求。
```

- Soft prompts (NLP/NLU sub task eg: classify, just prompt embedding to model)
```python
# 配置的每层都添加soft token
peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
```
```
 PrefixTuningConfig 的配置对象，用于配置 Prefix Tuning（前缀调整）模型的各种参数。以下是各个参数的含义解释：

peft_type: Prefix Tuning 模型类型，这里设定为 PREFIX_TUNING。

auto_mapping: 自动映射配置，用于指定是否使用自动映射。

base_model_name_or_path: 基础模型的名称或路径。

revision: 模型的修订版本。

task_type: 任务类型，这里设定为 CAUSAL_LM（因果语言模型）。

inference_mode: 推断模式，指示模型是否处于推断模式。

num_virtual_tokens: 虚拟token数量，用于指定前缀调整模型的虚拟token数量。

token_dim: token维度，用于指定token的维度大小。

num_transformer_submodules: Transformer 子模块数量。

num_attention_heads: 注意力头数量。

num_layers: 模型的层数。

encoder_hidden_size: 编码器隐藏层大小。

prefix_projection: 前缀投影，指示是否使用前缀投影。

这些参数可以根据具体任务和模型需求进行调整，以配置和定制 Prefix Tuning 模型，以适应特定的应用场景和任务要求。
```
------
```python
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)
```
```
`PromptTuningConfig` 的配置，用于配置 Prompt Tuning（PT）的相关选项。Prompt Tuning 是一种通过微调预训练语言模型来实现零样本或小样本学习的技术，它通过添加自定义的提示（prompt）来指导模型进行特定任务的学习。以下是各个参数的含义解释：

1. `peft_type`: PEFT（Parameter-Efficient Fine-Tuning）类型。在这里，设置为 `PeftType.PROMPT_TUNING`，表示使用 Prompt Tuning 进行微调。

2. `auto_mapping`: 自动映射。用于指定是否自动映射预训练模型的参数以适应新任务。

3. `base_model_name_or_path`: 基础模型名称或路径。用于指定要微调的基础语言模型的名称或路径。

4. `revision`: 模型修订版本。

5. `task_type`: 任务类型。在这里，设置为 `TaskType.CAUSAL_LM`，表示任务类型为因果语言建模（Causal Language Modeling）。

6. `inference_mode`: 推理模式。用于指定是否在推理时使用 Prompt Tuning。

7. `num_virtual_tokens`: 虚拟token数量。用于指定在 PT 中使用的虚拟token数量。

8. `token_dim`: token维度。用于指定 PT 中标记的维度。

9. `num_transformer_submodules`: Transformer 子模块数量。

10. `num_attention_heads`: 注意力头数量。

11. `num_layers`: 层数量。

12. `prompt_tuning_init`: Prompt Tuning 初始化方法。在这里，设置为 `PromptTuningInit.TEXT`，表示使用文本作为初始化。

13. `prompt_tuning_init_text`: Prompt Tuning 初始化文本。指定了用于初始化的文本提示。

14. `tokenizer_name_or_path`: 分词器名称或路径。用于指定分词器的名称或路径。

15. `tokenizer_kwargs`: 分词器参数。用于指定分词器的其他参数，如词汇表大小、特殊标记等。

这些参数用于配置 Prompt Tuning 过程中的各个方面，包括任务类型、模型初始化、分词器设置等。通过调整这些参数，可以根据具体的任务和需求来定制 Prompt Tuning 的行为。
```
------

- IA3(Infused Adapter by Inhibiting and Amplifying Inner Activations) (seq2seq task, embedding/attention layer/lm_head and ffn layer)
与LoRA类似，IA3具有许多相同的优势：
  - IA3通过大幅减少可训练参数的数量使微调更加高效。（对于T0模型，一个IA3模型只有大约0.01%的可训练参数，而LoRA甚至超过0.1%）
  - 原始的预训练权重保持冻结，这意味着您可以在其基础上为各种下游任务构建多个轻量级和便携式的IA3模型。
  - 使用IA3微调的模型的性能与完全微调的模型的性能相当。
  - IA3不会增加任何推理延迟，因为适配器权重可以与基础模型合并。

  原则上，IA3可以应用于神经网络中的任何子集权重矩阵，以减少可训练参数的数量。根据作者的实现，IA3权重被添加到变换器模型的关键、值和前馈层。
具体来说，对于变换器模型，IA3权重被添加到每个变换器块的键和值层的输出，以及第二个前馈层的输入。
  鉴于目标层注入IA3参数，可以根据权重矩阵的大小确定可训练参数的数量。
```
peft_config = IA3Config(
    task_type=TaskType.SEQ_CLS, target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"]
)
```
```
IA3Config 的参数含义解释：

peft_type: IA3 模型类型，这里设定为 IA3。

auto_mapping: 自动映射配置，用于指定是否使用自动映射。

base_model_name_or_path: 基础模型的名称或路径。

revision: 模型的修订版本。

task_type: 任务类型，这里设定为 SEQ_CLS（序列分类）。

inference_mode: 推断模式，指示模型是否处于推断模式。

target_modules: 目标模块集合，指示 IA3 模型需要转换的目标模块。

feedforward_modules：作为target_modules中的前馈层处理的模块列表。虽然学习到的向量与注意力块的输出激活相乘，但向量与经典前馈层的输入相乘。请注意，feedforward_modules必须是target_modules的子集。

fan_in_fan_out: 是否启用 IA3 中的 fan-in 和 fan-out 参数。

modules_to_save: 需要保存的模块集合。

init_ia3_weights: 是否初始化 IA3 模型的权重。
```

#### inference
use PeftModel.from_pretrained() to load `pre-trained base model` and `adapter model`; eg:
```
inference_model = PeftModel.from_pretrained(model_from_pretrained, "weege007/sft_peft_lora_gemma_2b_with_added_tokens")
```

#### Notebooks

- [sft_peft_lora_gemma_2b_with_additional_tokens.ipynb](https://github.com/weedge/doraemon-nb/blob/main/sft_peft_lora_gemma_2b_with_additional_tokens.ipynb)
- [sft_peft_prompt_tuning_bloomz-560m.ipynb](https://github.com/weedge/doraemon-nb/blob/main/sft-peft-prompt_tuning_bloomz-560m.ipynb)


## reference
- https://aws.amazon.com/cn/compare/the-difference-between-machine-learning-supervised-and-unsupervised/
- https://huggingface.co/PEFT (paper, example_code, docs)
- https://huggingface.co/collections/PEFT/notebooks-6573b28b33e5a4bf5b157fc1