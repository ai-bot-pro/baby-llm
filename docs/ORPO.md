## ORPO

论文: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/pdf/2403.07691.pdf)

这篇论文介绍了一种新的偏好对齐算法，名为Odds Ratio Preference Optimization (ORPO)，用于优化语言模型。以下是对论文内容的总结：

1. **背景**：尽管现有的偏好对齐算法在减少有害输出和提高模型性能方面取得了进展，但它们通常需要一个多阶段过程，包括使用监督式微调（Supervised Fine-Tuning, SFT）作为预热阶段和参考模型。

2. **ORPO算法**：提出了一种无需参考模型的单一步骤偏好对齐方法。ORPO通过在SFT的损失函数中加入一个基于赔率比（Odds Ratio）的惩罚项，有效地区分了优选和非优选的生成风格。

3. **理论基础**：论文强调了SFT在偏好对齐中的作用，并指出即使是轻微的对非优选生成风格的惩罚也足以实现偏好对齐的SFT。

4. **实验结果**：通过在不同大小的模型上进行实验，证明了ORPO的有效性。特别是，使用ORPO微调的Phi-2 (2.7B)、Llama-2 (7B) 和 Mistral (7B) 模型在AlpacaEval2.0、IFEval和MT-Bench等基准测试中超过了参数量更大的现有最先进模型。

5. **代码和模型发布**：作者发布了Mistral-ORPOα (7B) 和 Mistral-ORPO-β (7B) 的训练代码和模型检查点，以便其他研究人员可以复现结果。

6. **相关工作**：论文还讨论了与ORPO相关的其他偏好对齐技术，包括基于强化学习的人类反馈（RLHF）和直接偏好优化（DPO）等。

7. **算法比较**：通过与RLHF和DPO等现有方法的比较，展示了ORPO在效率和性能方面的优势。

8. **局限性和未来工作**：论文指出了ORPO的局限性，包括尚未与其他更广泛的偏好对齐算法进行比较，以及尚未在超过7B参数的模型上进行扩展。未来的工作将包括更广泛的比较、在不同领域和质量的数据集上进行微调，以及更深入地研究偏好对齐过程对预训练语言模型的影响。

这篇论文的主要贡献是提出了一种新的、计算效率高且无需额外参考模型的偏好对齐方法，并通过一系列实验展示了其有效性。

-----------------

ORPO（Odds Ratio Preference Optimization）算法是一种用于优化语言模型偏好对齐的算法。以下是关于ORPO算法的详细信息：

1. **问题背景**：在自然语言处理（NLP）中，预训练语言模型（PLMs）虽然在多种任务上表现出色，但通常需要进一步微调以适应特定应用领域。这个过程通常涉及指令调整和偏好对齐，以确保模型生成的内容符合人类的价值观和期望。

2. **现有方法的局限性**：传统的偏好对齐方法，如强化学习与人类反馈（RLHF）和直接偏好优化（DPO），通常包括多个阶段，需要额外的参考模型和预热阶段。

3. **ORPO的核心思想**：ORPO算法提出了一种无需参考模型的单一步骤偏好对齐方法。它通过在标准的负对数似然（Negative Log-Likelihood, NLL）损失函数中加入一个基于赔率比的惩罚项，来区分优选（favored）和非优选（disfavored）的响应。

4. **ORPO的目标函数**：ORPO的目标函数由两部分组成：
   - **监督式微调（SFT）损失**：最大化生成参考标记的似然性。
   - **相对比率损失（LOR）**：通过最大化优选响应和非优选响应之间的赔率比，来最小化非优选响应的生成概率。

5. **赔率比的定义**：给定输入序列 $x$，优选响应 $y_w$ 与非优选响应 $y_l$ 之间的赔率比 $OR_{\theta}(y_w, y_l)$ 定义为：
   $$ OR_{\theta}(y_w, y_l) = \frac{P_{\theta}(y_w|x)}{1 - P_{\theta}(y_w|x)} \cdot \frac{1 - P_{\theta}(y_l|x)}{P_{\theta}(y_l|x)} $$

6. **梯度计算**：ORPO算法的梯度由两部分组成，一部分惩罚错误预测，另一部分对比优选和非优选响应，以此来调整模型参数。

7. **实验设置**：作者在不同大小的模型上进行了实验，包括从125M到7B参数的模型，并使用了两个数据集：Anthropic的HH-RLHF和二元化的UltraFeedback。

8. **实验结果**：ORPO在多个基准测试中取得了优异的性能，包括AlpacaEval2.0、IFEval和MT-Bench。特别是，使用ORPO微调的Mistral-7B模型在这些基准测试中超过了参数量更大的现有模型。

9. **代码和模型发布**：为了促进研究和复现结果，作者公开了ORPO的代码和预训练模型检查点。

10. **算法优势**：与RLHF和DPO相比，ORPO在内存分配和每批的浮点运算（FLOPs）上更为高效，因为它不需要额外的参考模型。

11. **未来工作**：尽管ORPO在实验中表现出了良好的性能，但作者指出了其局限性，并提出了未来工作的方向，包括与更多偏好对齐算法的比较、扩展到超过7B参数的模型、在多样化的数据集上验证泛化能力，以及深入研究偏好对齐过程对预训练语言模型的影响。

ORPO算法通过一个简单而创新的方法提高了语言模型的偏好对齐效率，并且减少了所需的计算资源，这使得它在实际应用中具有潜在的优势。

### 代码和模型
在提供的论文摘要中，作者提到了ORPO算法的训练代码已经在GitHub上发布。具体的代码库链接在文档中以脚注的形式给出：

1. [ORPO GitHub Repository](https://github.com/xfactlab/orpo)

你可以通过访问这个链接来获取ORPO算法的源代码，其中应该包含了训练模型所需的所有脚本、文档和指南。通常，这些资源会帮助你理解如何设置环境、准备数据、配置参数以及执行训练。

此外，论文中还提到了与ORPO训练相关的一些预训练模型的检查点，它们发布在Hugging Face模型库中：

2. [Mistral-ORPO-α (7B) Model Checkpoint](https://huggingface.co/kaist-ai/mistral-orpo-alpha)
3. [Mistral-ORPO-β (7B) Model Checkpoint](https://huggingface.co/kaist-ai/mistral-orpo-beta)

这些检查点可以用于进一步的研究或作为微调其他任务的起点。

请注意，为了能够运行这些代码并使用这些模型，你可能需要具备深度学习和自然语言处理的基本知识，以及熟悉相关的技术栈，如Python、PyTorch或TensorFlow等。此外，根据你的计算资源（如GPU），运行这些训练过程可能需要一定的时间。

### trl-ORPOConfig
trl ORPOConfig 参数：
```
ORPOConfig(
_n_gpu=1,
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'gradient_accumulation_kwargs': None},
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
beta=0.1,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
dataset_num_proc=None,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_dropout=True,
disable_tqdm=False,
dispatch_batches=None,
do_eval=True,
do_predict=False,
do_train=False,
eval_accumulation_steps=None,
eval_delay=0,
eval_do_concat_batches=True,
eval_steps=0.2,
evaluation_strategy=steps,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
generate_during_eval=False,
gradient_accumulation_steps=4,
gradient_checkpointing=False,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
is_encoder_decoder=None,
jit_mode_eval=False,
label_names=None,
label_pad_token_id=-100,
label_smoothing_factor=0.0,
learning_rate=8e-06,
length_column_name=length,
load_best_model_at_end=False,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=./results/runs/Apr21_16-33-34_ca433e97f9ed,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=1,
logging_strategy=steps,
lr_scheduler_kwargs={},
lr_scheduler_type=linear,
max_completion_length=None,
max_grad_norm=1.0,
max_length=1024,
max_prompt_length=512,
max_steps=-1,
metric_for_best_model=None,
model_init_kwargs=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=1,
optim=paged_adamw_8bit,
optim_args=None,
optim_target_modules=None,
output_dir=./results/,
overwrite_output_dir=False,
padding_value=None,
past_index=-1,
per_device_eval_batch_size=2,
per_device_train_batch_size=2,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['wandb'],
resume_from_checkpoint=None,
run_name=./results/,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=500,
save_strategy=steps,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
split_batches=None,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
truncation_mode=keep_end,
use_cpu=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=10,
weight_decay=0.0,
)
```
```
这里提供的是一个名为 `ORPOConfig` 的配置文件，它包含了各种参数来配置模型训练过程。以下是每个参数的含义解释：

1. `_n_gpu`: 使用的GPU数量。
2. `accelerator_config`: 加速器配置，用于配置加速训练的选项。
3. `adafactor`: 是否使用Adafactor优化器。
4. `adam_beta1`: Adam优化器的beta1参数。
5. `adam_beta2`: Adam优化器的beta2参数。
6. `adam_epsilon`: Adam优化器的epsilon参数。
7. `auto_find_batch_size`: 是否自动寻找合适的batch size。
8. `beta`: ORPO（Online Random Path Optimization）算法中的参数。
9. `bf16`: 是否使用BF16混合精度。
10. `bf16_full_eval`: 是否在全评估中使用BF16混合精度。
11. `data_seed`: 数据种子。
12. `dataloader_drop_last`: 是否丢弃最后一个不完整的batch。
13. `dataloader_num_workers`: 数据加载器的工作进程数量。
14. `dataloader_persistent_workers`: 是否持久化数据加载器的工作进程。
15. `dataloader_pin_memory`: 是否将数据加载到固定的内存位置。
16. `dataloader_prefetch_factor`: 数据加载器的预取因子。
17. `dataset_num_proc`: 数据集处理的进程数量。
18. `ddp_backend`: 分布式数据并行的后端。
19. `ddp_broadcast_buffers`: 是否广播缓冲区。
20. `ddp_bucket_cap_mb`: DDP通信的最大消息大小。
21. `ddp_find_unused_parameters`: 是否查找未使用的参数。
22. `ddp_timeout`: DDP操作的超时时间。
23. `debug`: 调试选项。
24. `deepspeed`: 是否使用DeepSpeed进行训练。
25. `disable_dropout`: 是否禁用dropout。
26. `disable_tqdm`: 是否禁用tqdm进度条。
27. `dispatch_batches`: 分配批次。
28. `do_eval`: 是否执行评估。
29. `do_predict`: 是否执行预测。
30. `do_train`: 是否执行训练。
31. `eval_accumulation_steps`: 评估积累步骤。
32. `eval_delay`: 评估延迟。
33. `eval_do_concat_batches`: 评估是否进行批次连接。
34. `eval_steps`: 评估步数。
35. `evaluation_strategy`: 评估策略。
36. `fp16`: 是否使用混合精度。
37. `fp16_backend`: 混合精度的后端。
38. `fp16_full_eval`: 是否在全评估中使用混合精度。
39. `fp16_opt_level`: 混合精度的优化级别。
40. `fsdp`: Fully Sharded Data Parallelism（FSDP）选项。
41. `fsdp_config`: FSDP配置。
42. `fsdp_min_num_params`: FSDP的最小参数数量。
43. `fsdp_transformer_layer_cls_to_wrap`: 要包装的Transformer层类。
44. `full_determinism`: 是否启用完全确定性。
45. `generate_during_eval`: 是否在评估期间生成。
46. `gradient_accumulation_steps`: 梯度累积步数。
47. `gradient_checkpointing`: 是否启用梯度检查点。
48. `gradient_checkpointing_kwargs`: 梯度检查点参数。
49. `greater_is_better`: 是否更大更好的度量。
50. `group_by_length`: 是否按长度分组。
51. `half_precision_backend`: 半精度的后端。
52. `hub_always_push`: 是否总是推送到Hub。
53. `hub_model_id`: Hub模型ID。
54. `hub_private_repo`: 是否为私有仓库。
55. `hub_strategy`: Hub策略。
56. `hub_token`: Hub令牌。
57. `ignore_data_skip`: 是否忽略数据跳过。
58. `include_inputs_for_metrics`: 是否包括用于度量的输入。
59. `include_num_input_tokens_seen`: 是否包括已看到的输入标记数量。
60. `include_tokens_per_second`: 是否包括每秒标记数。
61. `is_encoder_decoder`: 是否是编码器-解码器模型。
62. `jit_mode_eval`: 是否在评估模式下启用JIT。
63. `label_names`: 标签名称。
64. `label_pad_token_id`: 标签填充标记的ID。
65. `label_smoothing_factor`: 标签平滑因子。
66. `learning_rate`: 学习率。
67. `length_column_name`: 长度列的名称。
68. `load_best_model_at_end`: 是否在结束时加载最佳模型。
69. `local_rank`: 本地排名。
70. `log_level`: 日志级别。
71. `log_level_replica`: 副本日志级别。
72. `log_on_each_node`: 是否在每个节点上记录日志。
73. `logging_dir`: 日志目录。
74. `logging_first_step`: 是否记录第一步。
75. `logging_nan_inf_filter`: 是否过滤NaN和Inf日志。
76. `logging_steps`: 记录步骤。
77. `logging_strategy`: 记录策略。
78. `lr_scheduler_kwargs`: 学习率调度器参数。
79. `lr_scheduler_type`: 学习率调度器类型。
80. `max_completion_length`: 最大完成长度。
81. `max_grad_norm`: 最大梯度范数。
82. `max_length`: 最大长度。
83. `max_prompt_length`: 最大提示长度。
84. `max_steps`: 最大步数。
85. `metric_for_best_model`: 最佳模型的度量。
86. `model_init_kwargs`: 模型初始化参数。
87. `mp_parameters`: 多处理参数。
88. `neftune_noise_alpha`: Neptune噪声alpha参数。
89. `no_cuda`: 是否禁用CUDA。
90. `num_train_epochs`: 训练周期数。
91. `optim`: 优化器。
92. `optim_args`: 优化器参数。
93. `optim_target_modules`: 优化目标模块。
94. `output_dir`: 输出目录。
95. `overwrite_output_dir`: 是否覆盖输出目录。
96. `padding_value`: 填充值。
97. `past_index`: 过去索引。
98. `per_device_eval_batch_size`: 每个设备的评估批次大小。
99. `per_device_train_batch_size`: 每个设备的训练批次大小。
100. `prediction_loss_only`: 仅预测损失。
101. `push_to_hub`: 是否推送到Hub。
102. `push_to_hub_model_id`: 推送到Hub的模型ID。
103. `push_to_hub_organization`: 推送到Hub的组织。
104. `push_to_hub_token`: 推送到Hub的令牌。
105. `ray_scope`: Ray作用域。
106. `remove_unused_columns`: 是否移除未使用的列。
107. `report_to`: 报告目标。
108. `resume_from_checkpoint`: 从检查点恢复。
109. `run_name`: 运行名称。
110. `save_on_each_node`: 是否在每个节点上保存。
111. `save_only_model`: 是否仅保存模型。
112. `save_safetensors`: 是否保存安全张量。
113. `save_steps`: 保存步数。
114. `save_strategy`: 保存策略。
115. `save_total_limit`: 保存总限制。
116. `seed`: 种子。
117. `skip_memory_metrics`: 是否跳过内存度量。
118. `split_batches`: 是否拆分批次。
119. `tf32`: 是否使用TF32精度。
120. `torch_compile`: 是否使用Torch编译。
121. `torch_compile_backend`: Torch编译后端。
122. `torch_compile_mode`: Torch编译模式。
123. `torchdynamo`: Torch动态模型。
124. `tpu_metrics_debug`: TPU度量调试。
125. `tpu_num_cores`: TPU核心数。
126. `truncation_mode`: 截断模式。
127. `use_cpu`: 是否使用CPU。
128. `use_ipex`: 是否使用IPex。
129. `use_legacy_prediction_loop`: 是否使用旧版预测循环。
130. `use_mps_device`: 是否使用MPS设备。
131. `warmup_ratio`: 热身比率。
132. `warmup_steps`: 热身步数。
133. `weight_decay`: 权重衰减。

这些参数用于配置训练过程中的各种设置，如优化器、学习率、批处理大小、日志记录、评估策略等。
```


## 笔记
- [Fine-tune Llama3-8B with bnb4bit+LoRA+ORPO](https://github.com/weedge/doraemon-nb/blob/main/Fine_tune_Llama3_8B_with_bnb4bit%2BLoRA%2BORPO.ipynb)