# train llama2 model records

## pre-training

datasets:
 - tinystories

recode:
 - run2024_02_01_13_59_58: https://wandb.ai/weege007/baby_llm_llama2/runs/3k9u1cer

notebook:
 - https://github.com/weedge/doraemon-nb/blob/main/baby_llm_llama.ipynb

## supervised fine-tuning



## huggingface transformers TrainingArguments

detail see(choose used transformers tag version):
https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

```
TrainingArguments(
_n_gpu=0,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
dispatch_batches=None,
do_eval=False,
do_predict=False,
do_train=False,
eval_accumulation_steps=None,
eval_delay=0,
eval_steps=None,
evaluation_strategy=IntervalStrategy.NO,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=True,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=1e-05,
length_column_name=length,
load_best_model_at_end=False,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=./results/runs/Apr18_19-58-28_wuyongdeMacBook-Pro.local,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=1,
logging_strategy=IntervalStrategy.STEPS,
lr_scheduler_kwargs={},
lr_scheduler_type=SchedulerType.COSINE,
max_grad_norm=0.3,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=1,
optim=OptimizerNames.PAGED_ADAMW,
optim_args=None,
output_dir=./results,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=4,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['wandb'],
resume_from_checkpoint=None,
run_name=./results,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=0,
save_strategy=IntervalStrategy.STEPS,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
split_batches=False,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_cpu=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_mps_device=False,
warmup_ratio=0.03,
warmup_steps=0,
weight_decay=0.001,
)
```
```
这个长长的参数列表是用来配置训练过程的。下面是每个参数的含义解释：

1. `_n_gpu`: 使用的GPU数量。
2. `adafactor`: 是否使用Adafactor优化器。
3. `adam_beta1`: Adam优化器的beta1参数。
4. `adam_beta2`: Adam优化器的beta2参数。
5. `adam_epsilon`: Adam优化器的epsilon参数。
6. `auto_find_batch_size`: 是否自动寻找合适的batch size。
7. `bf16`: 是否使用BF16混合精度。
8. `bf16_full_eval`: 是否在全评估中使用BF16混合精度。
9. `data_seed`: 数据种子。
10. `dataloader_drop_last`: 是否丢弃最后一个不完整的batch。
11. `dataloader_num_workers`: 数据加载器的工作进程数量。
12. `dataloader_persistent_workers`: 是否持久化数据加载器的工作进程。
13. `dataloader_pin_memory`: 是否将数据加载到固定的内存位置。
14. `ddp_backend`: 分布式数据并行的后端。
15. `ddp_broadcast_buffers`: 是否广播缓冲区。
16. `ddp_bucket_cap_mb`: DDP通信的最大消息大小。
17. `ddp_find_unused_parameters`: 是否查找未使用的参数。
18. `ddp_timeout`: DDP操作的超时时间。
19. `debug`: 调试选项。
20. `deepspeed`: 是否使用DeepSpeed进行训练。
21. `disable_tqdm`: 是否禁用tqdm进度条。
22. `dispatch_batches`: 分配批次。
23. `do_eval`: 是否执行评估。
24. `do_predict`: 是否执行预测。
25. `do_train`: 是否执行训练。
26. `eval_accumulation_steps`: 评估积累步骤。
27. `eval_delay`: 评估延迟。
28. `eval_steps`: 评估步骤。
29. `evaluation_strategy`: 评估策略。
30. `fp16`: 是否使用混合精度。
31. `fp16_backend`: 混合精度的后端。
32. `fp16_full_eval`: 是否在全评估中使用混合精度。
33. `fp16_opt_level`: 混合精度的优化级别。
34. `fsdp`: Fully Sharded Data Parallelism（FSDP）选项。
35. `fsdp_config`: FSDP配置。
36. `fsdp_min_num_params`: FSDP的最小参数数量。
37. `fsdp_transformer_layer_cls_to_wrap`: 要包装的Transformer层类。
38. `full_determinism`: 是否启用完全确定性。
39. `gradient_accumulation_steps`: 梯度累积步数。
40. `gradient_checkpointing`: 是否启用梯度检查点。
41. `gradient_checkpointing_kwargs`: 梯度检查点参数。
42. `greater_is_better`: 是否更大更好的度量。
43. `group_by_length`: 是否按长度分组。
44. `half_precision_backend`: 半精度的后端。
45. `hub_always_push`: 是否总是推送到Hub。
46. `hub_model_id`: Hub模型ID。
47. `hub_private_repo`: 是否为私有仓库。
48. `hub_strategy`: Hub策略。
49. `hub_token`: Hub令牌。
50. `ignore_data_skip`: 是否忽略数据跳过。
51. `include_inputs_for_metrics`: 是否包括用于度量的输入。
52. `include_num_input_tokens_seen`: 是否包括已看到的输入标记数量。
53. `include_tokens_per_second`: 是否包括每秒标记数。
54. `jit_mode_eval`: 是否在评估模式下启用JIT。
55. `label_names`: 标签名称。
56. `label_smoothing_factor`: 标签平滑因子。
57. `learning_rate`: 学习率。
58. `length_column_name`: 长度列的名称。
59. `load_best_model_at_end`: 是否在结束时加载最佳模型。
60. `local_rank`: 本地排名。
61. `log_level`: 日志级别。
62. `log_level_replica`: 副本日志级别。
63. `log_on_each_node`: 是否在每个节点上记录日志。
64. `logging_dir`: 日志目录。
65. `logging_first_step`: 是否记录第一步。
66. `logging_nan_inf_filter`: 是否过滤NaN和Inf日志。
67. `logging_steps`: 记录步骤。
68. `logging_strategy`: 记录策略。
69. `lr_scheduler_kwargs`: 学习率调度器参数。
70. `lr_scheduler_type`: 学习率调度器类型。
71. `max_grad_norm`: 最大梯度范数。
72. `max_steps`: 最大步数。
73. `metric_for_best_model`: 最佳模型的度量。
74. `mp_parameters`: 多处理参数。
75. `neftune_noise_alpha`: Neptune噪声alpha参数。
76. `no_cuda`: 是否禁用CUDA。
77. `num_train_epochs`: 训练周期数。
78. `optim`: 优化器。
79. `optim_args`: 优化器参数。
80. `output_dir`: 输出目录。
81. `overwrite_output_dir`: 是否覆盖输出目录。
82. `past_index`: 过去索引。
83. `per_device_eval_batch_size`: 每个设备的评估批次大小。
84. `per_device_train_batch_size`: 每个设备的训练批次大小。
85. `prediction_loss_only`: 仅预测损失。
86. `push_to_hub`: 是否推送到Hub。
87. `push_to_hub_model_id`: 推送到Hub的模型ID。
88. `push_to_hub_organization`: 推送到Hub的组织。
89. `push_to_hub_token`: 推送到Hub的令牌。
90. `ray_scope`: Ray作用域。
91. `remove_unused_columns`: 是否移除未使用的列。
92. `report_to`: 报告目标。
93. `resume_from_checkpoint`: 从检查点恢复。
94. `run_name`: 运行名称。
95. `save_on_each_node`: 是否在每个节点上保存。
96. `save_only_model`: 是否仅保存模型。
97. `save_safetensors`: 是否保存安全张量。
98. `save_steps`: 保存步骤。
99. `save_strategy`: 保存策略。
100. `save_total_limit`: 保存总限制。
101. `seed`: 种子。
102. `skip_memory_metrics`: 是否跳过内存度量。
103. `split_batches`: 是否拆分批次。
104. `tf32`: 是否使用TF32精度。
105. `torch_compile`: 是否使用Torch编译。
106. `torch_compile_backend`: Torch编译后端。
107. `torch_compile_mode`: Torch编译模式。
108. `torchdynamo`: Torch动态模型。
109. `tpu_metrics_debug`: TPU度量调试。
110. `tpu_num_cores`: TPU核心数。
111. `use_cpu`: 是否使用CPU。
112. `use_ipex`: 是否使用IPex。
113. `use_legacy_prediction_loop`: 是否使用旧版预测循环。
114. `use_mps_device`: 是否使用MPS设备。
115. `warmup_ratio`: 热身比率。
116. `warmup_steps`: 热身步数。
117. `weight_decay`: 权重衰减。

这些参数可以用来调整模型的训练行为，从优化器的选择到学习率的调度等等。
```

hf Transforms Trainer 类 和 SFTTrainer (Supervised Fine tuning trainer)类 的区别：

Trainer 和 SFTTrainer 是 Hugging Face 中用于训练 Transformer 模型的两个类，分别用于训练从头开始的模型和对预训练模型进行微调。其中Trainer也适用于lora模型微调任务

Trainer是一个通用目的的训练类，用于在监督学习任务（如文本分类、问答和摘要）上从头开始训练模型。它提供了广泛的配置选项，支持复杂的训练工作流和高度可定制的超参数、优化器、调度器、日志和评估指标。SFTTrainer是一个优化的微调类，专门用于使用较小数据集微调预训练模型。它提供了一个更简单的界面和更少的配置选项，以便更容易开始使用，并使用参数有效（PEFT）和打包优化来减少内存消耗。SFTTrainer可以使用较小的数据集和较短的训练时间实现与Trainer相当或更好的准确性。

1. **Trainer**：
   - 用于从头开始训练Transformer模型。
   - 提供广泛的配置选项，支持复杂的训练工作流和高度可定制的超参数、优化器、调度器、日志和评估指标。
   - 适用于具有大量数据集的情况。
2. **SFTTrainer**：
   - 用于微调预训练Transformer模型。
   - 提供一个更简单的界面和更少的配置选项，使其更容易开始使用。
   - 使用参数有效（PEFT）和打包优化来减少内存消耗。
   - 适用于具有较小数据集的情况，并且可以使用较短的训练时间实现与Trainer相当或更好的准确性。
3. **选择Trainer或SFTTrainer**：
   - **使用Trainer**：如果您有大量数据集，并且需要对训练循环或复杂训练工作流进行广泛定制。
   - **使用SFTTrainer**：如果您有一个预训练模型和一个相对较小的数据集，并且想要一个更简单和更快的微调体验，同时具有高效的内存使用。

