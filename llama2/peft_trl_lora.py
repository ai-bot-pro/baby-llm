r"""
use hf transformers, trl and peft with lora config to train model

Install the following libraries:
pip3 install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 scipy tensorboardX==2.6.2
"""
from typing import Optional
from dataclasses import dataclass, field

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
import datasets as ds
import torch
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        # default="meta-llama/Llama-2-7b-hf",
        default="./out/model_hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub"},
    )
    dataset_dir: Optional[str] = field(
        default="./datas/datasets/guanaco-llama2-1k",
        metadata={"help": "firstly The instruction dataset to use"},
    )
    hf_dataset_name: Optional[str] = field(
        default="weege007/guanaco-llama2-1k",
        metadata={"help": "The instruction hf dataset to use"},
    )
    new_model: Optional[str] = field(
        # default="llama-2-7b-miniguanaco",
        default="baby-llm-llama-2-miniguanaco",
        metadata={"help": "Fine-tuned model name"}
    )
    merge_and_push: Optional[bool] = field(
        default=False, metadata={"help": "Merge and push weights after training"}
    )

    # QLoRA parameters
    # https://arxiv.org/abs/2305.14314
    lora_r: Optional[int] = field(
        default=64, metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout probability for LoRA layers"}
    )

    # bitsandbytes parameters
    use_4bit: Optional[bool] = field(
        default=False, metadata={"help": "Activate 4-bit precision base model loading"}
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16", metadata={"help": "Compute dtype for 4-bit base models"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type (fp4 or nf4)"}
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Activate nested quantization for 4-bit base models (double quantization)"
        },
    )

    # TrainingArguments parameters
    output_dir: str = field(
        default="./results",
        metadata={
            "help": "Output directory where the model predictions and checkpoints will be stored"
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    fp16: Optional[bool] = field(
        default=False, metadata={"help": "Enable fp16 training"}
    )
    bf16: Optional[bool] = field(
        default=False, metadata={"help": "Enable bf16 training"}
    )
    tf32: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enable the TF32 mode (available in Ampere and newer GPUs)"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of update steps to accumulate the gradients for"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    max_grad_norm: Optional[float] = field(
        default=0.3, metadata={"help": "Maximum gradient normal (gradient clipping)"}
    )
    learning_rate: Optional[float] = field(
        default=1e-5, metadata={"help": "Initial learning rate (AdamW optimizer)"}
    )
    weight_decay: Optional[int] = field(
        default=0.001,
        metadata={
            "help": "Weight decay to apply to all layers except bias/LayerNorm weights"
        },
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "Optimizer to use"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule"},
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "Number of training steps (overrides num_train_epochs)"},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={
            "help": "Ratio of steps for a linear warmup (from 0 to learning rate)"
        },
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably"
        },
    )
    save_steps: float = field(
        default=0, metadata={"help": "Save checkpoint every X updates steps"}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every X updates steps"}
    )
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Allows to resume training from the latest checkpoint in output_dir"}
    )

    # SFT parameters
    max_seq_length: Optional[int] = field(
        default=2048, metadata={"help": "Maximum sequence length to use"}
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Pack multiple short examples in the same input sequence to increase efficiency"
        },
    )


parser = HfArgumentParser(ScriptArguments)
arr_script_args = parser.parse_args_into_dataclasses()
script_args = arr_script_args[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset (you can process it here)
if len(script_args.dataset_dir) > 0:
    dataset = ds.load_from_disk(script_args.dataset_dir)
elif len(script_args.hf_dataset_name) > 0:
    dataset = ds.load_dataset(script_args.hf_dataset_name, split="train")
else:
    raise ValueError("dataset don't load")

    # Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, script_args.bnb_4bit_compute_dtype)

# Accessible large language models via k-bit quantization for PyTorch.
# https://huggingface.co/docs/bitsandbytes/main/en/index
bnb_config = BitsAndBytesConfig(
    load_in_4bit=script_args.use_4bit,
    bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=script_args.use_nested_quant,
)
print(bnb_config)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and script_args.use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with --bf16")
        print("=" * 80)


device_map = "auto"
# max_memory = {0: "1GIB", 1: "1GIB", 2: "2GIB", 3: "10GIB", "cpu": "30GB"}
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    device_map=device_map,
    # max_memory=max_memory,
)
print(model.config)
model.config.use_cache = False
model.config.pretraining_tp = 1
print(model)
# model.print_trainable_parameters()
tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, trust_remote_code=True, is_fast=False
)
print(tokenizer)
# note: llama2 sp bpe no pad_token id, use eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
print(tokenizer)

prompt = "hello"
pipe = pipeline(task="text-generation", model=model,
                tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>{prompt}")
print(result)
# exit(0)

# Load LoRA configuration
peft_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    r=script_args.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
print(peft_config)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    num_train_epochs=script_args.num_train_epochs,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    weight_decay=script_args.weight_decay,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    # report_to="all",
    report_to="tensorboard",
)
print(training_arguments)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)
print(trainer)
print(trainer.model)
trainer.model.print_trainable_parameters()

# Train model
trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

prompt = "hello"
pipe = pipeline(task="text-generation", model=model,
                tokenizer=tokenizer, max_length=200)
# result = pipe(f"<s>{prompt}")
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result)
# exit(0)


# Save trained model
trainer.model.save_pretrained(script_args.new_model)

if script_args.merge_and_push:
    # Free memory for merging weights
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.new_model, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()
    print(model)

    model.push_to_hub(script_args.new_model,
                      use_temp_dir=False, private=True)
    tokenizer.push_to_hub(script_args.new_model,
                          use_temp_dir=False, private=True)
