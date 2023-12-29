# Model

1. [**Llama 2: Open Foundation and Fine-Tuned Chat Models**](https://arxiv.org/abs/2307.09288)
2. [meta-llam2-model.py](https://github.com/facebookresearch/llama/blob/main/llama/model.py)
3. [HF-transformer-llama2](https://huggingface.co/docs/transformers/model_doc/llama2)
   
## Model-Architecture

```python
from transformers import AutoModelForCausalLM
from torch import nn

def print_hf_llama2_model(model_path):
    # load HF pretrained llama2 model
    hf_model = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto')
    print(hf_model)
    hf_dict = hf_model.state_dict()
    print(hf_model.config)

    for i in range(0,hf_model.config.num_hidden_layers):
        print(f"model.layers.{i}.input_layernorm.weight => {hf_dict[f'model.layers.{i}.input_layernorm.weight'].shape}")
        print(f"model.layers.{i}.self_attn.q_proj.weight => {hf_dict[f'model.layers.{i}.self_attn.q_proj.weight'].shape}")
        print(f"model.layers.{i}.self_attn.k_proj.weight => {hf_dict[f'model.layers.{i}.self_attn.k_proj.weight'].shape}")
        print(f"model.layers.{i}.self_attn.k_proj.weight => {hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'].shape}")
        print(f"model.layers.{i}.self_attn.k_proj.weight => {hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'].shape}")
        print(f"model.layers.{i}.post_attention_layernorm.weight => {hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'].shape}")
        print(f"model.layers.{i}.mlp.gate_proj.weight => {hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'].shape}")
        print(f"model.layers.{i}.mlp.down_proj.weight => {hf_dict[f'model.layers.{i}.mlp.down_proj.weight'].shape}")
        print(f"model.layers.{i}.mlp.up_proj.weight => {hf_dict[f'model.layers.{i}.mlp.up_proj.weight'].shape}")

    # final classifier
    print(f"lm_head.weight => {hf_dict['lm_head.weight'].shape}")

```

- **7B**
``` python
model_path = "meta-llama/Llama-2-7b-hf"
print_hf_llama2_model(model_path)
```
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
LlamaConfig {
  "_name_or_path": "meta-llama/Llama-2-7b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.36.2",
  "use_cache": true,
  "vocab_size": 32000
}
```

![](llama2.drawio.png)


## Model-File-Structure