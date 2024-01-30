# Model
- [Mistral 7B](https://arxiv.org/pdf/2310.06825.pdf) | https://mistral.ai/news/announcing-mistral-7b/ the same as llama2 Architecture
- [**Outrageosly Large Neural Networks: The Sparsely-Gated Mixture-Of-Experts layer**](https://arxiv.org/pdf/1701.06538.pdf)
- [**Mixtral of Experts**](https://arxiv.org/pdf/2401.04088.pdf) | https://mistral.ai/news/mixtral-of-experts/
- **https://github.com/mistralai/mistral-src**

## Model-Architecture

```python
from transformers import AutoModelForCausalLM

def print_hf_model_for_causalLM(model_path):
  hf_model = AutoModelForCausalLM.from_pretrained(model_path)
  print(hf_model)
  hf_dict = hf_model.state_dict()
  print(hf_model.config)
  #print(hf_dict.keys())
```
```python
print_hf_model_for_causalLM("mistralai/Mistral-7B-v0.1")
```
```
MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm()
        (post_attention_layernorm): MistralRMSNorm()
      )
    )
    (norm): MistralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
MistralConfig {
  "_name_or_path": "mistralai/Mistral-7B-v0.1",
  "architectures": [
    "MistralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

```
```python
print_hf_model_for_causalLM("mistralai/Mixtral-8x7B-v0.1")
```