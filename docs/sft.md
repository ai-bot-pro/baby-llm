## supervised fine-tuning

- Full Fine-tuning: Adjusts all parameters of the LLM using task-specific data.
    like pre-training, but use resume ckpt.pt 
    and use prompt-text datasets with tokenizer(sp bpe)

- Parameter-efficient Fine-tuning (PEFT): Modifies select parameters for more efficient adaptation.
    need pre-training more parameters model
    eg: LoRA; Prefix-tuning P-tuning v2, P-tuning; IA3
    see hf peft: https://huggingface.co/docs/peft/conceptual_guides/adapter 
    - LoRA (generate task)
    - Soft prompts (NLP/NLU sub task eg: classify)
    - IA3 (generate task)