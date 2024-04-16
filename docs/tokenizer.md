## [sentencepiece](https://github.com/google/sentencepiece) 
1. https://github.com/google/sentencepiece/blob/master/doc/options.md
```python
# python spm train api
spm.SentencePieceTrainer.train(input=tiny_file,
                               model_prefix=prefix,
                               model_type="bpe",
                               vocab_size=vocab_size,
                               self_test_sample_size=0,
                               input_format="text",
                               character_coverage=1.0,
                               num_threads=os.cpu_count(),
                               split_digits=True,
                               allow_whitespace_only_pieces=True,
                               byte_fallback=True,
                               unk_surface=r" \342\201\207 ",
                               normalization_rule_name="identity")
```
```bash
spm_train \
  --input=$tiny_file \
  --model_prefix=$prefix \
  --model_type="bpe" \
  --vocab_size=$vocab_size \
  --elf_test_sample_size=0 \
  --input_format="text" \
  --character_coverage=1.0 \
  --num_threads=$cpu_nums \
  --split_digits=True \
  --allow_whitespace_only_pieces=True \
  --byte_fallback=True \
  --unk_surface=" \342\201\207 " \
  --normalization_rule_name="identity"

```

**从头训练词表**：https://huggingface.co/docs/tokenizers/pipeline

**扩展词表**：[sentencepiece_add_new_vocab.ipynb](https://github.com/google/sentencepiece/blob/9cf136582d9cce492ba5a0cfb775f9e777fe07ea/python/add_new_vocab.ipynb)

**缩减词表**：[Reducing the SentencePiece Vocabulary Size of Pretrained NLP Models](https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/) | [toknizer reduce](https://github.com/ceshine/finetuning-t5/blob/8d4db99e11c0356db7c4535e9caaae723f656a51/notebooks/Manipulate%20Sentencepiece%20Vocabulary.ipynb)



## [tiktoken](https://github.com/openai/tiktoken)

1. [How_to_count_tokens_with_tiktoken.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)


## 笔记
- [tokenizer](https://colab.research.google.com/drive/1xCAFx7xuXcVaYPNgCHHGLCmvvR2F-SJu?usp=sharing)

## 参考
- [karpathy/minibpe](https://github.com/karpathy/minbpe) | [youtube: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)