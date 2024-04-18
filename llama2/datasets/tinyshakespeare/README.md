
this dataset is from https://github.com/karpathy/char-rnn; 
just download and preprocess, then use this script to pretokenize, 
to debug/test pretokenize and pre-trainning

```shell
# 1. download tinyshakespeare
bash ./llama2/datasets/tinyshakespeare/download.sh -d ${datas_dir}

# 2. if need train tokenizer, train tokenizer vocab size=323
python3 ./llama2/datasets/tinyshakespeare/preprocess.py train_vocab --vocab_size=323 --data_dir=${data_dir}

# 3. if need merge, merge trained tokenizer to src tokenizer
# src from https://huggingface.co/meta-llama/Llama-2-7b-hf tokenizer to merge
python3 ./llama2/datasets/_common/preprocess.py merge_tokenizer --data_dir=${data_dir} --src_tokenizer_model="meta-llama/Llama-2-7b-hf" --merge_tokenizer_model=${merge} --src_from="llama2"
# src from custom tokenizer to merge
python3 ./llama2/datasets/_common/preprocess.py merge_tokenizer --data_dir=${data_dir} --src_tokenizer_model=${src} --merge_tokenizer_model=${merge} --src_from="coustom"

# print tokenizer vocab size
python3 ./llama2/datasets/_common/preprocess.py print_tokenizer --tokenizer_model=${model}

# 4. use trained tokenizer vocab model pretokenize tinyshakespeare datasets to tokenizer id for model trainning and inference
python3 ./llama2/datasets/tinyshakespeare/preprocess.py pretokenize --vocab_size=323 --data_dir=${data_dir} --tokenizer_model=${model} --train_size=0.9

# 5. export pb tokenizer model to binary file for LM training and inference if use sentencepiece
python3 ./llama2/datasets/_common/tokenizer.py export --tokenizer-model=${model}
# export hf tokenizer dir
python3 ./llama2/datasets/_common/tokenizer.py export_to_hf --tokenizer-model=${model} --output_hf_dir=${hf_dir}
```