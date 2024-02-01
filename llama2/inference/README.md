# inference

1. build
```shell
# use clang on macos, defualt use gcc on linux 
make build CC=/usr/local/opt/llvm/bin/clang -C llama2/inference
#make build_fast CC=/usr/local/opt/llvm/bin/clang -C llama2/inference
#make build_omp CC=/usr/local/opt/llvm/bin/clang -C llama2/inference
```
2. download pre-trained weight model and tokenizer model from HF
```shell
huggingface-cli download --repo-type model \
    weege007/babyllm  \
    --local-dir ./datas \
    --local-dir-use-symlinks False
```
3. run inference with pre-trained weight model and tokenizer model
```shell
llama2/inference/bin/inference ./datas/models/TinyStoriesZh/4216/25M.bin -z ./datas/tokenizers/datas/52AI/TinyStoriesZh/tok4216.bin
```