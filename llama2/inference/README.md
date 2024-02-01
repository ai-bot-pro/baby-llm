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
```
从前，有一个小女孩，名叫莉莉。她喜欢在阳光下外面玩耍。有一天，她在后院看到一棵柠檬树。它很高，上面结满了柠檬。
莉莉问妈妈：“能给我一个柠檬吗？”
妈妈说：“当然，但是我们要小心，挤多了。”
莉莉捏了一个柠檬，递给妈妈。但当她挤压时，柠檬汁就变酸了。
“呃！这柠檬汁太酸了，”莉莉说。
妈妈说：“我们去商店买一些甜柠檬吧。”
在商店里，他们找到了一些甜柠檬，并在里面装了一些甜柠檬。莉莉很高兴能吃到她的酸柠檬，她感谢妈妈给她带来了甜柠檬。
```