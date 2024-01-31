## Models-tinyllamas

- download from hf karpathy/tinyllamas pre trained models 

```shell
# u can use huggingface proxy to download
huggingface-cli download karpathy/tinyllamas --local-dir ./models/tinyllamas --local-dir-use-symlinks False

```

- upload pre-trained models with TinyStoriesZh dataset to hf weege007/babyllm  
```shell
# Usage:  huggingface-cli upload --repo-type {model,dataset,space} [repo_id] [local_path] [path_in_repo]
# u can cron to upload pre trained models
huggingface-cli upload \
    --repo-type model weege007/babyllm \
    ${ckpt_model_file} \
    /models/TinyStoriesZh/${vocab_size}/${params_size}.model
```
