
set -e
data_dir=./datas
vocab_size=512

while getopts d:v: flag
do
  case "${flag}" in
    d) data_dir=${OPTARG};;
    v) vocab_size=${OPTARG};;
  esac
done

#huggingface-cli login

# Usage:  huggingface-cli upload --repo-type {model,dataset,space} [repo_id] [local_path] [path_in_repo]

# upload tok${vocab_size}.model
huggingface-cli upload \
    --repo-type model weege007/babyllm \
    ${data_dir}/tok${vocab_size}.model \
    /tokenizers/${data_dir}/tok${vocab_size}.model

