
set -e
data_dir=./datas
vocab_size=512
hg_repo_id=weege007/babyllm

while getopts d:v:h: flag
do
  case "${flag}" in
    d) data_dir=${OPTARG};;
    v) vocab_size=${OPTARG};;
    h) hg_repo_id=${OPTARG};;
  esac
done

#huggingface-cli login

# Usage:  huggingface-cli upload --repo-type {model,dataset,space} [repo_id] [local_path] [path_in_repo]

# upload tok${vocab_size}.model
huggingface-cli upload \
    --repo-type model ${hg_repo_id} \
    ${data_dir}/tok${vocab_size}.model \
    /tokenizers/${data_dir}/tok${vocab_size}.model

