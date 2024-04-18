#!/bin/bash
# download from hg pleisto/wikipedia-cn-20230720-filtered datasets 
# u can use huggingface proxy to download

set -e

data_dir=./datas/datasets

while getopts d: flag
do
  case "${flag}" in
    d) data_dir=${OPTARG};;
  esac
done

#huggingface-cli login

# pleisto/wikipedia-cn-20230720-filtered
huggingface-cli download \
  --repo-type dataset pleisto/wikipedia-cn-20230720-filtered  \
  --local-dir ${data_dir}/pleisto/wikipedia-cn-20230720-filtered \
  --local-dir-use-symlinks False
