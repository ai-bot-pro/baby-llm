#!/bin/bash
# download from hg HuggingFaceTB/cosmopedia stories datasets 
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

# HuggingFaceTB/cosmopedia stories
huggingface-cli download \
  --repo-type dataset HuggingFaceTB/cosmopedia data/stories/train-00002-of-00043.parquet \
  --local-dir ${data_dir}/HuggingFaceTB/cosmopedia \
  --local-dir-use-symlinks False
