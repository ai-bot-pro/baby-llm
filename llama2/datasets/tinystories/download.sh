#!/bin/bash
# download from hg roneneldan/TinyStories TinyStories datasets 
# paper: TinyStories: How Small Can Language Models Be and Still Speak Coherent English? 
# u can use huggingface proxy to download

set -e

data_dir=./datas

while getopts d: flag
do
  case "${flag}" in
    d) data_dir=${OPTARG};;
  esac
done

#huggingface-cli login

# roneneldan/TinyStories
huggingface-cli download \
  --repo-type dataset roneneldan/TinyStories TinyStories_all_data.tar.gz \
  --local-dir ${data_dir}/roneneldan/TinyStories \
  --local-dir-use-symlinks False
cd ${data_dir}/roneneldan/TinyStories
tar -zxvf TinyStories_all_data.tar.gz
cd -

# 52AI/TinyStoriesZh (use https://github.com/nidhaloff/deep-translator translate TinyStories datasets)
huggingface-cli download \
  --repo-type dataset 52AI/TinyStoriesZh \
  --local-dir ${data_dir}/52AI/TinyStoriesZh \
  --local-dir-use-symlinks False
cd ${data_dir}/52AI/TinyStoriesZh 
tar -zxvf TinyStories_all_data_zh_1M.tar.gz
tar -zxvf TinyStories_all_data_zh_2M.tar.gz
cd -