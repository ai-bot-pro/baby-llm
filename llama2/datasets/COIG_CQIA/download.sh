#!/bin/bash
# download from hg m-a-p/COIG-CQIA stories datasets 
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

# m-a-p/COIG-CQIA stories
huggingface-cli download \
  --repo-type dataset m-a-p/COIG-CQIA \
  --local-dir ${data_dir}/m-a-p/COIG-CQIA \
  --local-dir-use-symlinks False
