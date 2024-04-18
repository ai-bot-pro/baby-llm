#!/bin/bash
# download from hg HuggingFaceTB/cosmopedia stories datasets 
# u can use huggingface proxy to download

set -e

data_dir=./datas/datasets
select_ds=0

while getopts d:s: flag; do
  case "${flag}" in
    d)
      data_dir=${OPTARG};;
    s)
      select_ds=${OPTARG};;
    \?)
      echo "Invalid option: ${OPTARG}" 1>&2
      exit 1
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." 1>&2
      exit 1
      ;;
  esac
done


file_name="data/stories/train-00000-of-00043.parquet"
if [ $select_ds -lt 0 ]; then
  file_name=""
elif [ $select_ds -ge 0 ] && [ $select_ds -lt 10 ]; then
  file_name="data/stories/train-0000${select_ds}-of-00043.parquet"
elif [ $select_ds -ge 10 ] && [ $select_ds -le 42 ]; then
  file_name="data/stories/train-000${select_ds}-of-00043.parquet"
else
  echo "select_ds 这个数不符合条件"
  exit 1
fi

#huggingface-cli login

# HuggingFaceTB/cosmopedia stories
huggingface-cli download \
  --repo-type dataset HuggingFaceTB/cosmopedia ${file_name} \
  --local-dir ${data_dir}/HuggingFaceTB/cosmopedia \
  --local-dir-use-symlinks False
