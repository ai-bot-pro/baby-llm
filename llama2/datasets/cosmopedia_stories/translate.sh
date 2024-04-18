#!/bin/bash
set -e 

begin=0
end=42
stories_cn=10000

while getopts b:e:s: flag; do
  case "${flag}" in
    b)
      begin=${OPTARG};;
    e)
      end=${OPTARG};;
    s)
      stories_cn=${OPTARG};;
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

[ $begin -lt 0 ] && echo "begin ${begin} must >= 0" && exit 1
[ $end -lt 0 ] && [ $end -ge 43 ] && echo "end ${end} must >= 0 && < 43" && exit 1
[ $end -lt $begin ] && echo "end ${end} must > begin ${begin}" && exit 1
[ $stories_cn -le 0 ] && echo "stories_cn ${stores_cn} must > 0" && exit 1

for i in `seq ${begin} ${end}`;do
    bash llama2/datasets/cosmopedia_stories/download.sh -s ${i}
    python3 llama2/datasets/cosmopedia_stories/translate.py translate \
        -s ./datas/datasets/HuggingFaceTB/cosmopedia/data/stories \
        -t ./datas/datasets/HuggingFaceTB/cosmopedia_zh${i} -ss $stories_cn
    python3 llama2/datasets/cosmopedia_stories/translate.py convert \
        -s ./datas/datasets/HuggingFaceTB/cosmopedia_zh${i} \
        -t ./datas/datasets/HuggingFaceTB/cosmopedia/csv/stories -ff csv 
    mv ./datas/datasets/HuggingFaceTB/cosmopedia/csv/stories/train.csv \
       ./datas/datasets/HuggingFaceTB/cosmopedia/csv/stories/${i}.csv
    rm -rf ./datas/datasets/HuggingFaceTB/cosmopedia_zh${i}
    rm -f ./datas/datasets/HuggingFaceTB/cosmopedia/data/stories/*.parquet
done