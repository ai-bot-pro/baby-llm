#!/bin/bash

set -e

data_dir=./datas/datasets

while getopts d: flag
do
  case "${flag}" in
    d) data_dir=${OPTARG};;
  esac
done

wget "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -O ${data_dir}/tinyshakespeare.txt