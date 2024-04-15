# chatGLM use the sp bpe tokenizer like llama2 tokenizer
# Qwen use the tiktoken bpe tokenizer like gpt tokenizer
import os
import argparse
import glob
import json

from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import sentencepiece as spm

import sys
sys.path.append(os.path.split(sys.path[0])[0])
from _common.tokenizer import Tokenizer
from _common.preprocess import print_tokenizer

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def train_vocab(data_dir, vocab_size):
    """
    train the tokenizer.model and vocab from the dataset
    """
    assert vocab_size > 0, "Vocab size must be positive"
    assert vocab_size < (1 << 16), "Vocab size must less than 2^16"

    # output file prefix path for sentencepiece
    prefix = os.path.join(data_dir, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10

    # 1) export a large chunk of text as a single text file all.txt
    all_file = os.path.join(data_dir, "all.txt")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {all_file} with {num_shards} shards...")
    with open(all_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["completion"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(all_file) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    spm.SentencePieceTrainer.train(input=all_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def custom_tokenizer_process(item, tokenizer):
    token_ids = tokenizer.batch_encode(item["completion"], bos=True, eos=False)

    return {"input_ids": token_ids}


def pretokenize(data_dir, vocab_size, tokenizer_model=None, batch_size=30, test_size=0.1, train_size=0.9):
    """
    pretokenize the dataset with custom tokenizer , save tokenids bytes to file 
    """
    tokenizer_model = os.path.join(
        data_dir, f"tok{vocab_size}.model") if tokenizer_model is None else tokenizer_model
    tokenizer = Tokenizer(tokenizer_model)

    tokenized_filename = {}
    tokenized_filename["test"] = os.path.join(
        data_dir, f"tok{vocab_size}", f"{test_size}.test.bin")
    tokenized_filename["train"] = os.path.join(
        data_dir, f"tok{vocab_size}", f"{train_size}.train.bin")

    # ds = load_dataset(data_dir, split="train[:3]")
    ds = load_dataset(data_dir, split="train")
    print(f'dataset: {ds}')
    ds = ds.train_test_split(test_size=test_size, train_size=train_size)
    print(f'split train dataset [{test_size}:{train_size}]: {ds}')
    for split in ["train", "test"]:
        ds[split] = ds[split].map(custom_tokenizer_process, batched=True,
                                  batch_size=batch_size, remove_columns=["source", "completion"], fn_kwargs={"tokenizer": tokenizer})
        print(f'after tokenizer process {split} dataset: {ds[split]}')
        # convert to uint16 nparray, note: tokenizer vocab_size must <= 2^16
        all_tokens = np.array(ds[split]["input_ids"], dtype=np.uint16)
        # write the bytes
        with open(tokenized_filename[split], "wb") as f:
            f.write(all_tokens.tobytes())
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / \
            ((all_tokens == tokenizer.special_tokens['<bos>']).sum())

        # eg: Saved ./datas/datasets/pleisto/wikipedia-cn-20230720-filtered/chatGLM_64793_0.9.train.bin, average seqlen: 2839.82
        print(
            f"Saved {tokenized_filename[split]}, average seqlen: {avg_seq_len:.2f}")
    return


def tokenizer_process(item, tokenizer):
    inputs = tokenizer(item["completion"],
                       padding=True, truncation=True, add_special_tokens=False).input_ids
    input_ids = []
    for input_id in inputs:
        # add sp bpe BOS id to partition the input_ids, just like tinystories pretokenizer
        input_id.append(tokenizer.special_tokens['<bos>'])
        if len(input_id) > 1:  # likely
            input_ids.extend(input_id)

    return {"input_ids": input_ids}


def pretokenize_with_chatGLM(data_dir, tokenizer, batch_size=30, test_size=0.1, train_size=0.9):
    """
    pretokenize the dataset with chatGLM tokenizer , save tokenids bytes to file 
    """
    tokenized_filename = {}
    tokenized_filename["test"] = os.path.join(
        data_dir, f"chatGLM_{tokenizer.vocab_size}_{test_size}.test.bin")
    tokenized_filename["train"] = os.path.join(
        data_dir, f"chatGLM_{tokenizer.vocab_size}_{train_size}.train.bin")

    # ds = load_dataset(data_dir, split="train[:3]")
    ds = load_dataset(data_dir, split="train")
    print(f'dataset: {ds}')
    ds = ds.train_test_split(test_size=test_size, train_size=train_size)
    print(f'split train dataset [{test_size}:{train_size}]: {ds}')
    for split in ["train", "test"]:
        ds[split] = ds[split].map(tokenizer_process, batched=True,
                                  batch_size=batch_size, remove_columns=["source", "completion"], fn_kwargs={"tokenizer": tokenizer})
        print(f'after tokenizer process {split} dataset: {ds[split]}')
        # convert to uint16 nparray, note: tokenizer vocab_size must <= 2^16
        all_tokens = np.array(ds[split]["input_ids"], dtype=np.uint16)
        # write the bytes
        with open(tokenized_filename[split], "wb") as f:
            f.write(all_tokens.tobytes())
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / \
            ((all_tokens == tokenizer.special_tokens['<bos>']).sum())

        # eg: Saved ./datas/datasets/pleisto/wikipedia-cn-20230720-filtered/chatGLM_64793_0.9.train.bin, average seqlen: 2839.82
        print(
            f"Saved {tokenized_filename[split]}, average seqlen: {avg_seq_len:.2f}")
    return


if __name__ == "__main__":
    """
    These stages are designed to be run in order.
    python prepocess.py train_vocab --data_dir=./datas/datasets/pleisto/wikipedia-cn-20230720-filtered --vocab_size=64793
    python preprocess.py pretokenize_with_chatGLM3 --data_dir=./datas/datasets/pleisto/wikipedia-cn-20230720-filtered
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "train_vocab", "pretokenize", "pretokenize_with_chatGLM3", "print_tokenizer"])
    parser.add_argument("-vs", "--vocab_size", type=int, default=4096,
                        help="pretokenization vocab size")
    parser.add_argument("-dd", "--data_dir", type=str,
                        default="", help="dataset dir")
    parser.add_argument("-tm", "--tokenizer_model", type=str,
                        default="", help="tokenizer_model file(*.bin/*.model)")
    parser.add_argument("-tts", "--test_size", type=float,
                        default=0.1, help="dataset test size")
    parser.add_argument("-tns", "--train_size", type=float,
                        default=0.9, help="dataset train size")
    parser.add_argument("-bs", "--batch_size", type=int,
                        default=2, help="pretokenize batch size")
    args = parser.parse_args()
    print(f'args: {args}')

    # depending on the stage call the appropriate function
    if args.stage == "train_vocab":
        train_vocab(data_dir=args.data_dir, vocab_size=args.vocab_size)
    if args.stage == "pretokenize":
        pretokenize(
            args.data_dir, args.vocab_size, batch_size=args.batch_size, test_size=args.test_size, train_size=args.train_size)
    elif args.stage == "pretokenize_with_chatGLM3":
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm3-6b", trust_remote_code=True)
        pretokenize_with_chatGLM(
            args.data_dir, tokenizer, batch_size=args.batch_size, test_size=args.test_size, train_size=args.train_size)
    elif args.stage == "print_tokenizer":
        print_tokenizer(tokenizer_model=args.tokenizer_model)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
