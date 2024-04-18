# llama2 use the sp bpe tokenizer
# gpt use the tiktoken bpe tokenizer
from _common.preprocess import print_tokenizer
from _common.tokenizer import Tokenizer
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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def train_vocab(data_dir, vocab_size):
    """
    train the tokenizer.model and vocab from the dataset
    """
    assert vocab_size > 0, "Vocab size must be positive"
    assert vocab_size < (1 << 16), "Vocab size must less than 2^16"

    # output file prefix path for sentencepiece
    prefix = os.path.join(data_dir, f"tok{vocab_size}")

    all_file = os.path.join(data_dir, "tinyshakespeare.txt")
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


def custom_tokenizer_process(sentences, tokenizer):
    token_ids = []
    for sentence in sentences:
        token_ids += tokenizer.encode(sentence, bos=True, eos=False)
    return token_ids


def pretokenize(data_dir, vocab_size, tokenizer_model=None, train_size=0.9):
    """
    pretokenize the dataset with custom tokenizer , save tokenids bytes to file 
    need rm *.{test,train}.bin
    """
    tokenizer_model = os.path.join(
        data_dir, f"tok{vocab_size}.model") if tokenizer_model is None else tokenizer_model
    tokenizer = Tokenizer(tokenizer_model)
    print(tokenizer.string())

    # save .bin files into a new tok{N} directory
    tokenized_filename = {}
    tokenized_filename["test"] = os.path.join(
        data_dir, f"tok{vocab_size}_{1-train_size:.1f}.test.bin")
    tokenized_filename["train"] = os.path.join(
        data_dir, f"tok{vocab_size}_{train_size}.train.bin")

    input_file_path = os.path.join(data_dir, "tinyshakespeare.txt")
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")
    split_data = data.split("\n\n")
    print(split_data[:4])
    sentence_len = len(split_data)
    print(f"length of dataset in sentences: {sentence_len:,}")
    ds = {}
    ds["train"] = split_data[:int(sentence_len*train_size)]
    ds["test"] = split_data[int(sentence_len*train_size):]

    print(
        f'split train dataset [train:test] [{len(ds["train"]):,}:{len(ds["test"]):,}]')
    for split in ["train", "test"]:
        token_ids = custom_tokenizer_process(ds[split], tokenizer)
        print(
            f'after tokenizer process {split} token_ids len: {len(token_ids):,}')
        # convert to uint16 nparray, note: tokenizer vocab_size must <= 2^16
        all_tokens = np.array(token_ids, dtype=np.uint16)
        # write the bytes
        with open(tokenized_filename[split], "wb") as f:
            f.write(all_tokens.tobytes())
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / \
            ((all_tokens == tokenizer.bos_id).sum())

        # eg: Saved ./datas/pleisto/wikipedia-cn-20230720-filtered/tok18899_0.9.train.bin, average seqlen: 751.24
        print(
            f"Saved {tokenized_filename[split]}, average seqlen: {avg_seq_len:.2f}")
    return


def tokenizer_process(sentences, tokenizer):
    input_ids = []
    for sentence in sentences:
        inputs = tokenizer(sentence, padding=True, truncation=True,
                           add_special_tokens=False).input_ids
        for input_id in inputs:
            # add sp bpe BOS id to partition the input_ids, just like tinystories pretokenizer
            input_id.append(tokenizer.special_tokens['<bos>'])
            if len(input_id) > 1:  # likely
                input_ids.extend(input_id)

    return input_ids


def pretokenize_with_llama2(data_dir, tokenizer, train_size=0.9):
    """
    pretokenize the dataset with llama2 tokenizer , save tokenids bytes to file 
    need rm *.{test,train}.bin
    """
    tokenized_filename = {}
    tokenized_filename["test"] = os.path.join(
        data_dir, f"llama2_{tokenizer.vocab_size}_{1-train_size:.1f}.test.bin")
    tokenized_filename["train"] = os.path.join(
        data_dir, f"llama2_{tokenizer.vocab_size}_{train_size}.train.bin")

    input_file_path = os.path.join(data_dir, "tinyshakespeare.txt")
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")
    split_data = data.split("\n\n")
    print(split_data[:4])
    sentence_len = len(split_data)
    print(f"length of dataset in sentences: {sentence_len:,}")
    ds = {}
    ds["train"] = split_data[:int(sentence_len*train_size)]
    ds["test"] = split_data[int(sentence_len*train_size):]

    print(
        f'split train dataset [train:test] [{len(ds["train"]):,}:{len(ds["test"]):,}]')
    for split in ["train", "test"]:
        token_ids = tokenizer_process(ds[split], tokenizer)
        print(
            f'after tokenizer process {split} token_ids len: {len(token_ids):,}')
        # convert to uint16 nparray, note: tokenizer vocab_size must <= 2^16
        all_tokens = np.array(token_ids, dtype=np.uint16)
        # write the bytes
        with open(tokenized_filename[split], "wb") as f:
            f.write(all_tokens.tobytes())
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / \
            ((all_tokens == tokenizer.bos_id).sum())

        # eg: Saved ./datas/pleisto/wikipedia-cn-20230720-filtered/tok18899_0.9.train.bin, average seqlen: 751.24
        print(
            f"Saved {tokenized_filename[split]}, average seqlen: {avg_seq_len:.2f}")
    return


if __name__ == "__main__":
    """
    These stages are designed to be run in order.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "train_vocab", "pretokenize",
                        "pretokenize_with_llama2", "print_tokenizer"])
    parser.add_argument("-vs", "--vocab_size", type=int, default=4096,
                        help="pretokenization vocab size")
    parser.add_argument("-dd", "--data_dir", type=str,
                        default="", help="dataset dir")
    parser.add_argument("-tm", "--tokenizer_model", type=str,
                        default=None, help="tokenizer_model file(*.bin/*.model)")
    parser.add_argument("-tns", "--train_size", type=float,
                        default=0.9, help="dataset train size")
    args = parser.parse_args()
    print(f'args: {args}')

    # depending on the stage call the appropriate function
    if args.stage == "train_vocab":
        train_vocab(data_dir=args.data_dir, vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(
            args.data_dir, args.vocab_size,
            tokenizer_model=args.tokenizer_model,
            train_size=args.train_size)
    elif args.stage == "pretokenize_with_llama2":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", trust_remote_code=True)
        pretokenize_with_llama2(
            args.data_dir, tokenizer, batch_size=args.batch_size, test_size=args.test_size, train_size=args.train_size)
    elif args.stage == "print_tokenizer":
        print_tokenizer(tokenizer_model=args.tokenizer_model)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
