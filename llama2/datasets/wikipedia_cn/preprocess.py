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

from datasets.preprocess import print_tokenizer

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


def pretokenize_with_chatGLM(data_dir, tokenizer, tokenized_filename, batch_size=30):
    """
    pretokenize the dataset with chatGLM tokenizer , save tokenids bytes to file 
    """
    # ds = load_dataset(data_dir, split="train[:3]")
    ds = load_dataset(data_dir, split="train")
    print(f'dataset: {ds}')
    ds = ds.map(tokenizer_process, batched=True,
                batch_size=batch_size, remove_columns=["source", "completion"], fn_kwargs={"tokenizer": tokenizer})
    print(f'after tokenizer process dataset: {ds}')
    # convert to uint16 nparray, note: tokenizer vocab_size must <= 2^16
    all_tokens = np.array(ds["input_ids"], dtype=np.uint16)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / \
        ((all_tokens == tokenizer.special_tokens['<bos>']).sum())

    # Saved ./datas/datasets/pleisto/wikipedia-cn-20230720-filtered.bin, average seqlen: 2839.82
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")
    return


if __name__ == "__main__":
    """
    These stages are designed to be run in order.
    python prepocess.py train_vocab --data_dir=./datas/datasets/pleisto/wikipedia-cn-20230720-filtered --vocab_size=64793
    python preprocess.py pretokenize_with_chatGLM3 --data_dir=./datas/datasets/pleisto/wikipedia-cn-20230720-filtered
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "train_vocab", "pretokenize_with_chatGLM3", "print_tokenizer"])
    parser.add_argument("-vs", "--vocab_size", type=int, default=4096,
                        help="pretokenization vocab size")
    parser.add_argument("-dd", "--data_dir", type=str,
                        default="", help="dataset dir")
    parser.add_argument("-tm", "--tokenizer_model", type=str,
                        default="", help="tokenizer_model file(*.bin/*.model)")
    parser.add_argument("-tf", "--tokenized_filename", type=str,
                        default="./datas/datasets/pleisto/wikipedia-cn-20230720-filtered.bin", help="dataset dir")
    parser.add_argument("-bs", "--batch_size", type=int,
                        default=2, help="pretokenize batch size")
    args = parser.parse_args()
    print(f'args: {args}')

    # depending on the stage call the appropriate function
    if args.stage == "train_vocab":
        train_vocab(data_dir=args.data_dir, vocab_size=args.vocab_size)
    elif args.stage == "pretokenize_with_chatGLM3":
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm3-6b", trust_remote_code=True)
        pretokenize_with_chatGLM(
            args.data_dir, tokenizer, args.tokenized_filename, batch_size=args.batch_size)
    elif args.stage == "print_tokenizer":
        print_tokenizer(tokenizer_model=args.tokenizer_model)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
