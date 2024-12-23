import glob
import json
import os
import argparse
import sys

from tqdm import tqdm
import sentencepiece as spm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np

sys.path.append(os.path.split(sys.path[0])[0])
from _common.preprocess import merge_tokenizer, print_tokenizer
from _common.tokenizer import Tokenizer

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def train_vocab(data_dir, vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in data_dir/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"
    assert vocab_size <= (1 << 16), "Vocab size must less than 2^16"

    # output file prefix path for sentencepiece
    prefix = os.path.join(data_dir, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(data_dir, "tiny.txt")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    spm.SentencePieceTrainer.train(
        input=tiny_file,
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
        normalization_rule_name="identity",
    )

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def process_shard(args, data_dir, vocab_size, tokenizer_model=None):
    shard_id, shard = args
    tokenizer_model = (
        os.path.join(data_dir, f"tok{vocab_size}.model")
        if tokenizer_model is None
        else tokenizer_model
    )
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        # encode the text, use BOS
        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(data_dir, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(data_dir, vocab_size, tokenizer_model=None):
    # iterate the shards and tokenize all of them one by one
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(data_dir, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(
        process_shard, data_dir=data_dir, vocab_size=vocab_size, tokenizer_model=tokenizer_model
    )
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python preprocess.py train_vocab --vocab_size=2048 --data_dir=./datas
    python preprocess.py merge_tokenizer --data_dir=./datas --src_tokenizer_model=${src} --merge_tokenizer_model=${merge}
    python preprocess.py pretokenize --vocab_size=2048 --data_dir=./datas
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        type=str,
        choices=["train_vocab", "merge_tokenizer", "pretokenize", "print_tokenizer"],
    )
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size")
    parser.add_argument("--data_dir", type=str, default="./datas", help="process data dir")
    parser.add_argument(
        "--src_tokenizer_model", type=str, default=None, help="src tokenizer model file"
    )
    parser.add_argument(
        "--merge_tokenizer_model", type=str, default="", help="merge tokenizer model file"
    )
    parser.add_argument("--tokenizer_model", type=str, default="", help=" tokenizer model file")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "train_vocab":
        train_vocab(data_dir=args.data_dir, vocab_size=args.vocab_size)
    elif args.stage == "merge_tokenizer":
        merge_tokenizer(
            data_dir=args.data_dir,
            merge_tokenizer_model=args.merge_tokenizer_model,
            src_tokenizer_model=args.src_tokenizer_model,
        )
    elif args.stage == "pretokenize":
        pretokenize(
            data_dir=args.data_dir, vocab_size=args.vocab_size, tokenizer_model=args.tokenizer_model
        )
    elif args.stage == "print_tokenizer":
        print_tokenizer(tokenizer_model=args.tokenizer_model)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
