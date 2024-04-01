# chatGLM use the sp bpe tokenizer like llama2 tokenizer
# Qwen use the tiktoken bpe tokenizer like gpt tokenizer

from transformers import AutoTokenizer
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


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


def print_tokenizer(tokenizer_model):
    return


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    python preprocess.py pretokenize_with_chatGLM3 --data_dir=./datas/datasets/pleisto/wikipedia-cn-20230720-filtered
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "pretokenize_with_chatGLM3", "print_tokenizer"])
    parser.add_argument("--data_dir", type=str,
                        default="", help="dataset dir")
    parser.add_argument("--tokenizer_model", type=str,
                        default="", help="tokenizer_model file(*.bin/*.model)")
    parser.add_argument("--tokenized_filename", type=str,
                        default="./datas/datasets/pleisto/wikipedia-cn-20230720-filtered.bin", help="dataset dir")
    parser.add_argument("--batch_size", type=int,
                        default=2, help="pretokenize batch size")
    args = parser.parse_args()
    print(f'args: {args}')

    # depending on the stage call the appropriate function
    if args.stage == "pretokenize_with_chatGLM3":
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm3-6b", trust_remote_code=True)
        pretokenize_with_chatGLM(
            args.data_dir, tokenizer, args.tokenized_filename, batch_size=args.batch_size)
    elif args.stage == "print_tokenizer":
        print_tokenizer(tokenizer_model=args.tokenizer_model)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
