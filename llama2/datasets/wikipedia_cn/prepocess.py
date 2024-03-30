from transformers import LlamaTokenizer
import argparse
from tqdm import tqdm
from datasets import load_dataset


def pretokenize_with_chatGLM2(data_dir, tokenizer_model=""):
    # iterate the shards and tokenize all of them one by one
    return


def print_tokenizer(tokenizer_model):
    return


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    python preprocess.py pretokenize_with_chatGLM2 --data_dir=./datas
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "pretokenize_with_chatGLM2", "print_tokenizer"])
    parser.add_argument("--tokenizer_model", type=str,
                        default="", help=" tokenizer model file")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "pretokenize_with_chatGLM2":
        pretokenize_with_chatGLM2(data_dir=args.data_dir,
                                  tokenizer_model=args.tokenizer_model)
    elif args.stage == "print_tokenizer":
        print_tokenizer(tokenizer_model=args.tokenizer_model)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
