r"""
base datasets preprocess

meta llama2 convert to hf:
 https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
"""
import os
import argparse
import warnings


import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer

# 是用 fast tokenizer 会多出一个tokenizer.json 用于加速定位偏移加载
try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None
#LlamaTokenizerFast = None

from huggingface_hub import upload_file, upload_folder

import sys
sys.path.append(os.path.split(sys.path[0])[0])
from _common.tokenizer import Tokenizer

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def merge_tokenizer(data_dir, merge_tokenizer_model,
                    src_tokenizer_model="meta-llama/Llama-2-7b-hf",
                    src_from="llama2", prefix_name="new",
                    push_to_hub=False, hf_path="", repo_id=""):
    if src_from == 'llama2':
        merge_tokenizer_from_llama2(
            data_dir, merge_tokenizer_model, src_tokenizer_model)
    elif src_from == 'custom':
        merge_tokenizer_from_custom(
            data_dir, merge_tokenizer_model, src_tokenizer_model, prefix_name=prefix_name)
    else:
        raise ValueError(f"src_from:{src_from} not supported")

    output_hf_dir = os.path.join(data_dir, 'merged_tokenizer_hf')
    if push_to_hub:
        upload_folder(
            folder_path=output_hf_dir,
            path_in_repo=hf_path,
            repo_id=repo_id,
        )


def write_tokenizer_to_hf(tokenizer_path, input_tokenizer_path):
    # Initialize the tokenizer based on the `spm` model, the same as llama tokenizer
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"{input_tokenizer_path} tokenizer has been saved to hf tokenizer {tokenizer_path}")


def merge_tokenizer_from_custom(data_dir, merge_tokenizer_model, src_tokenizer_model, prefix_name="new"):
    src_sp_model = spm.SentencePieceProcessor()
    src_sp_model.Load(src_tokenizer_model)
    src_spm = sp_pb2_model.ModelProto()
    src_spm.ParseFromString(src_sp_model.serialized_model_proto())

    merge_sp_model = spm.SentencePieceProcessor()
    merge_sp_model.Load(merge_tokenizer_model)
    merge_spm = sp_pb2_model.ModelProto()
    merge_spm.ParseFromString(merge_sp_model.serialized_model_proto())

    src_spm_tokens_set = set(p.piece for p in src_spm.pieces)
    print(f"before src_tokens:{len(src_spm_tokens_set)}")
    for p in merge_spm.pieces:
        piece = p.piece
        if piece not in src_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            src_spm.pieces.append(new_p)
    print(f"New model pieces: {len(src_spm.pieces)}")

    # 保存合并后的模型(pb序列化)
    output_sp_dir = os.path.join(data_dir, 'merged_tokenizer_sp')
    os.makedirs(output_sp_dir, exist_ok=True)
    tokenizer_vocab_model_file = output_sp_dir + \
        f'/{prefix_name}_tokenizer.model'
    with open(tokenizer_vocab_model_file, 'wb') as f:
        f.write(src_spm.SerializeToString())
        print(
            f"{src_tokenizer_model} merge {merge_tokenizer_model} tokenizer has been saved to {tokenizer_vocab_model_file}")

    print_tokenizer(tokenizer_vocab_model_file)

    # 保存hf tokenizer格式
    output_hf_dir = os.path.join(data_dir, 'merged_tokenizer_hf')
    write_tokenizer_to_hf(output_hf_dir, tokenizer_vocab_model_file)


def merge_tokenizer_from_llama2(data_dir, merge_tokenizer_model, src_tokenizer_model="meta-llama/Llama-2-7b-hf"):
    """
    merge tokenizer sp pb2 format model, save to huggingface format model
    """
    llama_tokenizer = LlamaTokenizer.from_pretrained(src_tokenizer_model)
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(
        llama_tokenizer.sp_model.serialized_model_proto())

    merge_sp_model = spm.SentencePieceProcessor()
    merge_sp_model.Load(merge_tokenizer_model)
    merge_spm = sp_pb2_model.ModelProto()
    merge_spm.ParseFromString(merge_sp_model.serialized_model_proto())

    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    print(f"before src_tokens:{len(llama_spm_tokens_set)}")
    for p in merge_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
    print(f"New model pieces: {len(llama_spm.pieces)}")

    # 保存合并后的模型(pb序列化)
    output_sp_dir = os.path.join(data_dir, 'merged_tokenizer_sp')
    os.makedirs(output_sp_dir, exist_ok=True)
    tokenizer_vocab_model_file = output_sp_dir+'/new_llama_tokenizer.model'
    with open(tokenizer_vocab_model_file, 'wb') as f:
        f.write(llama_spm.SerializeToString())
        print(
            f"{src_tokenizer_model} merge {merge_tokenizer_model} tokenizer has been saved to {tokenizer_vocab_model_file}")

    print_tokenizer(tokenizer_vocab_model_file)

    # 保存hf tokenizer格式
    output_hf_dir = os.path.join(data_dir, 'merged_tokenizer_hf')
    write_tokenizer_to_hf(output_hf_dir, tokenizer_vocab_model_file)


def print_tokenizer(tokenizer_model):
    """
    just print trained tokenizer meta spec, don't print merged tokenizer
    """
    mp = sp_pb2_model.ModelProto()
    mp.ParseFromString(open(tokenizer_model, "rb").read())
    print(f"src trainer_spec:\n{mp.trainer_spec}")
    print(f"src normalizer_spec:\n{mp.normalizer_spec}")

    # print tokenizer
    tokenizer = Tokenizer(tokenizer_model)
    print(tokenizer.string())


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python preprocess.py merge_tokenizer --data_dir=./datas --src_tokenizer_model=${src} --merge_tokenizer_model=${merge}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "merge_tokenizer", "print_tokenizer"])
    parser.add_argument("--data_dir", type=str,
                        default="./datas", help="process data dir")
    parser.add_argument("--src_tokenizer_model", type=str,
                        default="meta-llama/Llama-2-7b-hf", help="src tokenizer model file")
    parser.add_argument("--src_from", type=str,
                        default="llama2", help="src tokenizer from")
    parser.add_argument("--merge_tokenizer_model", type=str,
                        default="", help="merge tokenizer model file")
    parser.add_argument("--prefix_name", type=str,
                        default="new", help="merged file prefix name")
    parser.add_argument("--tokenizer_model", type=str,
                        default="", help="tokenizer model file")
    parser.add_argument("--push_to_hub", type=bool,
                        help="Whether or not to push the model to the hub",
                        default=False)
    parser.add_argument("--hf_path", type=str, help="huggingface model path")
    parser.add_argument("--repo_id", type=str, help="huggingface repo id")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "merge_tokenizer":
        merge_tokenizer(data_dir=args.data_dir,
                        merge_tokenizer_model=args.merge_tokenizer_model,
                        src_tokenizer_model=args.src_tokenizer_model,
                        src_from=args.src_from,
                        prefix_name=args.prefix_name,
                        push_to_hub=args.push_to_hub,
                        hf_path=args.hf_path,
                        repo_id=args.repo_id)
    elif args.stage == "print_tokenizer":
        print_tokenizer(tokenizer_model=args.tokenizer_model)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
