# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import struct
import argparse
import warnings
from typing import List

from sentencepiece import SentencePieceProcessor
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


def write_tokenizer_to_hf(tokenizer_path, input_tokenizer_path):
    os.makedirs(tokenizer_path, exist_ok=True)
    # Initialize the tokenizer based on the `spm` model, the same as llama tokenizer
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"{input_tokenizer_path} tokenizer has been saved to hf tokenizer {tokenizer_path}")


class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.unk_id: int = self.sp_model.unk_id()
        # print(self.string())
        assert self.sp_model.vocab_size() == self.sp_model.piece_size()

    def string(self):
        str_format = f"vocab_size: {self.n_words}\nBOS ID: {self.bos_id}\nEOS ID: {self.eos_id}\nPAD ID: {self.pad_id}\nUNK ID: {self.unk_id}\n"
        return str_format

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    # https://github.com/google/sentencepiece/issues/486#issuecomment-1207354033
    def batch_encode(self, sl: List[str], bos: bool, eos: bool) -> List[int]:
        res = []
        ts = self.sp_model.encode(sl)
        for t in ts:
            if bos:
                t = [self.bos_id] + t
            if eos:
                t = t + [self.eos_id]
            res.extend(t)
        return res

    def export(self):
        """
        export pb tokenizer model to binary file for LM training and inference if use sentencepiece
        """

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            # sentencepiece uses this character as whitespace
            t = t.replace('▁', ' ')
            b = t.encode('utf-8')  # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "export_to_hf", "export"])
    parser.add_argument("-t", "--tokenizer-model", type=str,
                        help="optional path to custom tokenizer ")
    parser.add_argument("-o", "--output_hf_dir", type=str,
                        help="output hf tokenizer dir ")
    args = parser.parse_args()

    if args.stage=="export":
        t = Tokenizer(args.tokenizer_model)
        t.export()
    elif args.stage == "export_to_hf":
        output_hf_dir = args.tokenizer_model.replace('.model', '')
        write_tokenizer_to_hf(output_hf_dir, args.tokenizer_model)
    else:
        raise ValueError(f"Unknown stage {args.stage}")