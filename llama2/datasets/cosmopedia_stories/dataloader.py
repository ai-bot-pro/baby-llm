import os

from torch import from_numpy
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ChatGLMPretokSftDataset(Dataset):
    """
    Loads pretokenized examples from disk and yields them as PyTorch tensors.
    - sft dataset less than pre-training dataset, so just pretokenize don't need to save pretokenized data
    - u can choose custom tokenizer to encode sft dataset
    - u can choose sp bpe tokenizer to encode sft dataset, eg: chatglm(zh), llama2(en)
    """

    def __init__(self, data_dir, tokenizer, max_seq_len, prompt_max_len, text_max_len):
        super().__init__()
        self.data_dir = data_dir
        self.df = pd.read_csv(os.path.join(data_dir, 'sft_data.csv'))
        # like shuffle
        self.df = self.df.sample(frac=1.0)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prompt_max_len = prompt_max_len
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.bos = tokenizer.special_tokens['<bos>']
        self.eos = tokenizer.special_tokens['<eos>']
        # note: if use chatGLM tokenizer.special_tokens['<pad>'] is unkw_id, just use 0 as pad_id
        self.pad = 0

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        prompt = self.tokenizer.encode(
            sample['prompt_zh'], add_special_tokens=False)
        text = self.tokenizer.encode(
            sample['text_zh'], add_special_tokens=False)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len-2]
        if len(text) > self.text_max_len:
            text = text[:self.text_max_len-2]
        #
        input_id = prompt+[self.bos]+text+[self.eos]
        context_length = input_id.index(self.bos)
        mask_position = context_length - 1
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        loss_mask = [0]*context_length+[1] * \
            (len(input_id[mask_position+1:])) if pad_len == 0 else [0]*context_length+[1] * \
            (len(input_id[mask_position+1:-pad_len])) + [0]*pad_len
        #
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[:-1])
        return from_numpy(X), from_numpy(Y), from_numpy(loss_mask)
