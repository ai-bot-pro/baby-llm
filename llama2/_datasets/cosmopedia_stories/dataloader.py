
from torch import from_numpy
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd


class ChatGLMPretokSftDataset(Dataset):
    """
    Loads pretokenized examples from disk and yields them as PyTorch tensors.
    - sft dataset less than pre-training dataset, so just pretokenize don't need to save pretokenized data
    - u can choose custom tokenizer to encode sft dataset
    - u can choose sp bpe tokenizer to encode sft dataset, eg: chatglm(zh), llama2(en)
    """

    def __init__(
        self, csv_file_path, max_seq_len=512, prompt_max_len=256, text_max_len=256, split="train"
    ):
        super().__init__()
        self.split = split
        self.df = pd.read_csv(csv_file_path)
        # like shuffle
        self.df = self.df.sample(frac=1.0)
        self.max_seq_len = max_seq_len
        self.prompt_max_len = prompt_max_len
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        self.bos_id = self.tokenizer.special_tokens["<bos>"]  # 1
        self.eos_id = self.tokenizer.special_tokens["<eos>"]  # 2
        # note: if use chatGLM tokenizer.special_tokens['<pad>'] is unkw_id, just use -1 as pad_id from sp
        self.pad_id = -1

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        prompt = self.tokenizer.encode(sample["prompt_zh"], add_special_tokens=False)
        text = self.tokenizer.encode(sample["text_zh"], add_special_tokens=False)
        print(len(prompt), len(text))
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[: self.prompt_max_len - 2]
        if len(text) > self.text_max_len:
            text = text[: self.text_max_len - 2]
        print(len(prompt), len(text))

        input_id = prompt + [self.bos_id] + text + [self.eos_id]
        context_length = input_id.index(self.bos_id)
        mask_position = context_length - 1
        pad_len = self.max_seq_len - len(input_id)
        input_id = input_id + [self.pad_id] * pad_len
        loss_mask = (
            [self.pad_id] * context_length + [1] * (len(input_id[mask_position + 1 :]))
            if pad_len == 0
            else [self.pad_id] * context_length
            + [1] * (len(input_id[mask_position + 1 : -pad_len]))
            + [self.pad_id] * pad_len
        )

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[:-1])
        print(X.shape, Y.shape, len(input_id))
        return from_numpy(X), from_numpy(Y), from_numpy(loss_mask)


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["check"])
    parser.add_argument("-f", "--csv_file_path", type=str, help="csv file path ")
    args = parser.parse_args()
    print(args)

    ds = ChatGLMPretokSftDataset(csv_file_path=args.csv_file_path)
    dl = DataLoader(
        ds, batch_size=2, drop_last=False, shuffle=False, pin_memory=True, num_workers=0
    )
    for step, (X, Y, loss_mask) in enumerate(dl):
        print(step)
        print(X.shape, Y.shape)
        print(X[0])
        print(Y[0])
        print(loss_mask[0])
        break
