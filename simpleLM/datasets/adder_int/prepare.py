"""
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits correctly.
"""
import os
import pickle
import requests
import numpy as np
import torch


def encode_adder(data,ndigit):
    res = ""
    for idx in data:
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        res += render
    return res

# just supoort n-digit addition problems
ndigit = 2
num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
rng = torch.Generator()
rng.manual_seed(1337)
data = torch.randperm(num, generator=rng)

chars = ['0','1','2','3','4','5','6','7','8','9']
vocab_size = 10 #0...9
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
train_data = encode_adder(data[num_test:],ndigit)
val_data = encode_adder(data[:num_test],ndigit)

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# a,b,a+b, and +1 due to potential carry overflow,
# but then also -1 because very last digit doesn't ever plug back
# as there is no explicit <EOS> token to predict, it is implied
# if n=2; block_size = 3*n + 1 - 1 = 6 
block_size = 3*ndigit + 1 - 1

# save the meta information as well, to help us encode/decode later
meta = {
    'ndigit': ndigit,
    'vocab_size': vocab_size,
    'block_size': block_size,
    'itos': itos,
    'stoi': stoi,
}
print(f"meta: {meta}")
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

#vocab size: 10
#train has 66,500 tokens
#val has 3,500 tokens
#meta: {'ndigit': 2, 'vocab_size': 10, 'itos': {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}, 'stoi': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}}