## datasets
preprocess datasets, to train and inference better

- train sp bpe tokenizer and vocabulary for each sub dataset
- merge sp bpe tokenizer and vocabulary
- tokenize datasets(train/valid/test) for each sub datasets
- merge sub datasets(train/valid/test)
- export pb tokenizer model to binary file for LM training and inference if use sentencepiece