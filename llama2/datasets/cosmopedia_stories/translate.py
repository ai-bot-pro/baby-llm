from datasets import load_dataset

data = load_dataset(
    "./datas/dataset/HuggingFaceTB/cosmopedia", split="train[:10]")

# todo: batch translate HuggingFaceTB/cosmopedia stories 1~10% per day (1-4 train parquet file)
