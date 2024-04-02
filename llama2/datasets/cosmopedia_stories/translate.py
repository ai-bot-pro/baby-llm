# https://huggingface.co/learn/nlp-course/chapter5/
# https://zhuanlan.zhihu.com/p/564816807
# u can change this for M/R op

from deep_translator import GoogleTranslator
from datasets import load_dataset, load_from_disk
import argparse
import time
import random


def local_mt(item):
    inputs = tokenizer([item["text"], item["prompt"]], return_tensors="pt",
                       padding=True, truncation=True).input_ids
    translation = model.generate(
        inputs, max_new_tokens=512, do_sample=True, top_k=30, top_p=0.95)
    result = tokenizer.batch_decode(translation, skip_special_tokens=True)

    return {"text_zh": result[0], "prompt_zh": result[1]}


def remote_mt_batch(item):
    translated_prompt = ""
    translated_text = ""
    try_cn = 100
    while try_cn > 0:
        try:
            translated_prompt = GoogleTranslator(source='en', target='zh-CN').translate_batch(
                item["prompt"])
            translated_text = GoogleTranslator(source='en', target='zh-CN').translate_batch(
                item["text"])
            break
        except Exception as e:
            print("An error occurred:", e)
            time.sleep(1)
            try_cn -= 1

    return {"text_zh": translated_text, "prompt_zh": translated_prompt}


def batch_check_data(batch):
    return [len(item) != 0 for item in batch["text_zh"]]


def batch_filter_larg_text(batch):
    return [len(item) < 5000 for item in batch["text"]]


def batch_filter_larg_prompt(batch):
    return [len(item) < 5000 for item in batch["prompt"]]


def batch_filter_middle_text(batch):
    return [len(item) < 1024 for item in batch["text"]]


def batch_filter_middle_prompt(batch):
    return [len(item) < 1024 for item in batch["prompt"]]


def batch_filter_young_children(batch):
    return [item == "young_children" for item in batch["audience"]]


def translate2save(src_dataset_dir: str, target_dataset_dir: str, hf_repo_id="", sample_size=0, format=""):
    data = load_dataset(src_dataset_dir, split="train")
    print(data)
    data = data.filter(batch_filter_larg_text, batched=True)
    print(data)
    data = data.filter(batch_filter_larg_prompt, batched=True)
    print(data)
    data = data.filter(batch_filter_young_children, batched=True)
    print(data)
    if sample_size > 0:
        data = data.select(range(sample_size))
        print(data)
    data = data.map(remote_mt_batch, batched=True,
                    batch_size=3, remove_columns=[])
    print(data)
    if format == "csv":
        # use pandas DF
        data = data.to_pandas()
        data.to_csv(target_dataset_dir)
    elif format == "json":
        data = data.to_pandas()
        data.to_json(target_dataset_dir)
    elif format == "parquet":
        data = data.to_pandas()
        data.to_parquet(target_dataset_dir)
    else:
        # defualt arrow format
        data.save_to_disk(target_dataset_dir)
    if len(hf_repo_id) > 0:
        data.push_to_hub(hf_repo_id)
    print("translate ok, save to", target_dataset_dir)


def load2check(dataset_dir):
    data = load_from_disk(dataset_dir)
    print("filter before", data)
    data = data.filter(batch_check_data, batched=True)
    print("filter after", data)
    index = random.randint(0, data.num_rows)
    print("\n----sample prompt_zh-----\n", data["prompt_zh"]
          [index])
    print("\n----sample text_zh-----\n", data["text_zh"]
          [index])


def local_translate2save(src_dataset_dir: str, target_dataset_dir: str):
    data = load_dataset(src_dataset_dir, split="train")
    print(data)
    data = data.filter(batch_filter_middle_text, batched=True)
    print(data)
    data = data.filter(batch_filter_middle_prompt, batched=True)
    print(data)
    data = data.map(local_mt, batched=False)
    print(data)
    data.save_to_disk(target_dataset_dir)
    print("local translate ok, save to", target_dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[
                        "translate", "check", "local_mt"])
    parser.add_argument("-s", "--src_dir", type=str,
                        help="path to src dataset dir ")
    parser.add_argument("-t", "--target_dir", type=str,
                        help="path to target dataset dir ", required=lambda x: x is not None)
    parser.add_argument("-r", "--hf_repo_id", type=str, default="",
                        help="target dataset huggingface repo id")
    parser.add_argument("-ss", "--sample_size", type=int, default=1000,
                        help="dataset sample size")
    parser.add_argument("-ff", "--file_format", type=str, default="",
                        help="target dataset file format:[csv,json,parquet]")
    args = parser.parse_args()

    if args.stage == "translate":
        translate2save(args.src_dir, args.target_dir,
                       hf_repo_id=args.hf_repo_id, sample_size=args.sample_size, format=args.file_format)
    elif args.stage == "check":
        load2check(args.target_dir)
    elif args.stage == "local_mt":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        # model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
        model_checkpoint = "weege007/opus-mt-en-zh-finetuned-en-to-zh"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        local_translate2save(args.target_dir)
