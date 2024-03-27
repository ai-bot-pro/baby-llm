# https://huggingface.co/learn/nlp-course/chapter5/
# https://zhuanlan.zhihu.com/p/564816807

from deep_translator import GoogleTranslator
from datasets import load_dataset, load_from_disk
import argparse
import time


def translate_en2cn(item):
    translated = ""
    try_cn = 100
    while try_cn > 0:
        try:
            translated_prompt = GoogleTranslator(source='en', target='zh-CN').translate(
                item["prompt"])
            translated_text = GoogleTranslator(source='en', target='zh-CN').translate(
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


def translate2save(src_dataset_dir: str, target_dataset_dir: str):
    data = load_dataset(src_dataset_dir, split="train")
    print(data)
    data = data.filter(batch_filter_larg_text, batched=True)
    print(data)
    data = data.filter(batch_filter_larg_prompt, batched=True)
    print(data)
    data = data.map(translate_en2cn, batched=False)
    print(data)
    data.save_to_disk(target_dataset_dir)
    print("translate ok, save to", target_dataset_dir)


def load2check(dataset_dir):
    data = load_from_disk(dataset_dir)
    print("filter before", data)
    data = data.filter(batch_check_data, batched=True)
    print("filter after", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["translate", "check"])
    parser.add_argument("-s", "--src_dir", type=str,
                        help="path to src dataset dir ")
    parser.add_argument("-t", "--target_dir", type=str,
                        help="path to target dataset dir ", required=lambda x: x is not None)
    args = parser.parse_args()

    if args.stage == "translate":
        translate2save(args.src_dir, args.target_dir)
    elif args.stage == "check":
        load2check(args.target_dir)
