import argparse
import os
import sys

from datasets import load_dataset
import sentencepiece as spm


def transform_conversation(example):
    conversation_text = example['text']
    segments = conversation_text.split('###')
    reformatted_segments = []
    # Iterate over pairs of segments
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace('Human:', '').strip()
        # Check if there is a corresponding assistant segment before processing
        if i + 1 < len(segments):
            assistant_text = segments[i + 1].strip().replace(
                'Assistant:', '').strip()
            # Apply the new template
            reformatted_segments.append(
                f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
        else:
            # Handle the case where there is no corresponding assistant segment
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] </s>')

    return {'text': ''.join(reformatted_segments)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--target_dataset_dir", type=str,
                        default="./datas/datasets/guanaco-llama2-1k", help="dataset dir")
    parser.add_argument("-hd", "--hf_repo_id", type=str,
                        default="weege007/guanaco-llama2-1k", help="hf repo dataset dir")
    args = parser.parse_args()
    print(f'args: {args}')

    # Load the dataset
    dataset = load_dataset('timdettmers/openassistant-guanaco')
    # Shuffle the dataset and slice it
    dataset = dataset['train'].shuffle(seed=42).select(range(1000))
    # Apply the transformation
    data = dataset.map(transform_conversation)
    print(data)
    # defualt arrow format
    data.save_to_disk(args.target_dataset_dir)
    if len(args.hf_repo_id) > 0:
        data.push_to_hub(args.hf_repo_id,  private=True)
    print("transform_conversation ok, save to", args.target_dataset_dir)
