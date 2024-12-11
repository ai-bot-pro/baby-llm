import os
import argparse

from huggingface_hub import HfApi

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ["CURL_CA_BUNDLE"] = ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["upload_file"])
    parser.add_argument(
        "-rt", "--repo-type", type=str, choices=["dataset", "model"], help="huggingface repo id "
    )
    parser.add_argument("-r", "--repo_id", type=str, help="huggingface repo id ")
    parser.add_argument("-f", "--path_or_fileobj", type=str, help="path to src dataset ")
    parser.add_argument("-p", "--path_in_repo", type=str, help="path to hf dataset dir ")
    args = parser.parse_args()
    print(args)
    print(
        os.environ["CURL_CA_BUNDLE"],
    )

    api = HfApi()
    if args.stage == "upload_file":
        api.upload_file(
            repo_id=args.repo_id,
            path_or_fileobj=args.path_or_fileobj,
            path_in_repo=args.path_in_repo,
            token=os.environ["HF_TOKEN"],
            repo_type=args.repo_type,
        )
    else:
        raise ValueError("stage not supported")
