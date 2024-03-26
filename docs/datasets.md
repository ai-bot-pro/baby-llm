# Datasets-Tiny-Series
 https://huggingface.co/collections/nampdn-ai/tiny-series-6503910fd491144159519c70

探索使用这些微小的数据宝藏构建小型语言模型的可能性和局限性！

- [TinyStories](https://arxiv.org/abs/2305.07759): 引发我对微型系列探索之旅兴趣的论文。
- [tiny-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks): 42万个合成的“互联网事物”教科书。
- [tiny-orca-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-orca-textbooks): 合成的教科书，帮助模型在上下文中学习如何正确执行任务的正确方式。
- [tiny-webtext](https://huggingface.co/datasets/nampdn-ai/tiny-webtext): 6GB（450万条记录）的多样化网络文本，注入了批判性思维方法，制作了一个无偏见的英语数据集。
- [tiny-lessons](https://huggingface.co/datasets/nampdn-ai/tiny-lessons): [tiny-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks) 数据集的子集，以适合教科书的Markdown格式提供了关于“互联网事物”的各种课程。
- [tiny-bridgedict](https://huggingface.co/datasets/nampdn-ai/tiny-bridgedict): 一个在英语、越南语和中文之间建立知识链接和传递的微型多语言模型数据集。

## Datasets-TinyStories

- download from hg roneneldan/TinyStories TinyStories datasets 
  paper: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/pdf/2305.07759.pdf)
```shell
# u can use huggingface proxy to download
# roneneldan/TinyStories
huggingface-cli download \
    --repo-type dataset roneneldan/TinyStories TinyStories_all_data.tar.gz \
    --local-dir ${data_dir}/roneneldan/TinyStories \
    --local-dir-use-symlinks False

# 52AI/TinyStoriesZh (use https://github.com/nidhaloff/deep-translator translate TinyStories datasets)
huggingface-cli download \
  --repo-type dataset 52AI/TinyStoriesZh \
  --local-dir ${data_dir}/52AI/TinyStoriesZh \
  --local-dir-use-symlinks False
```

Example story:
```json
{
  "story": "\n\nLily and Ben are friends. They like to play in the park. One day, they see a big tree with a swing. Lily wants to try the swing. She runs to the tree and climbs on the swing.\n\"Push me, Ben!\" she says. Ben pushes her gently. Lily feels happy. She swings higher and higher. She laughs and shouts.\nBen watches Lily. He thinks she is cute. He wants to swing too. He waits for Lily to stop. But Lily does not stop. She swings faster and faster. She is having too much fun.\n\"Can I swing too, Lily?\" Ben asks. Lily does not hear him. She is too busy swinging. Ben feels sad. He walks away.\nLily swings so high that she loses her grip. She falls off the swing. She lands on the ground. She hurts her foot. She cries.\n\"Ow, ow, ow!\" she says. She looks for Ben. She wants him to help her. But Ben is not there. He is gone.\nLily feels sorry. She wishes she had shared the swing with Ben. She wishes he was there to hug her. She limps to the tree. She sees something hanging from a branch. It is Ben's hat. He left it for her.\nLily smiles. She thinks Ben is nice. She puts on his hat. She hopes he will come back. She wants to say sorry. She wants to be friends again.",
  "instruction": {
    "prompt:": "Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would understand. The story should use the verb \"hang\", the noun \"foot\" and the adjective \"cute\". The story has the following features: the story should contain at least one dialogue. Remember to only use simple words!\n\nPossible story:",
    "words": ["hang", "foot", "cute"],
    "features": ["Dialogue"]
  },
  "summary": "Lily and Ben play in the park and Lily gets too caught up in swinging, causing Ben to leave. Lily falls off the swing and hurts herself, but Ben leaves his hat for her as a kind gesture.",
  "source": "GPT-4"
}
```

# 其他具有教科书般质量的小型高质量数据集

- [devdocs.io](https://huggingface.co/datasets/nampdn-ai/devdocs.io): FreeCodeCamp提供了189,000份涵盖广泛技术栈和编程语言的全面API文档。
- [sciphi-python-textbook](https://huggingface.co/datasets/emrgnt-cmplxty/sciphi-python-textbook)
- [textbook_quality_programming](https://huggingface.co/datasets/vikp/textbook_quality_programming)
- [sciphi-textbooks-are-all-you-need](https://huggingface.co/datasets/emrgnt-cmplxty/sciphi-textbooks-are-all-you-need)

# 其他数据集
- OpenWebText (GPT2 webtext)
- Common Crawl (GPT3 datasets)
- 维基百科: [英文](https://dumps.wikimedia.org/enwiki/) | [中文](https://dumps.wikimedia.org/zhwiki/)
- [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia): 该数据集由 Mixtral-8x7B-Instruct-v0.1 生成的综合教科书、博客文章、故事、帖子和 WikiHow 文章组成。该数据集包含超过 3000 万个文件和 250 亿个token，这使其成为迄今为止最大的开放综合数据集。英文数据集，主要关注教科书和故事。