# Datasets

This directory contains information about datasets used throughout the NLP modules.

## Datasets Used in This Repository

| Dataset | Module | Task | Size | Source |
|---------|--------|------|------|--------|
| IMDB Reviews | 09, 26 | Sentiment Analysis | 50K reviews | HuggingFace |
| CoNLL-2003 | 10, 27 | Named Entity Recognition | 22K sentences | HuggingFace |
| Penn Treebank | 11 | Language Modeling | 1M words | torchtext |
| Multi30k | 12, 13 | Machine Translation | 30K pairs | torchtext |
| SQuAD | 17 | Question Answering | 100K+ QA pairs | HuggingFace |
| AG News | 02, 09 | Text Classification | 120K articles | torchtext |

## Download Instructions

Most datasets are downloaded automatically using HuggingFace Datasets or torchtext:

```python
# HuggingFace Datasets
from datasets import load_dataset
dataset = load_dataset("imdb")

# torchtext
from torchtext.datasets import IMDB
train_data, test_data = IMDB(split=('train', 'test'))
```

## Custom Datasets

For production projects (Modules 26-28), you may need to prepare custom datasets.
See individual module notebooks for data preparation guidelines.

## Data Size Considerations

- **Small datasets (<10K samples)**: Can run on CPU
- **Medium datasets (10K-100K)**: GPU recommended
- **Large datasets (>100K)**: GPU required, consider data streaming

## Caching

Downloaded datasets are cached in:
- HuggingFace: `~/.cache/huggingface/datasets/`
- torchtext: `~/.cache/torchtext/`
