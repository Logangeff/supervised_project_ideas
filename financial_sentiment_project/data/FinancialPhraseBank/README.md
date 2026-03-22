---
annotations_creators:
- expert-annotated
language_creators:
- found
language:
- en
license:
- cc-by-nc-sa-4.0
multilinguality:
- monolingual
size_categories:
- 1k<n<10k
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- sentiment-classification
paperswithcode_id: financial-phrasebank
pretty_name: Financial PhraseBank
dataset_info:
  features:
  - name: sentiment
    dtype: string
  - name: sentence
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': negative
          '1': neutral
          '2': positive
  splits:
  - name: train
    num_bytes: 586208
    num_examples: 3872
  - name: validation
    num_bytes: 73996
    num_examples: 484
  - name: test
    num_bytes: 73088
    num_examples: 484
  download_size: 417897
  dataset_size: 733292
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Dataset Card for Financial PhraseBank

## Dataset Description

**Repository:** [Link to the source, e.g., on Kaggle or original paper's site]
**Paper:** [Good debt or bad debt: Detecting semantic orientations in economic texts](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.23062)

This dataset (FinancialPhraseBank) contains the sentiments for 4846 financial news headlines from the perspective of a retail investor. The dataset is labeled with "negative", "neutral", or "positive" sentiments.

## Content

The dataset contains two columns:
*   `sentiment`: The sentiment label (negative, neutral, or positive).
*   `sentence`: The news headline text.

## Intended Uses

This dataset is primarily intended for training and evaluating sentiment analysis models, specifically in the financial domain. It can be used for:
- Supervised fine-tuning of language models.
- Benchmarking text classification models.
- Research into financial text semantics.

## Acknowledgements

This dataset was created by the authors of the following paper. Please cite them if you use this dataset in your work:

```bibtex
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={Pekka Malo and Ankur Sinha and Pekka Korhonen and Jyrki Wallenius and Pasi Takala},
  journal={Journal of the Association for Information Science and Technology},
  year={2014},
  volume={65},
  pages={782-796}
}