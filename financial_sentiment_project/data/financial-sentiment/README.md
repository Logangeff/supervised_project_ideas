---
license: odc-by
task_categories:
- text-classification
language:
- en
tags:
- financial-sentiment
- finance
- sentiment-analysis
- nlp
size_categories:
- 100K<n<1M
---

<p align="center">
  <img src="https://github.com/NosibleAI/nosible-py/blob/main/docs/_static/readme.png?raw=true"/>
<p>

# NOSIBLE Financial Sentiment Dataset

## Changelog
- **v1.0.0:** Initial version

## Who is NOSIBLE?

[**NOSIBLE**](https://www.nosible.com/) is a vertical web-scale search engine. Our worldwide media surveillance products help companies build AI systems that see every worldwide event and act with complete situational awareness. In short, we help companies know everything, all the time. The financial institutions we work with rely on us to deliver media intelligence from every country in every language in real-time. Shortcomings in existing financial datasets and financial models are what inspired us to release this dataset and related models.

- [**NOSIBLE Financial Sentiment v1.1 Base**](https://huggingface.co/NOSIBLE/financial-sentiment-v1.1-base)

## What is it?

The NOSIBLE Financial Sentiment Dataset is an open collection of **100,000** cleaned, deduplicated, and sentiment-labeled news samples. Each label reflects the financial sentiment of a short text snippet, categorizing it based on whether the described events are likely to have a **positive**, **neutral**, or **negative** financial impact on a company.

All text is sourced from the **NOSIBLE Search Feeds** product using a curated set of finance-related queries. Sentiment labels are assigned through a multi-stage, LLM-based annotation pipeline (described below).

Models trained using this dataset outperform those trained solely on the [**Financial PhraseBank**](https://huggingface.co/datasets/takala/financial_phrasebank), even when PhraseBank is used only as an unseen evaluation dataset.

## How to use it
Using the [HuggingFace datasets library](https://huggingface.co/docs/datasets/):

Install the dataset library with `pip install datasets`, then load the dataset:

```python
from datasets import load_dataset

dataset = load_dataset("NOSIBLE/financial-sentiment")
print(dataset)
```

#### Expected Output

```text
DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'netloc', 'url'],
        num_rows: 100000
    })
})
```

You can also access this dataset through any interface supported by [Hugging Face](https://huggingface.co/).

## Dataset Structure

### Data Instances

The following is an example sample from the dataset:

```json
{
    "text": "Offshore staff HOUSTON \u2013 VAALCO Energy is looking to bring in a floating storage and offloading (FSO) unit at the Etame Marin oil field offshore Gabon. The company has signed a non-binding letter of intent with Omni Offshore Terminals to supply and operate the vessel at Etame for up to 11 years, following the expiry of the current FPSO Petr\u00f3leo Nautipa contract with BW Offshore in September 2022. Omni has provided a preliminary proposal for leasing and operating the FSO, which could reduce VAALCO's operating costs by 15-25%, compared with the current FPSO contract during the term of the proposed agreement. Maintaining the current FPSO beyond its contract or transitioning to a different FPSO, VAALCO added, would require substantial capex investments.",
    "label": "positive",
    "netloc": "www.offshore-mag.com",
    "url": "https://www.offshore-mag.com/rigs-vessels/article/14202267/vaalco-contemplating-switch-to-fso-at-etame-offshore-gabon"
}
```

### Data Fields

- `text` (string): A text chunk from a search result.
- `label` (string): The financial-sentiment label.
- `netloc` (string): The domain name of the source document.
- `url` (string): The URL of the document.

## Dataset creation

### Data source
The dataset was sampled from the NOSIBLE Search Feeds, which provides web-scale surveillance data to customers. Samples consist of top-ranked search results from the NOSIBLE search engine in response to safe, curated, and finance-specific queries. All data is sourced exclusively from the public web.

### Relabeling algorithm
Labels were first generated using multiple LLM annotators and were then refined using an active-learning–based relabeling loop.

The algorithm outline is as follows:

1. Hand-label ~200 samples to tune prompts for the LLM annotators.
2. Label a set of 100k samples with LLM labelers:
    - [`xAI: Grok 4 Fast`](https://openrouter.ai/x-ai/grok-4-fast)
    - [`xAI: Grok 4 Fast (reasoning enabled)`](https://openrouter.ai/x-ai/grok-4-fast)
    - [`Google: Gemini 2.5 Flash`](https://openrouter.ai/google/gemini-2.5-flash)
    - [`OpenAI: GPT-5 Nano`](https://openrouter.ai/openai/gpt-5-nano)
    - [`OpenAI: GPT-4.1 Mini`](https://openrouter.ai/openai/gpt-4.1-mini)
    - [`OpenAI: gpt-oss-120b`](https://openrouter.ai/openai/gpt-oss-120b)
    - [`Meta: Llama 4 Maverick`](https://openrouter.ai/meta-llama/llama-4-maverick)
    - [`Qwen: Qwen3 32B`](https://openrouter.ai/qwen/qwen3-32b)
3. Train multiple linear models to predict the majority-vote of the LLM labelers. The features are the text embeddings of the following models:
    - [`Qwen3-Embedding-8B`](https://openrouter.ai/qwen/qwen3-embedding-8b)
    - [`Qwen3-Embedding-4B`](https://openrouter.ai/qwen/qwen3-embedding-4b)
    - [`Qwen3-Embedding-0.6B`](https://openrouter.ai/qwen/qwen3-embedding-0.6b)
    - [`OpenAI: Text Embedding 3 Large`](https://openrouter.ai/openai/text-embedding-3-large)
    - [`Google: Gemini Embedding 001`](https://openrouter.ai/google/gemini-embedding-001)
    - [`Mistral: Mistral Embed 2312`](https://openrouter.ai/mistralai/mistral-embed-2312)
4. Perform iterative relabeling:
    - Compare all the linear models' predictions to the majority-vote label.
    - Identify disagreements where all linear models agree but the majority-vote label does not.
    - Use a larger LLM (“oracle”) to evaluate ambiguous cases and relabel when appropriate.
    - Drop the worst performing linear models from the ensemble.
    - Repeat until no additional samples require relabeling.
5. This is the final dataset used for training the [NOSIBLE Financial Sentiment v1.1 Base](https://huggingface.co/NOSIBLE/financial-sentiment-v1.1-base) model.

We used [`OpenAI: GPT-5.1`](https://openrouter.ai/openai/gpt-5.1) as the oracle.

## Additional information

### License
The dataset is released under the **Open Data Commons Attribution License (ODC-By) v1.0** [license](https://opendatacommons.org/licenses/by/1-0/).

### Attribution
- [NOSIBLE Inc](https://www.nosible.com/) Team

**Contributors**

This dataset was developed and maintained by the following team:

* [**Matthew Dicks**](https://www.linkedin.com/in/matthewdicks98/)
* [**Simon van Dyk**](https://www.linkedin.com/in/simon-van-dyk/)
* [**Stuart Reid**](https://www.linkedin.com/in/stuartgordonreid/)

## Citations
Coming soon, we're working on a white paper.
