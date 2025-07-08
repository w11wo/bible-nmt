# 📖 Bible Neural Machine Translation with NLLB and LLMs

## 🔧 Setup

```sh
pip install -r requirements.txt
```

You also need access to a GPU that supports FlashAttention and bfloat16 (e.g. H100/L40S) if you're fine-tuning NLLB models.

## 🌏 Fine-tuning with NLLB

Fine-tune a multilingual translation model (e.g. [NLLB-200](huggingface.co/facebook/nllb-200-distilled-1.3b/)) for Bible verse translation between languages, using domain-specific corpora like:

- [bible-nlp/biblenlp-corpus](https://huggingface.co/datasets/bible-nlp/biblenlp-corpus)
- [LazarusNLP/alkitab-sabda-mt](https://huggingface.co/datasets/LazarusNLP/alkitab-sabda-mt)

It supports language-pair fine-tuning and evaluation using BLEU and chrF metrics, leveraging `transformers` and `datasets` from 🤗 Hugging Face, and is optimized for fast training using Flash Attention and bfloat16.

### 🚀 Usage

To fine-tune the NLLB-200 distilled 1.3B model from Indonesian to a regional language (e.g. Bambam) using eBible corpus, run:

```sh
python src/run_translation.py \
    --model_name facebook/nllb-200-distilled-1.3B \
    --dataset_name bible-nlp/biblenlp-corpus \
    --src_lang ind \
    --tgt_lang ptu \
    --src_lang_nllb ind_Latn \
    --tgt_lang_nllb ptu_Latn \
    --max_length 256 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-4 \
    --max_steps 5000
```

To fine-tune the model from Indonesian to a regional language (e.g. Batak Karo) using Alkitab Sabda, run:

```sh
python src/run_translation.py \
    --model_name facebook/nllb-200-distilled-1.3B \
    --dataset_name LazarusNLP/alkitab-sabda-mt \
    --src_lang ind \
    --tgt_lang btx \
    --src_lang_nllb ind_Latn \
    --tgt_lang_nllb btx_Latn \
    --max_length 256 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-4 \
    --max_steps 5000
```

### 📊 Results

| Source | Target | Dataset            | Eval Set          |  BLEU   |  chrF   |
| ------ | ------ | ------------------ | ----------------- | :-----: | :-----: |
| `ind`  | `ptu`  | `biblenlp-corpus`  | `validation`      | 18.4774 | 47.1922 |
| `ind`  | `ptu`  | `biblenlp-corpus`  | `test`            | 19.4204 | 49.0809 |
| `ind`  | `btx`  | `alkitab-sabda-mt` | `validation` (NT) | 26.2684 | 52.8884 |
| `ind`  | `btx`  | `alkitab-sabda-mt` | `test` (OT)       | 8.4475  | 28.9892 |

## 🦙 Few-shot Translation with LLMs

We can also leverage LLMs (e.g. GPT, Gemini) for few-shot translation of Bible verses, using the same datasets. This approach is more flexible and can be used for languages not covered by NLLB. We implemented few-shot prompting using TF-IDF, BM25, and Sentence Transformers to retrieve similar paired examples from the corpus, and then use them to prompt the LLM for translation.

### 🚀 Usage

To run few-shot translation using an LLM (e.g. Gemini 2.0 Flash) from Indonesian to Bambam with BM25 as the retriever, you can run the following command:

```sh
python src/run_llm_few_shot.py \
    --model gemini-2.0-flash \
    --src_lang ind \
    --tgt_lang ptu \
    --src_lang_name Indonesian \
    --tgt_lang_name Bambam \
    --vectorizer bm25 \
    --num_few_shot 5
```

### 📊 Results

| Source | Target | Dataset           | Eval Set |  BLEU   |  chrF   |
| ------ | ------ | ----------------- | -------- | :-----: | :-----: |
| `ind`  | `ptu`  | `biblenlp-corpus` | `test`   | 14.5615 | 41.1438 |