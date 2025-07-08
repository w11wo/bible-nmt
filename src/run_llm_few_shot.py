from argparse import ArgumentParser
from pathlib import Path
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm.contrib.concurrent import thread_map
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from openai import OpenAI
import evaluate


class GPT:
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def create_prompt(
        self,
        query: str,
        src_lang_name: str,
        tgt_lang_name: str,
        few_shot_samples: list[str] = None,
    ) -> str:
        instruction_prompt = f"""Translate the following text from {src_lang_name} to {tgt_lang_name}.
Return your response in JSON format with a single key 'translation'. The value should be the translated text."""

        if few_shot_samples:
            few_shot_prompt = "\n\n".join(
                [f"{src_lang_name}: {sample[0]}\n{tgt_lang_name}: {sample[1]}" for sample in few_shot_samples]
            )
            instruction_prompt += f"\n\nHere are some examples:\n\n{few_shot_prompt}"

        prompt = f"{instruction_prompt}\n\n{src_lang_name}: {query}\n{tgt_lang_name}:"
        return prompt

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1000,
        )

        result = completion.choices[0].message.content
        try:
            translation = json.loads(result)["translation"]
        except:
            return ""
        return translation


class Gemini(GPT):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--src_lang", type=str, default="ind")
    parser.add_argument("--tgt_lang", type=str, default="ptu")
    parser.add_argument("--src_lang_name", type=str, default="Indonesian")
    parser.add_argument("--tgt_lang_name", type=str, default="Bambam")
    parser.add_argument("--vectorizer", type=str, default="bm25", choices=["bm25", "tfidf", "sbert"])
    parser.add_argument("--num_few_shot", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def load_ebible_corpus(src_lang, tgt_lang):
    dataset = load_dataset("bible-nlp/biblenlp-corpus", languages=[src_lang, tgt_lang], trust_remote_code=True)
    dataset = dataset.map(
        lambda x: {"text_source": x["translation"][0], "text_target": x["translation"][1]},
        input_columns=["translation"],
    )
    return dataset


def main(args):
    output_dir = Path(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_ebible_corpus(args.src_lang, args.tgt_lang)

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    train_df = train_df[["text_source", "text_target"]]
    test_df = test_df[["text_source", "text_target"]]

    source_texts = train_df["text_source"]
    target_texts = train_df["text_target"]

    if "gemini" in args.model:
        llm = Gemini(model=args.model)
    elif "gpt" in args.model:
        llm = GPT(model=args.model)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.vectorizer == "tfidf":
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(source_texts)  # (n_samples, n_features)

        def get_top_k_similar_texts_tfidf(query, k):
            query_vector = vectorizer.transform([query])  # (1, n_features)
            cosine_similarities = (tfidf_matrix * query_vector.T).toarray().flatten()  # (n_samples,)
            top_k_indices = cosine_similarities.argsort()[-k:][::-1]
            top_k_texts = [(source_texts[i], target_texts[i]) for i in top_k_indices]
            return top_k_texts

        similar_texts = [get_top_k_similar_texts_tfidf(q, k=args.num_few_shot) for q in test_df["text_source"]]

    elif args.vectorizer == "sbert":
        sbert_model = SentenceTransformer("LazarusNLP/all-indo-e5-small-v4")
        embeddings = sbert_model.encode(source_texts.tolist(), show_progress_bar=True)  # (n_samples, dim)

        def get_top_k_similar_texts_sbert(query, k):
            query_embedding = sbert_model.encode([query])  # (1, dim)
            cosine_similarities = embeddings @ query_embedding.T  # (n_samples, 1)
            top_k_indices = cosine_similarities[:, 0].argsort()[-k:][::-1]
            top_k_texts = [(source_texts[i], target_texts[i]) for i in top_k_indices]
            return top_k_texts

        similar_texts = [get_top_k_similar_texts_sbert(q, k=args.num_few_shot) for q in test_df["text_source"]]

    elif args.vectorizer == "bm25":
        tokenized_corpus = [doc.split() for doc in source_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        def get_top_k_similar_texts_bm25(query, k):
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)
            top_k_indices = scores.argsort()[-k:][::-1]
            top_k_texts = [(source_texts[i], target_texts[i]) for i in top_k_indices]
            return top_k_texts

        similar_texts = [get_top_k_similar_texts_bm25(q, k=args.num_few_shot) for q in test_df["text_source"]]

    # create few-shot prompts
    prompts = [
        llm.create_prompt(
            test_df["text_source"][i],
            src_lang_name=args.src_lang_name,
            tgt_lang_name=args.tgt_lang_name,
            few_shot_samples=similar_texts[i],
        )
        for i in range(len(test_df))
    ]

    # generate predictions
    predictions = thread_map(llm.generate, prompts, max_workers=args.num_workers)
    test_df["prediction"] = predictions
    test_df.to_csv(output_dir / f"{args.src_lang}_{args.tgt_lang}_{args.vectorizer}_predictions.csv", index=False)

    # evaluate predictions
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds["prediction"], eval_preds["text_target"]
        cleaned_preds, cleaned_labels = postprocess_text(preds, labels)

        sacrebleu_result = sacrebleu.compute(predictions=cleaned_preds, references=cleaned_labels)
        chrf_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels)
        eval_result = {"bleu": sacrebleu_result["score"], "chrf": chrf_result["score"]}
        eval_result = {k: round(v, 4) for k, v in eval_result.items()}

        return eval_result

    output = compute_metrics(test_df.to_dict(orient="list"))
    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
