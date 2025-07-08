from argparse import ArgumentParser
import json

from unsloth import FastLanguageModel
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import evaluate
import torch

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

NT_BOOKS = [
    "MAT",
    "MRK",
    "LUK",
    "JHN",
    "ACT",
    "ROM",
    "1CO",
    "2CO",
    "GAL",
    "EPH",
    "PHP",
    "COL",
    "1TH",
    "2TH",
    "1TI",
    "2TI",
    "TIT",
    "PHM",
    "HEB",
    "JAB",
    "1PE",
    "2PE",
    "1JN",
    "2JN",
    "3JN",
    "JUD",
    "REV",
]


def load_ebible_corpus(src_lang, tgt_lang):
    dataset = load_dataset("bible-nlp/biblenlp-corpus", languages=[src_lang, tgt_lang], trust_remote_code=True)
    dataset = dataset.map(
        lambda x: {"text_source": x["translation"][0], "text_target": x["translation"][1]}, input_columns="translation"
    )
    dataset = dataset.map(lambda x: {"verse_id": x[0], "book": x[0].split()[0]}, input_columns="ref")
    return dataset


def load_alkitab_sabda(src_lang, tgt_lang):
    dataset = load_dataset(
        "LazarusNLP/alkitab-sabda-mt",
        f"{src_lang}-{tgt_lang}",
        split="train+validation+test",
        trust_remote_code=True,
    )
    dataset = dataset.map(lambda x: {"book": x["verse_id"].split("_")[0]})

    # OT books for testing, NT books for training and validation
    train_ds = dataset.filter(lambda x: x["book"] in NT_BOOKS)
    test_ds = dataset.filter(lambda x: x["book"] not in NT_BOOKS)
    train_val_ds = train_ds.train_test_split(test_size=0.1, seed=41)
    dataset = DatasetDict({"train": train_val_ds["train"], "validation": train_val_ds["test"], "test": test_ds})
    return dataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-0.6B")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["bible-nlp/biblenlp-corpus", "LazarusNLP/alkitab-sabda-mt"],
    )
    parser.add_argument("--src_lang", type=str, default="ind")
    parser.add_argument("--tgt_lang", type=str, default="btx")
    parser.add_argument("--src_lang_name", type=str, default="Indonesian")
    parser.add_argument("--tgt_lang_name", type=str, default="Bambam")
    parser.add_argument("--dataset_split_name", type=str, default="test")
    parser.add_argument("--book_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_proc", type=int, default=16)
    return parser.parse_args()


def main(args):
    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    src_lang_name, tgt_lang_name = args.src_lang_name, args.tgt_lang_name
    dataset_name = args.dataset_name.split("/")[-1]
    output_dir = args.model_name.split("/")[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset_name == "biblenlp-corpus":
        dataset = load_ebible_corpus(src_lang, tgt_lang)
    elif dataset_name == "alkitab-sabda-mt":
        dataset = load_alkitab_sabda(src_lang, tgt_lang)

    dataset = dataset[args.dataset_split_name]

    if args.book_name:
        dataset = dataset.filter(lambda x: x["book"] == args.book_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels, verse_ids = eval_preds["prediction"], eval_preds["target"], eval_preds["verse_id"]
        cleaned_preds, cleaned_labels = postprocess_text(preds, labels)

        sacrebleu_result = sacrebleu.compute(predictions=cleaned_preds, references=cleaned_labels)
        chrf_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels)
        eval_result = {"bleu": sacrebleu_result["score"], "chrf": chrf_result["score"]}
        eval_result = {k: round(v, 4) for k, v in eval_result.items()}

        results = [
            {"verse_id": verse_id, "prediction": pred, "target": label}
            for verse_id, pred, label in zip(verse_ids, preds, labels)
        ]

        return {"eval_metrics": eval_result, "results": results}

    def preprocess_function(example, src_lang_name, tgt_lang_name):
        prompt = f"Translate the following text from {src_lang_name} to {tgt_lang_name}:\n\n{example['text_source']}"
        conversation = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return {"text": text}

    processed_dataset = dataset.map(
        preprocess_function,
        num_proc=args.num_proc,
        fn_kwargs={"src_lang_name": src_lang_name, "tgt_lang_name": tgt_lang_name},
    )

    predictions, targets, verse_ids = [], [], []
    for item in tqdm(processed_dataset):
        inputs = tokenizer(item["text"], return_tensors="pt").to(device)
        output = model.generate(
            **inputs, max_new_tokens=args.max_length, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
        )
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        prediction = prediction.split("</think>")[-1].strip()

        predictions.append(prediction)
        targets.append(item["text_target"])
        verse_ids.append(item["verse_id"])

    results = {"verse_id": verse_ids, "prediction": predictions, "target": targets}

    output = compute_metrics(results)

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
