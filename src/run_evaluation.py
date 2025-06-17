from argparse import ArgumentParser
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import load_dataset, DatasetDict
import evaluate
import torch

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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--src_lang", type=str, default="ind")
    parser.add_argument("--tgt_lang", type=str, default="btx")
    parser.add_argument("--src_lang_nllb", type=str, default="ind_Latn")
    parser.add_argument("--tgt_lang_nllb", type=str, default="btx_Latn")
    parser.add_argument("--source_text_column_name", type=str, default="text_source")
    parser.add_argument("--target_text_column_name", type=str, default="text_target")
    parser.add_argument("--dataset_split_name", type=str, default="test")
    parser.add_argument("--book_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--num_proc", type=int, default=16)
    return parser.parse_args()


def main(args):
    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    src_lang_nllb, tgt_lang_nllb = args.src_lang_nllb, args.tgt_lang_nllb
    output_dir = args.model_name.split("/")[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    dataset = dataset[args.dataset_split_name]

    if args.book_name:
        dataset = dataset.filter(lambda x: x["book"] == args.book_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.src_lang = src_lang_nllb
    tokenizer.tgt_lang = tgt_lang_nllb

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        src_lang=src_lang_nllb,
        tgt_lang=tgt_lang_nllb,
    )

    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels, verse_ids = eval_preds["prediction"], eval_preds["target"], eval_preds["verse_id"]
        cleaned_preds, cleaned_labels = postprocess_text(preds, labels)

        bleu_result = bleu.compute(predictions=cleaned_preds, references=cleaned_labels)
        chrf_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels)
        eval_result = {"bleu": bleu_result["bleu"], "chrf": chrf_result["score"]}
        eval_result = {k: round(v, 4) for k, v in eval_result.items()}

        results = [
            {"verse_id": verse_id, "prediction": pred, "target": label}
            for verse_id, pred, label in zip(verse_ids, preds, labels)
        ]

        return {"eval_metrics": eval_result, "results": results}

    def infer(batch):
        predictions = [
            out["translation_text"]
            for out in translator(
                batch[args.source_text_column_name],
                batch_size=args.per_device_eval_batch_size,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )
        ]
        batch["prediction"] = predictions
        batch["target"] = batch[args.target_text_column_name]
        return batch

    results = dataset.map(infer, batched=True, batch_size=args.per_device_eval_batch_size)
    output = compute_metrics(results)

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
