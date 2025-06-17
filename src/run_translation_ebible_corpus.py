from argparse import ArgumentParser

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
import numpy as np
import evaluate
import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--src_lang", type=str, default="ind")
    parser.add_argument("--tgt_lang", type=str, default="ptu")
    parser.add_argument("--src_lang_nllb", type=str, default="ind_Latn")
    parser.add_argument("--tgt_lang_nllb", type=str, default="ptu_Latn")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--num_proc", type=int, default=16)
    return parser.parse_args()


def main(args):
    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    src_lang_nllb, tgt_lang_nllb = args.src_lang_nllb, args.tgt_lang_nllb
    output_dir = f"{args.model_name.split('/')[-1]}-ebible-corpus-mt-{src_lang}-{tgt_lang}"

    dataset = load_dataset(
        "bible-nlp/biblenlp-corpus",
        languages=[src_lang, tgt_lang],
        trust_remote_code=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Add special tokens for source and target languages if they are not already present
    if src_lang_nllb not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [src_lang_nllb]})

    if tgt_lang_nllb not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [tgt_lang_nllb]})

    tokenizer.src_lang = src_lang_nllb
    tokenizer.tgt_lang = tgt_lang_nllb

    # Initialize new token embeddings with the mean of old embeddings
    old_embeddings = model.get_input_embeddings()
    old_num_tokens = old_embeddings.weight.size(dim=0)
    old_embeddings_mean = old_embeddings.weight.mean(dim=0, keepdim=True)

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    embeddings.weight.data[old_num_tokens:, :] = old_embeddings_mean
    model.tie_weights()

    def preprocess_function(examples):
        batch = examples["translation"]
        source_texts = [b["translation"][0] for b in batch]
        target_texts = [b["translation"][1] for b in batch]
        return tokenizer(
            source_texts,
            text_target=target_texts,
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
        )

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_proc,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_result = chrf.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": bleu_result["bleu"], "chrf": chrf_result["score"]}
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience)],
    )

    trainer.train()
    test_results = trainer.evaluate(
        eval_dataset=processed_dataset["test"],
        metric_key_prefix="test",
        max_length=args.max_length,
        num_beams=args.num_beams,
    )
    print(test_results)

    trainer.save_model()
    trainer.create_model_card()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
