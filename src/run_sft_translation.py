from argparse import ArgumentParser

from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

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
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-0.6B")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["bible-nlp/biblenlp-corpus", "LazarusNLP/alkitab-sabda-mt"],
    )
    parser.add_argument("--src_lang", type=str, default="ind")
    parser.add_argument("--tgt_lang", type=str, default="ptu")
    parser.add_argument("--src_lang_name", type=str, default="Indonesian")
    parser.add_argument("--tgt_lang_name", type=str, default="Bambam")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--num_proc", type=int, default=16)
    return parser.parse_args()


def load_ebible_corpus(src_lang, tgt_lang):
    dataset = load_dataset("bible-nlp/biblenlp-corpus", languages=[src_lang, tgt_lang], trust_remote_code=True)
    dataset = dataset.map(
        lambda x: {"text_source": x["translation"][0], "text_target": x["translation"][1]},
        input_columns=["translation"],
    )
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


def main(args):
    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    src_lang_name, tgt_lang_name = args.src_lang_name, args.tgt_lang_name
    dataset_name = args.dataset_name.split("/")[-1]
    output_dir = f"{args.model_name.split('/')[-1]}-{dataset_name}-{src_lang}-{tgt_lang}"

    if dataset_name == "biblenlp-corpus":
        dataset = load_ebible_corpus(src_lang, tgt_lang)
    elif dataset_name == "alkitab-sabda-mt":
        dataset = load_alkitab_sabda(src_lang, tgt_lang)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    def preprocess_function(example, src_lang_name, tgt_lang_name):
        prompt = f"Translate the following text from {src_lang_name} to {tgt_lang_name}:\n\n{example['text_source']}"
        response = f"{example['text_target']}"
        conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    processed_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_proc,
        fn_kwargs={"src_lang_name": src_lang_name, "tgt_lang_name": tgt_lang_name},
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        dataset_text_field="text",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        optim="adamw_8bit",
        report_to="none",
        seed=3407,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_dataset["train"],
        args=training_args,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
