import json

from datasets import Dataset, DatasetDict


def gen(languages, split):
    for language in languages.split("-"):
        with open(f"data/{language}/{split}.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        for item in data:
            for i in range(len(item["text"])):
                yield {
                    "text": item["text"][i],
                    "labels": int(item["labels"][i]),
                    "language": language,
                }


def get_dataset(cfg, tokenizer):
    generate = lambda split: Dataset.from_generator(
        gen,
        cache_dir="./tokens_cache",
        gen_kwargs={
            "languages": cfg.languages,
            "split": split,
        },
    )
    splits = {}
    for split in ["train", "dev", "test"]:
        splits[split] = generate(split)

    dataset = DatasetDict(splits).shuffle(seed=cfg.seed)

    print(dataset["train"][0])

    return dataset.map(
        lambda example: tokenizer(
            example["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        ),
        remove_columns=(["text"]),
        batched=True,
    )
