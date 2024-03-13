import sys

import numpy as np

from datasets import Dataset

import json

from datasets import Dataset, DatasetDict, concatenate_datasets


def transform_data(data):
    transformed_examples = []
    texts = data["text"]
    labels = data["labels"]

    for i in range(len(texts)):
        transformed_example = {
            "text": texts[i],
            "label": int(labels[i]),
            "left_context": " \n ".join(texts[:i]),
            "right_context": " \n ".join(texts[i + 1 :]),
        }
        transformed_examples.append(transformed_example)

    return transformed_examples


def gen(language, split):
    with open(f"data/{language}/{split}.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    transformed_data = [example for item in data for example in transform_data(item)]
    for item in transformed_data:
        yield item


def get_dataset(cfg):
    splits = {}
    splits["train"] = Dataset.from_generator(
        gen,
        cache_dir="./tokens_cache",
        gen_kwargs={"language": cfg.language, "split": "train"},
    )
    splits["dev"] = Dataset.from_generator(
        gen,
        cache_dir="./tokens_cache",
        gen_kwargs={"language": cfg.language, "split": "test"},
    )
    splits["test"] = Dataset.from_generator(
        gen,
        cache_dir="./tokens_cache",
        gen_kwargs={"language": cfg.language, "split": "dev"},
    )

    return DatasetDict(splits)


def preprocess_dataset(dataset, tokenizer, cfg):

    def process(examples):

        concatenated_texts = [
            f"{a.replace('§', '')} \n § {b.replace('§', '')} § \n {c.replace('§', '')}"
            for a, b, c in zip(
                examples["left_context"], examples["text"], examples["right_context"]
            )
        ]

        # Tokenize the concatenated texts
        tokenized_inputs = tokenizer(
            concatenated_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        ignore = []

        for i, ids in enumerate(tokenized_inputs["input_ids"]):
            counts = len([x for x in ids if int(x) == 5360])
            if counts == 2:
                ignore.append(False)
            else:
                ignore.append(True)

        tokenized_inputs["ignore"] = ignore

        return tokenized_inputs

    # dataset = dataset.shuffle(seed=cfg.seed)

    tokenized_dataset = dataset.map(
        process,
        remove_columns=["text", "left_context", "right_context"],
        batched=True,
    )
    filtered_dataset = tokenized_dataset.filter(lambda example: not example["ignore"])
    return filtered_dataset
