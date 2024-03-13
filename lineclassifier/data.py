import sys

import numpy as np

from datasets import Dataset

import json

from datasets import Dataset, DatasetDict, concatenate_datasets


def transform_data(data, context):
    transformed_examples = []
    texts = data["text"]
    labels = data["labels"]

    for i in range(len(texts)):
        transformed_example = {
            "text": texts[i],
            "label": int(labels[i]),
        }
        if context:
            transformed_example["left_context"] = " \n ".join(texts[:i])
            transformed_example["right_context"] = (" \n ".join(texts[i + 1 :]),)
        transformed_examples.append(transformed_example)

    return transformed_examples


def gen(language, split, context):
    with open(f"data/{language}/{split}.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    transformed_data = [
        example for item in data for example in transform_data(item, context)
    ]
    for item in transformed_data:
        yield item


def get_dataset(cfg):
    generate = lambda split: Dataset.from_generator(
        gen,
        cache_dir="./tokens_cache",
        gen_kwargs={"language": cfg.language, "split": split, "context": cfg.context},
    )
    splits = {}
    splits["train"] = generate("train")
    splits["dev"] = generate("dev")
    splits["test"] = generate("test")

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
            concatenated_texts if cfg.context else examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if not cfg.context:
            return tokenized_inputs

        ignore = []

        for i, ids in enumerate(tokenized_inputs["input_ids"]):
            counts = len([x for x in ids if int(x) == 5360])
            if counts == 2:
                ignore.append(False)
            else:
                ignore.append(True)

        tokenized_inputs["ignore"] = ignore

        return tokenized_inputs

    dataset = dataset.shuffle(seed=cfg.seed)

    tokenized_dataset = dataset.map(
        process,
        remove_columns=(
            ["text", "left_context", "right_context"] if cfg.context else ["text"]
        ),
        batched=True,
    )
    if cfg.context:
        tokenized_dataset = tokenized_dataset.filter(
            lambda example: not example["ignore"]
        )
    return tokenized_dataset
