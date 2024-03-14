import sys

import numpy as np

from datasets import Dataset

import json

from datasets import Dataset, DatasetDict, concatenate_datasets


def transform_data(data, context, window):
    transformed_examples = []
    texts = data["text"]
    labels = data["labels"]

    for i in range(len(texts)):
        # Define the transformed example with basic text and label
        transformed_example = {
            "text": texts[i],
            "label": int(labels[i]),
        }

        if context:

            # Calculate the indices for left and right contexts
            left_context_start = max(0, i - window)
            right_context_end = min(len(texts), i + window + 1)

            # Include the left and right contexts if requested
            transformed_example["left_context"] = " \n ".join(texts[left_context_start:i])
            transformed_example["right_context"] = " \n ".join(
                texts[i + 1 : right_context_end]
            )

        # Append the transformed example to the results list
        transformed_examples.append(transformed_example)

    return transformed_examples


def gen(language, split, context, window):
    with open(f"data/{language}/{split}.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    transformed_data = [
        example for item in data for example in transform_data(item, context, window)
    ]
    for item in transformed_data:
        yield item


def get_dataset(cfg):
    generate = lambda split: Dataset.from_generator(
        gen,
        cache_dir="./tokens_cache",
        gen_kwargs={
            "language": cfg.language,
            "split": split,
            "context": cfg.context,
            "window": cfg.window,
        },
    )
    splits = {}
    splits["train"] = generate("train")
    splits["dev"] = generate("dev")
    splits["test"] = generate("test")

    return DatasetDict(splits)


def preprocess_dataset(dataset, tokenizer, cfg):

    def concatenate(left, target, right):
        left = (left + " \n") if left else left
        right = ("\n " + right) if right else right
        return f"{left} § {target} § {right}"

    def process(examples):

        concatenated_texts = (
            [
                concatenate(a, b, c)
                for a, b, c in zip(
                    examples["left_context"],
                    examples["text"],
                    examples["right_context"],
                )
            ]
            if cfg.context
            else examples["text"]
        )

        # Tokenize the concatenated texts
        tokenized_inputs = tokenizer(
            concatenated_texts,
            padding="max_length",
            truncation=True,
            max_length=512 if not cfg.context else cfg.context,
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
