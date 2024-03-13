from transformers import (
    XLMRobertaTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
from torch import nn
from datasets import load_dataset

from torch.nn.utils.rnn import pad_sequence
from .model import XLMRobertaForLineClassification, CustomXLMRobertaConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
from .data import get_dataset, preprocess_dataset

count_separated_sequences = lambda l, n: len(
    [xa for xa in "-".join([str(x) for x in l]).split(str(n)) if xa]
)


def run(cfg):

    tokenizer = XLMRobertaTokenizer.from_pretrained(cfg.model)

    dataset = get_dataset(cfg)

    print(dataset["train"][0])

    dataset = preprocess_dataset(dataset, tokenizer, cfg)

    print(dataset["train"][0])

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")

            # Perform a forward pass to get model outputs
            logits = model(**inputs)
            labels_flat = labels.view(-1)

            # Initialize the loss function with ignore_index to skip the padded labels
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            # Compute the loss
            loss = loss_fct(logits, labels_flat)
            # f"loss: {loss.item()}")
            return (loss, logits) if return_outputs else loss

    def preprocess_data(examples):
        # Concatenate lines with <s> token and process labels
        concatenated_texts = [
            " § " + (" § ".join([x.replace("§", "") for x in text])) + " § "
            for text in examples["text"]
        ]
        labels = [[int(y) for y in x] for x in examples["labels"]]

        # Tokenize the concatenated texts
        tokenized_inputs = tokenizer(
            concatenated_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        sep_id = 5360

        # Adjust labels to match tokenized and potentially truncated texts
        adjusted_labels = []
        for i, (input_ids, label) in enumerate(
            zip(tokenized_inputs["input_ids"], labels)
        ):
            input_ids = [int(x) for x in input_ids if x not in [0, 1, 2]]

            sep_sequences = count_separated_sequences(input_ids, sep_id)

            adjusted_labels.append(label[:sep_sequences])

        tokenized_inputs["labels"] = adjusted_labels

        return tokenized_inputs

        # Adjust labels to match tokenized and potentially truncated texts
        adjusted_labels = []
        for i, (input_ids, label) in enumerate(
            zip(tokenized_inputs["input_ids"], labels)
        ):
            # Count <s> tokens in the tokenized input
            special_token_count = (
                (input_ids == tokenizer.convert_tokens_to_ids("<s>")).sum().item()
            )
            # Adjust the labels to match the number of <s> tokens in the truncated sequence
            adjusted_labels.append(label[:special_token_count])

        tokenized_inputs["labels"] = adjusted_labels

        return tokenized_inputs

    # Load the dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"data/{cfg.language}/train.json",
            "validation": f"data/{cfg.language}/dev.json",
            "test": f"data/{cfg.language}/test.json",
        },
    )

    # Apply the preprocessing function to all splits
    tokenized_dataset = dataset.map(
        preprocess_data,
        remove_columns=["text"],
        # preprocess_data,
        batched=True,
    )

    max_labels = 0

    tokenized_dataset = tokenized_dataset.shuffle(seed=cfg.seed)

    # Assuming 'dataset' is your dataset object and it's iterable
    for ds in tokenized_dataset.values():
        for item in ds:
            # Assuming each item is a dictionary and labels are stored under the "labels" key
            num_labels = len(item["labels"])
            if num_labels > max_labels:
                max_labels = num_labels

    def custom_data_collator(features):
        batch = {}

        # Handle inputs and attention masks normally
        batch["input_ids"] = pad_sequence(
            [torch.tensor(f["input_ids"]) for f in features],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        batch["attention_mask"] = pad_sequence(
            [torch.tensor(f["attention_mask"]) for f in features],
            batch_first=True,
            padding_value=0,
        )

        max_labels_length = max_labels
        labels_padded = [
            (
                torch.cat(
                    (
                        torch.tensor(f["labels"], dtype=torch.long),
                        torch.tensor(
                            [-100] * (max_labels_length - len(f["labels"])),
                            dtype=torch.long,
                        ),
                    )
                )
                if len(f["labels"]) < max_labels_length
                else torch.tensor(f["labels"], dtype=torch.long)
            )
            for f in features
        ]

        batch["labels"] = torch.stack(labels_padded, dim=0)
        return batch

    def compute_metrics(p):
        _, labels = p
        predictions = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        flattened_labels = labels.flatten()

        # Find indices of non-padding labels
        valid_indices = np.where(flattened_labels != -100)[0]
        valid_indices = valid_indices[valid_indices < predictions.shape[0]]

        preds = predictions[:, 1].flatten()[valid_indices]
        labels = flattened_labels[valid_indices]

        probs = 1 / (1 + np.exp(-preds))
        binary_preds = (probs >= 0.5).astype(int)

        # Compute evaluation metrics
        accuracy = accuracy_score(labels, binary_preds)
        precision = precision_score(labels, binary_preds)
        recall = recall_score(labels, binary_preds)
        f1 = f1_score(labels, binary_preds)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.b,
        per_device_eval_batch_size=cfg.b,
        warmup_ratio=0.05,
        weight_decay=0.05,
        learning_rate=cfg.lr,
        logging_dir="./logs",
        evaluation_strategy=cfg.eval_strategy,
        save_strategy="no",
        logging_strategy="epoch",
        eval_steps=1,
    )

    sep_id = tokenizer.convert_tokens_to_ids("§")
    # print(sep_id)
    # Create a custom configuration with max_lines
    config = CustomXLMRobertaConfig.from_pretrained(
        "xlm-roberta-large",
        num_labels=2,
        max_lines=max_labels,
        pooling=cfg.pool,
        sep_id=5360,
    )

    model = XLMRobertaForLineClassification(config)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    test_results = trainer.evaluate(tokenized_dataset["test"])

    print(test_results)