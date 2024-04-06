import random
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset, DatasetDict


def get_dataset(cfg, tokenizer):
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

    return dataset.map(
        lambda example: tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        ),
        remove_columns=(["text"]),
        batched=True,
    )


def run(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    dataset = get_dataset(cfg, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model_name, num_labels=2
    ).to(device)

    training_args = TrainingArguments(
        output_dir=f"./models/{cfg.base_model_name}",
        overwrite_output_dir=True,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.05,
        weight_decay=0.01,
        learning_rate=cfg.base_learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        eval_accumulation_steps=1,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=2,
        tf32=True,
        group_by_length=True,
    )

    def compute_metrics(p):
        labels = p.label_ids
        predictions = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        predictions = np.argmax(predictions, axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )

    trainer.train()

    print("Evaluating on test set...")
    results = trainer.predict(dataset["test"])
    print(results.metrics)
