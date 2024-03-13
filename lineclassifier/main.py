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
            loss_fct = nn.CrossEntropyLoss()

            # Compute the loss
            loss = loss_fct(logits, labels_flat)
            # f"loss: {loss.item()}")
            return (loss, logits) if return_outputs else loss

    def compute_metrics(p):
        _, labels = p
        predictions = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        labels = labels.flatten()
        preds = predictions[:, 1].flatten()

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
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        warmup_ratio=0.05,
        weight_decay=0.05,
        learning_rate=cfg.learning_rate,
        logging_dir="./logs",
        evaluation_strategy=cfg.eval_strategy,
        save_strategy="no",
        logging_strategy="epoch",
        eval_steps=1,
    )

    config = CustomXLMRobertaConfig.from_pretrained(
        "xlm-roberta-large",
        num_labels=2,
        sep_id=5360,
    )

    model = XLMRobertaForLineClassification(config)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate(dataset["test"]))
