import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
)

from .data import get_dataset, preprocess_dataset
from .model import CustomXLMRobertaConfig, XLMRobertaForLineClassification

count_separated_sequences = lambda l, n: len(
    [xa for xa in "-".join([str(x) for x in l]).split(str(n)) if xa]
)


def run(cfg):

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    dataset = get_dataset(cfg)

    print(dataset["train"][0])

    dataset = preprocess_dataset(dataset, tokenizer, cfg)

    print(dataset["train"][0])

    def compute_metrics(p):
        labels = p.label_ids
        predictions = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        # Apply softmax to the predictions to convert logits to probabilities
        predictions = np.argmax(predictions, axis=-1)

        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)

        # Calculate precision, recall, and F1 score. `average='binary'` is for binary classification.
        # For multi-class classification, you might want to use `average='micro'`, `average='macro'`, or `average='weighted'`
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )

        # Return a dictionary with your metrics of interest
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
        weight_decay=0.01,
        learning_rate=cfg.learning_rate,
        logging_dir="./logs",
        evaluation_strategy=cfg.eval_strategy,
        save_strategy="no",
        logging_strategy="epoch",
        eval_steps=cfg.eval_steps,
        gradient_accumulation_steps=cfg.grad_acc_steps,
        metric_for_best_model=cfg.best_model_metric,
    )

    if cfg.context:

        config = CustomXLMRobertaConfig.from_pretrained(
            cfg.model,
            num_labels=2,
            sep_id=5360,
        )

        model = XLMRobertaForLineClassification(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model, num_labels=2
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )

    trainer.train()
    print(trainer.evaluate(dataset["test"]))
