import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import dotenv_values
from datasets import load_dataset
from huggingface_hub import login

from torch.nn.utils.rnn import pad_sequence

from .train_lstm import LSTMForLineClassification


def parse_and_clean(doc, lang):
    data = []
    for line, label in zip(
        doc["text"].split("\n"),
        doc["meta"]["sentence_identifications"],
    ):
        if label and label["label"] == lang:
            data.append(line)

    return data, doc["meta"]["warc_headers"]["warc-record-id"]


def run(cfg):
    env = dotenv_values(".env")
    login(token=env["HF_TOKEN"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.trained_model_name
    ).to(device)

    # Get the original model's name
    with open(f"{cfg.trained_model_name}/config.json", "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path", None))

    # Get trained lstm model
    lstm_model = LSTMForLineClassification(
        embedding_dim=1024,
        hidden_dim=512,
        num_layers=2,
        num_classes=1,
        bidirectional=True,
    ).to(device)

    model_state_dict = torch.load(cfg.lstm_model_name, map_location=device)
    lstm_model.load_state_dict(model_state_dict)

    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        cfg.predict_language,
        "train",
        streaming=True,
        trust_remote_code=True,
    )

    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=1000)["train"]

    print("Dataset loaded")

    batch_size = 2
    max_lines_per_batch = 8
    limit = 100000
    n = 0
    epoch = 0
    epoch_n = 0
    lstm_max_lines = 100000000

    dataset_iterator = iter(shuffled_dataset)

    # Process in batches
    while True:
        batch = []

        for _ in range(batch_size):
            try:
                example = next(dataset_iterator)
                batch.append(example)
            except StopIteration:
                # End of the dataset reached
                break

        batch_cls_embeddings = []
        base_batch_labels = []
        base_batch_probs = []

        base_model_labels = []
        base_model_probs = []
        base_model_cls_embeddings = []

        all_lines = [line for ex in batch for line in ex["text"].split("\n")]

        for start_idx in range(0, len(all_lines), max_lines_per_batch):
            end_idx = min(start_idx + max_lines_per_batch, num_lines)
            lines_sub_batch = all_lines[start_idx:end_idx]

            # Tokenize the current sub-batch of lines
            sub_batch_inputs = tokenizer(
                lines_sub_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            ).to(device)

            # Forward pass for the current sub-batch
            with torch.no_grad():
                sub_batch_outputs = base_model(
                    **sub_batch_inputs, output_hidden_states=True
                )
                cls_embeddings = sub_batch_outputs.hidden_states[-1][:, 0, :]
                sub_batch_preds = torch.sigmoid(sub_batch_outputs.logits)

                sub_batch_probs = sub_batch_preds[:, 1].tolist()
                sub_batch_labels = (sub_batch_preds > 0.5).long()[:, 1].tolist()

            # Aggregate the results
            base_model_cls_embeddings.append(cls_embeddings)
            base_model_probs.extend(sub_batch_probs)
            base_model_labels.extend(sub_batch_labels)

        # Concatenate all cls_embeddings from sub-batches
        cls_embeddings = torch.cat(cls_embeddings, dim=0)

        """

        # Collect lines from all documents in the batch for parallel processing
        all_lines = [line for ex in batch for line in ex["text"].split("\n")]
        all_inputs = tokenizer(
            all_lines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )

        all_inputs = {k: v.to(device) for k, v in all_inputs.items()}

        with torch.no_grad():
            batch_outputs = base_model(**all_inputs, output_hidden_states=True)
            cls_embeddings = batch_outputs.hidden_states[-1][:, 0, :]

            base_model_preds = torch.sigmoid(batch_outputs.logits)
            base_model_probs = base_model_preds[:, 1].tolist()
            base_model_labels = (base_model_preds > 0.5).long()[:, 1].tolist()
        
        """

        # Now, split the cls_embeddings back into per-document batches
        current_idx = 0
        for ex in batch:
            num_lines = len(ex["text"].split("\n"))
            ex_embeddings = cls_embeddings[current_idx : current_idx + num_lines]
            ex_probs = base_model_probs[current_idx : current_idx + num_lines]
            ex_labels = base_model_labels[current_idx : current_idx + num_lines]
            batch_cls_embeddings.append(ex_embeddings)
            base_batch_probs.append(ex_probs)
            base_batch_labels.append(ex_labels)
            current_idx += num_lines

        labeled_by_lstm = False

        if len(base_model_labels) <= lstm_max_lines:

            # Pad the embeddings for LSTM processing
            padded_embeddings = pad_sequence(
                batch_cls_embeddings, batch_first=True, padding_value=0
            )

            # Process the padded embeddings through the LSTM
            lstm_outputs = lstm_model(padded_embeddings)
            batch_probs = lstm_outputs.squeeze().tolist()

            batch_labels = (lstm_outputs.squeeze() > 0.5).long().tolist()

        else:
            batch_probs = base_batch_probs
            batch_labels = base_batch_labels

        if len(batch_labels) == batch_size and all(
            type(x) == int for x in batch_labels
        ):
            batch_labels = [[item] for item in batch_labels]
        if len(batch_probs) == batch_size and all(
            type(x) == float for x in batch_probs
        ):
            batch_probs = [[item] for item in batch_probs]

        for ex_i, ex in enumerate(batch):
            ex_len = len(ex["text"].split("\n"))

            ex["meta"]["quality_labels"] = batch_labels[ex_i][:ex_len]

            ex["meta"]["quality_probs"] = batch_probs[ex_i][:ex_len]

            n += 1
            epoch_n += 1
            out_file = f"{cfg.predict_language}_cleaned.jsonl"

            with open(f"cleaned/{out_file}", "a", encoding="utf-8") as file:
                file.write(json.dumps(ex, ensure_ascii=False) + "\n")

            if n >= limit:
                exit()

        if epoch_n >= 1000:
            epoch += 1
            shuffled_dataset.set_epoch(epoch)
            dataset_iterator = iter(shuffled_dataset)
            epoch_n = 0
