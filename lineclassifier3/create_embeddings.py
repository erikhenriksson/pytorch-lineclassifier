import json

from transformers import XLMRobertaModel, XLMRobertaTokenizer
import torch
import torch.nn as nn

from .classifier import DocumentClassifier

from tqdm import tqdm
import pickle
import os


def run(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    model = XLMRobertaModel.from_pretrained(cfg.model_name).to(device)

    # Load documents
    documents = {"train": [], "dev": [], "test": []}
    for language in cfg.languages.split("-"):
        for split in ["train", "dev", "test"]:
            with open(f"data/{language}/{split}.json", "r", encoding="utf-8") as file:
                data = json.load(file)
            for item in data:
                documents[split].append(item)

    def extract_cls_embeddings_and_labels(docs, tokenizer, model, batch_size=32):
        all_cls_embeddings = []
        all_labels = []

        # Process documents in batches
        for i in range(0, len(docs["text"]), batch_size):
            batch_texts = docs["text"][i : i + batch_size]
            batch_labels = [int(x) for x in docs["labels"][i : i + batch_size]]

            # Tokenize the batch of lines
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            ).to(
                model.device
            )  # Ensure inputs are on the same device as the model

            # Perform inference and detach to avoid memory issues
            with torch.no_grad():  # Ensure no gradients are computed for memory efficiency
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[
                    :, 0, :
                ].detach()  # Extract and detach [CLS] embeddings

            all_cls_embeddings.append(cls_embeddings)
            all_labels.extend(batch_labels)  # Collect labels

        # Concatenate all [CLS] embeddings and labels collected from each batch
        doc_matrix = torch.cat(
            all_cls_embeddings, dim=0
        )  # Concatenate along the batch dimension
        labels_tensor = torch.tensor(all_labels, dtype=torch.float)

        return doc_matrix, labels_tensor

    def pad_tensors(tensors, pad_token=0, is_embedding=True):
        """Pad a list of tensors to the same length with pad_token. Adjusts for 1D and 2D tensors."""
        max_length = max(t.size(0) for t in tensors)
        padded_tensors = []
        for t in tensors:
            if is_embedding:  # For 2D tensors (embeddings)
                pad_size = (
                    0,
                    0,
                    0,
                    max_length - t.size(0),
                )  # Pad the sequence dimension
            else:  # For 1D tensors (labels)
                pad_size = (
                    0,
                    max_length - t.size(0),
                )  # Pad the end of the 1D tensor
            padded = torch.nn.functional.pad(t, pad_size, value=pad_token)
            padded_tensors.append(padded)
        return torch.stack(padded_tensors)

    def process_documents(documents, tokenizer, model):
        embeddings_list = []
        labels_list = []

        for doc in tqdm(documents):
            doc_embeddings, doc_labels = extract_cls_embeddings_and_labels(
                doc, tokenizer, model
            )
            embeddings_list.append(doc_embeddings)
            labels_list.append(doc_labels)

        # Pad the embeddings and labels to have the same sequence length
        padded_embeddings = pad_tensors(embeddings_list, pad_token=0)
        padded_labels = pad_tensors(labels_list, pad_token=-100, is_embedding=False)

        return padded_embeddings, padded_labels

    for split in cfg.splits.split("-"):
        documents_to_process = documents[split]
        if cfg.sample:
            documents_to_process = documents_to_process[: cfg.sample]
        embeddings, labels = process_documents(documents_to_process, tokenizer, model)

        with open(f"{split}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        with open(f"{split}_labels.pkl", "wb") as f:
            pickle.dump(labels, f)
