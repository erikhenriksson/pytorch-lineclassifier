import json

from transformers import XLMRobertaModel, XLMRobertaTokenizer
import torch
import torch.nn as nn

from .classifier import DocumentClassifier

from tqdm import tqdm


def run(cfg):

    # Load model and tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-large")

    # Assuming 'documents' is your list of documents as described
    documents = {"train": [], "dev": [], "test": []}

    for language in cfg.languages.split("-"):
        for split in ["train", "dev", "test"]:
            with open(f"data/{language}/dev.json", "r", encoding="utf-8") as file:
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

    # Example: Extract embeddings for the first document
    doc_matrix, labels = extract_cls_embeddings_and_labels(
        documents["train"][0], tokenizer, model
    )
    print(doc_matrix.shape)  # Should be [seq_length, 1024] for xlm-roberta-large
    print(doc_matrix)
    print(labels)

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
                pad_size = (0, max_length - t.size(0))  # Pad the end of the 1D tensor
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

        print(embeddings_list)
        print(labels_list)

        # Pad the embeddings and labels to have the same sequence length
        padded_embeddings = pad_tensors(embeddings_list, pad_token=0)
        padded_labels = pad_tensors(labels_list, pad_token=-100, is_embedding=False)

        return padded_embeddings, padded_labels

    # Assuming documents["train"] is a list of document dictionaries
    train_embeddings, train_labels = process_documents(
        documents["train"][:4], tokenizer, model
    )

    print(train_embeddings)
    print(train_labels)

    from torch.utils.data import Dataset, DataLoader

    class EmbeddingsDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    # Assuming train_embeddings and train_labels are prepared
    train_dataset = EmbeddingsDataset(train_embeddings, train_labels)

    # Define a DataLoader for batching
    batch_size = 2  # Adjust based on your GPU memory and model size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DocumentClassifier(embedding_dim=1024, nhead=8, nhid=2048, nlayers=6).to(
        device
    )

    # Assuming binary classification with labels in {0, 1}
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(5):
        model.train()  # Set the model to training mode
        total_loss = 0  # Reset total loss for each epoch

        for batch_embeddings, batch_labels in train_loader:
            print(batch_embeddings.shape)
            print(batch_labels.shape)
            print(batch_labels)
            exit()
            optimizer.zero_grad()  # Reset gradients for the current batch

            # Forward pass: Compute the model output for the current batch
            output = model(batch_embeddings.to(device))
            loss = criterion(output.squeeze(), batch_labels.to(device))

            # Backward pass: Compute gradients
            loss.backward(retain_graph=True)
            # Parameters update: Apply gradients
            optimizer.step()

            # Accumulate loss
            total_loss += (
                loss.item()
            )  # Safe to do after loss.backward(), does not affect the graph

        print(f"Epoch {epoch+1}, Total Loss: {total_loss}")
