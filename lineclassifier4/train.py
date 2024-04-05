import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .model import TransformerForLineClassification
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

criterion = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DocumentDataset(Dataset):
    def __init__(self, documents):
        """
        Initializes the dataset with a list of documents.
        Each document is a dictionary with a 'text' key, which is a list of tuples.
        Each tuple in the list contains an embedding array and its associated label (as a string).
        """
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        # Extract embeddings and labels from the document's lines
        embeddings, labels = zip(*document["text"])
        # Convert the embeddings and labels into tensors
        embeddings_tensor = torch.stack(
            [torch.tensor(e, dtype=torch.float) for e in embeddings]
        )
        labels_tensor = torch.tensor([float(l) for l in labels], dtype=torch.float)
        return embeddings_tensor, labels_tensor


def collate_fn(batch):
    embeddings, labels = zip(*batch)
    # Dynamically pad the embeddings and labels within each batch
    embeddings_padded = pad_sequence(embeddings, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=-1
    )  # Assuming -1 is an invalid label used for padding
    return embeddings_padded, labels_padded


def evaluate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Inference mode, no backpropagation
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            predictions = model(embeddings).squeeze()

            # Ensure predictions are always 2-dimensional
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0).unsqueeze(
                    0
                )  # Adjust for a single value
            elif predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)  # Adjust for a single line

            mask = labels != -1
            valid_labels = labels[mask].cpu().numpy()  # Filters out padded labels
            valid_predictions = (
                predictions[mask].cpu().numpy()
            )  # Correspondingly filters predictions to match valid labels

            # Threshold predictions for binary classification
            valid_predictions = (valid_predictions > 0.5).astype(np.int32)

            all_labels.extend(valid_labels)
            all_predictions.extend(valid_predictions)

            # Compute loss on valid_predictions and valid_labels only
            loss = criterion(
                torch.tensor(valid_predictions, dtype=torch.float, device=device),
                torch.tensor(valid_labels, dtype=torch.float, device=device),
            )
            total_loss += loss.item()

    # Calculate metrics
    f1 = f1_score(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, f1, accuracy


def run(cfg):
    with open("documents_embeddings_with_labels.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    # Create separate dataset instances for train, dev, and test
    train_dataset = DocumentDataset(
        [doc for doc in loaded_data["train"] if doc["text"]]
    )
    dev_dataset = DocumentDataset([doc for doc in loaded_data["dev"] if doc["text"]])
    test_dataset = DocumentDataset([doc for doc in loaded_data["test"] if doc["text"]])

    print(train_dataset[0])

    batch_size = 4  # Define your batch size; adjust as necessary

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Assume the model is already created as 'model'
    model = TransformerForLineClassification(
        embedding_dim=1024, nhead=8, num_encoder_layers=3, num_classes=1
    ).to(device)

    data_len = len(train_dataloader)

    print(data_len)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(cfg.epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for embeddings, labels in train_dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            # Forward pass: Compute predicted labels by passing embeddings to the model
            predictions = model(embeddings).squeeze()

            mask = labels != -1
            # Ensure predictions are always 2-dimensional
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0).unsqueeze(
                    0
                )  # Adjust for a single value
            elif predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)  # Adjust for a single line

            valid_labels = labels[mask]  # Filters out padded labels
            valid_predictions = predictions[
                mask
            ]  # Correspondingly filters predictions to match valid labels

            # Now compute loss on valid_predictions and valid_labels only
            loss = criterion(valid_predictions, valid_labels)

            # Calculate loss
            # loss = criterion(predictions, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        print(
            f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {total_loss / len(train_dataloader)}"
        )
        avg_loss, f1, accuracy = evaluate(model, dev_dataloader, device)
        print(f"Validation Loss: {avg_loss}, F1 Score: {f1}, Accuracy: {accuracy}")

    exit()

    # Training Loop
    for epoch in range(cfg.epochs):
        # Train on training data
        model.train()
        total_loss = 0
        for embeddings_padded, labels_padded in train_dataloader:
            optimizer.zero_grad()

            # Forward pass
            predictions = model(embeddings_padded).squeeze()

            # Calculate loss only for non-padded values
            mask = labels_padded != -1  # Assuming -1 used for padding labels
            loss = criterion(predictions[mask], labels_padded[mask])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

        validation_loss = evaluate(model, dev_dataloader)
        print(validation_loss)
    exit()
    # Finally, evaluate performance on test data
    model.eval()
    with torch.no_grad():
        for embeddings_padded, labels_padded in test_dataloader:
            # Testing steps...
            pass
