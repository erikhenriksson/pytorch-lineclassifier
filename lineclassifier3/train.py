import json

from transformers import XLMRobertaModel, XLMRobertaTokenizer
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

from .classifier import DocumentClassifier

from tqdm import tqdm
import pickle
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def evaluate(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_predictions = []
    all_true_labels = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for batch_embeddings, batch_labels in data_loader:
            batch_embeddings, batch_labels = batch_embeddings.to(
                device
            ), batch_labels.to(device)
            batch_embeddings = batch_embeddings.transpose(
                0, 1
            )  # (seq_len, batch_size, feature)
            src_mask = (batch_embeddings.sum(dim=-1) == 0).transpose(0, 1)

            output = model(batch_embeddings, src_mask=src_mask)
            output = output.view(-1)  # Flatten output

            labels_flat = batch_labels.view(-1)
            loss_mask = (
                labels_flat != -100
            )  # Mask to exclude padded elements in loss calculation
            valid_output = output[loss_mask]
            valid_labels = labels_flat[loss_mask]

            # Compute loss for non-padded labels
            loss = criterion(valid_output, valid_labels)
            loss = loss.mean()
            total_loss += loss.item() * loss_mask.sum().item()

            # Convert logits to predictions
            predictions = torch.sigmoid(valid_output) >= 0.5
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(valid_labels.cpu().numpy())

    average_loss = total_loss / len(all_true_labels) if len(all_true_labels) > 0 else 0
    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions)

    print(f1)

    return average_loss


def predict(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        for batch_embeddings in data_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_embeddings = batch_embeddings.transpose(
                0, 1
            )  # (seq_len, batch_size, feature)
            src_mask = (batch_embeddings.sum(dim=-1) == 0).transpose(0, 1)

            output = model(batch_embeddings, src_mask=src_mask)
            output = output.view(-1)
            predictions.append(torch.sigmoid(output))  # Assuming binary classification

    predictions = torch.cat(predictions).cpu().numpy()
    return predictions


def run(cfg):

    with open("train_embeddings.pkl", "rb") as f:
        train_embeddings = pickle.load(f)
    with open("train_labels.pkl", "rb") as f:
        train_labels = pickle.load(f)
    with open("dev_embeddings.pkl", "rb") as f:
        dev_embeddings = pickle.load(f)
    with open("dev_labels.pkl", "rb") as f:
        dev_labels = pickle.load(f)
    with open("test_embeddings.pkl", "rb") as f:
        test_embeddings = pickle.load(f)
    with open("test_labels.pkl", "rb") as f:
        test_labels = pickle.load(f)

    # Prepare DataLoader for training, dev, and test sets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_embeddings.cpu()), torch.FloatTensor(train_labels.cpu())
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    dev_dataset = TensorDataset(
        torch.FloatTensor(dev_embeddings.cpu()), torch.FloatTensor(dev_labels.cpu())
    )
    dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False)

    test_dataset = TensorDataset(
        torch.FloatTensor(test_embeddings.cpu())
    )  # No labels for test
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DocumentClassifier(embedding_dim=1024, nhead=8, nhid=2048, nlayers=6).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(15):
        model.train()  # Set the model to training mode
        total_loss = 0
        n_elements = 0

        for batch_embeddings, batch_labels in train_loader:
            optimizer.zero_grad()

            batch_embeddings, batch_labels = batch_embeddings.to(
                device
            ), batch_labels.to(device)

            # Transpose batch_embeddings to (seq_len, batch_size, feature) for the Transformer model
            batch_embeddings = batch_embeddings.transpose(0, 1)

            # Generate src_mask with shape (batch_size, seq_len), correctly indicating padded positions
            src_mask = (batch_embeddings.sum(dim=-1) == 0).transpose(
                0, 1
            )  # Transpose back to (batch_size, seq_len) for the mask

            # Forward pass
            output = model(batch_embeddings, src_mask=src_mask)
            output = output.view(-1)  # Flatten output

            # Prepare labels and mask for loss computation
            labels_flat = batch_labels.view(-1)
            loss_mask = (
                labels_flat != -100
            )  # Mask to exclude padded elements in loss calculation

            # Compute loss only for non-padded labels
            loss = criterion(output[loss_mask], labels_flat[loss_mask])
            loss = loss.mean()  # Take the mean of the remaining losses
            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * loss_mask.sum().item()
            )  # Accumulate the total loss
            n_elements += loss_mask.sum().item()  # Count the non-padded elements

        average_loss = (
            total_loss / n_elements
        )  # Calculate the average loss for the epoch
        print(f"Epoch {epoch+1}, Training Average Loss: {average_loss}")

        # Evaluate on dev data after each training epoch
        dev_loss = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1}, Dev Loss: {dev_loss}")
