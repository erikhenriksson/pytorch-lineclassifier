import json

from transformers import XLMRobertaModel, XLMRobertaTokenizer
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


from sklearn.metrics import accuracy_score, f1_score

from .classifier import DocumentClassifier, SimpleRNNClassifier

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

            # output = model(batch_embeddings, src_mask=src_mask)
            output = model(batch_embeddings)
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

    return average_loss, accuracy, f1


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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    dev_dataset = TensorDataset(
        torch.FloatTensor(dev_embeddings.cpu()), torch.FloatTensor(dev_labels.cpu())
    )
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    test_dataset = TensorDataset(
        torch.FloatTensor(test_embeddings.cpu())
    )  # No labels for test
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DocumentClassifier(embedding_dim=1024, nhead=4, nhid=2048, nlayers=6).to(
    #    device
    # )

    # Example of initializing the SimpleRNNClassifier
    embedding_dim = 1024  # The size of each line embedding
    hidden_dim = 512  # Hidden dimension size in LSTM
    num_layers = 2  # Number of LSTM layers
    num_classes = (
        1  # For binary classification, use 1; for multi-class, change accordingly
    )

    model = SimpleRNNClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        # num_classes=num_classes,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Assume optimizer and model initialization remains unchanged
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.01)
    # Placeholder for a chosen scheduler, you might choose a different one based on your requirements
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Placeholder variables for demonstration
    initial_lr = 1e-7
    target_lr = 1e-6
    warmup_steps = 50
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n_elements = 0

        for step, (batch_embeddings, batch_labels) in enumerate(train_loader):
            batch_embeddings, batch_labels = batch_embeddings.to(
                device
            ), batch_labels.to(device)
            batch_embeddings = batch_embeddings.transpose(
                0, 1
            )  # Transpose for Transformer

            # Warmup logic
            if epoch * len(train_loader) + step < warmup_steps:
                lr_scale = (epoch * len(train_loader) + step + 1) / warmup_steps
                lr = initial_lr + lr_scale * (target_lr - initial_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            elif epoch * len(train_loader) + step == warmup_steps:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = target_lr

            optimizer.zero_grad()

            # Assuming the padding in batch_embeddings is represented by all-zero vectors
            src_mask = (batch_embeddings.sum(dim=-1) == 0).transpose(0, 1)
            # output = model(batch_embeddings, src_mask=src_mask)
            output = model(batch_embeddings)
            output = output.view(-1)

            labels_flat = batch_labels.view(-1)
            loss_mask = (
                labels_flat != -100
            )  # Mask to exclude padded elements in loss calculation
            valid_output = output[loss_mask]
            valid_labels = labels_flat[loss_mask]

            loss = criterion(valid_output, valid_labels).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * loss_mask.sum().item()
            n_elements += loss_mask.sum().item()

        # After warmup, adjust learning rate based on the scheduler
        if epoch >= warmup_steps / len(train_loader):
            scheduler.step()

        average_loss = total_loss / n_elements
        print(f"Epoch {epoch+1}, Training Average Loss: {average_loss}")

        # Evaluate on dev data after each training epoch
        dev_loss, dev_accuracy, dev_f1 = evaluate(model, dev_loader, device)
        print(
            f"Epoch {epoch+1}, Dev Loss: {dev_loss}, Dev Accuracy: {dev_accuracy}, Dev F1 Score: {dev_f1}"
        )
