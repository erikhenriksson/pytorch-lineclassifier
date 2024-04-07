import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

criterion = nn.BCELoss()


class LSTMForLineClassification(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, num_layers, num_classes, bidirectional=True
    ):
        super(LSTMForLineClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Define the output layer
        lstm_output_dim = hidden_dim if not bidirectional else hidden_dim * 2
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, src):
        # src shape: [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(src)
        # If using a bidirectional LSTM, lstm_out shape will be [batch_size, seq_len, hidden_dim * 2]

        # Pass the output of the LSTM to the classifier
        logits = self.classifier(lstm_out)
        # Apply sigmoid activation to get probabilities
        probabilities = torch.sigmoid(logits)

        return probabilities


class DocumentDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        # Extract embeddings and labels from the document's lines
        embeddings, labels = zip(*document)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with open(cfg.embeddings_file, "rb") as f:
        loaded_data = pickle.load(f)

    # Create separate dataset instances for train, dev, and test
    train_dataset = DocumentDataset([doc for doc in loaded_data["train"] if doc])
    dev_dataset = DocumentDataset([doc for doc in loaded_data["dev"] if doc])
    test_dataset = DocumentDataset([doc for doc in loaded_data["test"] if doc])

    print(train_dataset[0])

    batch_size = 8  # Define your batch size; adjust as necessary

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = LSTMForLineClassification(
        embedding_dim=1024,
        hidden_dim=512,
        num_layers=2,
        num_classes=1,
        bidirectional=True,
    ).to(device)

    print(f"Train len: {len(train_dataloader)}")
    print(f"Dev len: {len(dev_dataloader)}")
    print(f"Test len: {len(test_dataloader)}")

    print("Testing pre-training...")
    avg_loss, f1, accuracy = evaluate(model, test_dataloader, device)
    print(f"Loss: {avg_loss}, F1 Score: {f1}, Accuracy: {accuracy}")

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.lstm_learning_rate, weight_decay=0.01
    )

    best_val_loss = float("inf")
    patience = cfg.patience
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0

        for embeddings, labels in train_dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        print(
            f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {total_loss / len(train_dataloader)}"
        )
        val_loss, f1, accuracy = evaluate(model, dev_dataloader, device)
        print(f"Validation Loss: {val_loss}, F1 Score: {f1}, Accuracy: {accuracy}")

        # Check if the validation loss improved
        if val_loss < best_val_loss:
            print(
                f"Validation loss decreased from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            # Reset patience counter
            patience_counter = 0

            # Save the model
            torch.save(model.state_dict(), "models/lstm_model/best_model.pt")

        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}."
            )

        # Early stopping check
        if patience_counter >= patience:
            print("Stopping early due to no improvement in validation loss.")
            break

    print("Testing..")
    avg_loss, f1, accuracy = evaluate(model, test_dataloader, device)
    print(f"Test Loss: {avg_loss}, F1 Score: {f1}, Accuracy: {accuracy}")
