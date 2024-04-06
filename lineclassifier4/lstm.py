import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Example model parameters
embedding_dim = 1024  # Dimensionality of the input embeddings
hidden_dim = 512  # Hidden dimension size of the LSTM
num_layers = 4  # Number of LSTM layers
num_classes = 1  # For binary classification
bidirectional = True  # Specify if you want to use a bidirectional LSTM

lstm_model = LSTMForLineClassification(
    embedding_dim, hidden_dim, num_layers, num_classes, bidirectional
)
