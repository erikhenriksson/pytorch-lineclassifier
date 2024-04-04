import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DocumentClassifier(nn.Module):
    def __init__(
        self, embedding_dim, nhead, nhid, nlayers, dropout=0.5, classification_head=None
    ):
        super(DocumentClassifier, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=nhid, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer, num_layers=nlayers
        )
        # Use the provided classification head if available; otherwise, initialize a new linear layer
        if classification_head is not None:
            self.decoder = classification_head
        else:
            self.decoder = nn.Linear(
                embedding_dim, 1
            )  # Fallback to default if no external head is provided

    def forward(self, src, src_mask=None):
        encoded_src = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.decoder(encoded_src)
        return output


class SimpleRNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1):
        super(SimpleRNNClassifier, self).__init__()
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings):
        # embeddings shape: (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeddings)
        logits = self.classifier(lstm_out)
        # Return logits directly without applying sigmoid here; shape: (batch_size, seq_len, 1)
        return logits
