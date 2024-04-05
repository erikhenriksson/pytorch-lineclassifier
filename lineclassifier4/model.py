import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class TransformerForLineClassification(nn.Module):
    def __init__(self, embedding_dim, nhead, num_encoder_layers, num_classes):
        super(TransformerForLineClassification, self).__init__()
        self.positional_encoding = PositionalEncoding(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, src):
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        logits = self.classifier(output)
        return torch.sigmoid(logits)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough PE matrix up front
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Make PE not trainable and persistently saved, but not as a parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0), :]
        return x
