import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        # Initialize a learnable embedding for positions
        self.positional_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Generate a tensor of positions
        positions = (
            torch.arange(x.size(0), device=x.device)
            .unsqueeze(1)
            .expand(x.size(0), x.size(1))
        )
        # Retrieve the positional embeddings for the positions
        pos_embed = self.positional_embedding(positions)
        # Add the positional embeddings to the input embeddings
        return x + pos_embed


class TransformerForLineClassification(nn.Module):
    def __init__(
        self, embedding_dim, nhead, num_encoder_layers, num_classes, max_len=5000
    ):
        super(TransformerForLineClassification, self).__init__()
        self.positional_encoding = LearnablePositionalEncoding(embedding_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, src):
        # src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        logits = self.classifier(output)
        return torch.sigmoid(logits)
