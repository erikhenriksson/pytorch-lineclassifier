import torch
import torch.nn as nn


class TransformerWithPositionalEmbeddings(nn.Module):
    def __init__(
        self, embedding_dim, max_len, num_heads, num_encoder_layers, num_classes
    ):
        super(TransformerWithPositionalEmbeddings, self).__init__()
        self.line_embeddings = nn.Linear(
            embedding_dim, embedding_dim
        )  # Assuming input embeddings are already of correct dimension
        self.positional_embeddings = nn.Embedding(max_len, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.max_len = max_len

    def forward(self, x):
        # x: [batch_size, seq_length, embedding_dim]
        seq_length = x.size(1)
        positions = torch.arange(
            0, seq_length, dtype=torch.long, device=x.device
        )  # [seq_length]
        pos_embeddings = self.positional_embeddings(
            positions
        )  # [seq_length, embedding_dim]

        # Add (or concatenate) positional embeddings to line embeddings
        x = x + pos_embeddings.unsqueeze(0)  # Broadcasting addition

        x = x.permute(
            1, 0, 2
        )  # Rearrange to [seq_length, batch_size, embedding_dim] for Transformer
        x = self.transformer_encoder(x)

        # Apply classifier to each position
        logits = self.classifier(
            x.permute(1, 0, 2)
        )  # Back to [batch_size, seq_length, num_classes]
        return torch.sigmoid(logits)
