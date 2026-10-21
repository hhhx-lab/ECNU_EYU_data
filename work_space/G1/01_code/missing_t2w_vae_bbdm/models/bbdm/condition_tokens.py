
import torch.nn as nn
import torch

class ConditionTokens(nn.Module):
    def __init__(self, num_conditions, embed_dim=256, hidden_dim=128, use_self_attention=False, n_heads=8, n_layers=1):
        """
        Projects each condition into a latent space and optionally applies a Transformer Encoder
        to enable self-attention among tokens.

        Parameters:
          - num_conditions: Total number of conditions to consider.
          - embed_dim: Dimension of the resulting embedding for each token.
          - hidden_dim: Dimension of the hidden layers in each condition's projection.
          - use_self_attention: Boolean flag to apply self-attention among tokens.
          - n_heads: Number of attention heads in the Transformer Encoder.
          - n_layers: Number of layers in the Transformer Encoder.
        """
        super(ConditionTokens, self).__init__()
        self.num_conditions = num_conditions
        self.embed_dim = embed_dim
        self.use_self_attention = use_self_attention

        # Projection module for each condition with hidden layers
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ) for _ in range(num_conditions)
        ])

        # Transformer Encoder for self-attention among tokens
        if self.use_self_attention:
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, conditions):
        """
        conditions: Tensor of shape (batch_size, num_conditions) where each column corresponds to a specific condition.
        """
        # # Project each condition into its latent space
        # tokens = torch.stack([
        #     self.projections[i](conditions[:, i:i+1]) for i in range(self.num_conditions)
        # ], dim=1)  # (batch_size, num_conditions, embed_dim)

        tokens = torch.stack([
            self.projections[i](conditions[:, i:i+1,0]) for i in range(self.num_conditions)
        ], dim=1)  # (batch_size, num_conditions, embed_dim)

        # Optionally apply self-attention among tokens
        if self.use_self_attention:
            tokens = tokens.transpose(0, 1)  # (num_conditions, batch_size, embed_dim)
            tokens = self.transformer_encoder(tokens)
            tokens = tokens.transpose(0, 1)  # (batch_size, num_conditions, embed_dim)

        return tokens
