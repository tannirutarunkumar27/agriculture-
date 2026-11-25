"""
File: tokenizer.py

Purpose:
Feature Tokenizer for FT-Transformer.

Each numerical feature is converted into a learnable embedding.
The output tokens are then passed to the Transformer encoder.
"""

import torch
import torch.nn as nn

from .config import (
    NUM_FEATURES,
    EMBEDDING_DIM
)


class FeatureTokenizer(nn.Module):
    """
    Numerical Feature Tokenizer.

    Input:
        (batch_size, num_features)

    Output:
        (batch_size, num_features, embedding_dim)
    """

    def __init__(
        self,
        num_features=NUM_FEATURES,
        embedding_dim=EMBEDDING_DIM
    ):

        super().__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # Learnable weight for each feature
        self.weight = nn.Parameter(
            torch.empty(
                num_features,
                embedding_dim
            )
        )

        # Learnable bias for each feature
        self.bias = nn.Parameter(
            torch.empty(
                num_features,
                embedding_dim
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Xavier Initialization
        """

        nn.init.xavier_uniform_(self.weight)

        nn.init.zeros_(self.bias)

    def forward(self, x):

        """
        x shape:
            (batch_size, num_features)

        Output:
            (batch_size, num_features, embedding_dim)
        """

        if x.dim() != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input shape (batch_size, {self.num_features}), "
                f"got {tuple(x.shape)}"
            )

        x = x.unsqueeze(-1)

        tokens = x * self.weight + self.bias

        return tokens


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    sample = torch.randn(
        32,
        NUM_FEATURES
    )

    tokenizer = FeatureTokenizer()

    output = tokenizer(sample)

    print("=" * 60)
    print("FEATURE TOKENIZER TEST")
    print("=" * 60)

    print(f"Input Shape  : {sample.shape}")
    print(f"Output Shape : {output.shape}")

    print("\nExpected Output Shape:")
    print(f"(32, {NUM_FEATURES}, {EMBEDDING_DIM})")