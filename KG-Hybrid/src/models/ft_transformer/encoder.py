"""
File: encoder.py

Purpose:
Transformer Encoder Block for FT-Transformer.

This encoder performs:
1. Multi-Head Self Attention
2. Residual Connection
3. Layer Normalization
4. Feed Forward Network
5. Residual Connection
6. Layer Normalization

It also returns attention weights for explainability.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import (
    EMBEDDING_DIM,
    NUM_HEADS,
    FF_DIM,
    DROPOUT
)


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    ):

        super().__init__()

        # Multi-head Self Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.norm2 = nn.LayerNorm(embedding_dim)

        # Feed Forward Network
        self.ffn = nn.Sequential(

            nn.Linear(
                embedding_dim,
                ff_dim
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                ff_dim,
                embedding_dim
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:

        if x.dim() != 3 or x.shape[2] != self.attention.embed_dim:
            raise ValueError(
                f"Expected input shape (batch_size, seq_len, {self.attention.embed_dim}), "
                f"got {tuple(x.shape)}"
            )

        # ==========================================
        # Multi-Head Attention
        # ==========================================

        attention_output, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False
        )

        # Residual Connection
        x = self.norm1(
            x + self.dropout(attention_output)
        )

        # ==========================================
        # Feed Forward Network
        # ==========================================

        ffn_output = self.ffn(x)

        # Residual Connection
        output = self.norm2(
            x + self.dropout(ffn_output)
        )

        return output, attention_weights


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    sample = torch.randn(
        32,
        8,
        EMBEDDING_DIM
    )

    encoder = TransformerEncoder()

    output, attention = encoder(sample)

    print("=" * 60)
    print("TRANSFORMER ENCODER TEST")
    print("=" * 60)

    print(f"Input Shape      : {sample.shape}")
    print(f"Output Shape     : {output.shape}")
    print(f"Attention Shape  : {attention.shape}")

    print("\nExpected:")
    print(f"Output    : (32, 8, {EMBEDDING_DIM})")
    print(f"Attention : (32, {NUM_HEADS}, 8, 8)")