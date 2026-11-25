"""
File: model.py

Purpose:
Research-grade FT-Transformer architecture for crop recommendation.

Citation:
Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).
"Revisiting Deep Learning Models for Tabular Data."
Advances in Neural Information Processing Systems (NeurIPS), 34, 18932-18943.

Architecture:
Input Features
        │
Feature Tokenizer
        │
CLS Token
        │
Transformer Encoder × N
        │
CLS Representation
        │
MLP Classifier
        │
Crop Prediction
"""

import torch
import torch.nn as nn

from .config import (
    NUM_FEATURES,
    NUM_CLASSES,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FF_DIM,
    DROPOUT,
)

from .tokenizer import FeatureTokenizer
from .cls_token import CLSToken
from .encoder import TransformerEncoder


class FTTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        # Feature Tokenizer
        self.tokenizer = FeatureTokenizer(
            num_features=NUM_FEATURES,
            embedding_dim=EMBEDDING_DIM
        )

        # CLS Token
        self.cls_token = CLSToken(
            embedding_dim=EMBEDDING_DIM
        )

        # Transformer Encoder Stack
        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    embedding_dim=EMBEDDING_DIM,
                    num_heads=NUM_HEADS,
                    ff_dim=FF_DIM,
                    dropout=DROPOUT
                )
                for _ in range(NUM_LAYERS)
            ]
        )

        # Classification Head
        self.classifier = nn.Sequential(

            nn.Linear(
                EMBEDDING_DIM,
                128
            ),

            nn.GELU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    def extract_features(self, x):
        """Extract 64-dim CLS embedding features before classifier."""
        if x.dim() != 2 or x.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected input shape (batch_size, {NUM_FEATURES}), "
                f"got {tuple(x.shape)}"
            )
        x = self.tokenizer(x)
        x = self.cls_token(x)
        for encoder in self.encoders:
            x, _ = encoder(x)
        return x[:, 0]

    def forward(self, x, return_attention=False, return_features=False):

        if x.dim() != 2 or x.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected input shape (batch_size, {NUM_FEATURES}), "
                f"got {tuple(x.shape)}"
            )

        # ------------------------------------------
        # Feature Tokenization
        # ------------------------------------------
        x = self.tokenizer(x)

        # ------------------------------------------
        # Add CLS Token
        # ------------------------------------------
        x = self.cls_token(x)

        attention_maps = []

        # ------------------------------------------
        # Transformer Encoder Stack
        # ------------------------------------------
        for encoder in self.encoders:

            x, attention = encoder(x)

            attention_maps.append(attention)

        # ------------------------------------------
        # CLS Representation
        # ------------------------------------------
        cls_embedding = x[:, 0]

        if return_features:
            return cls_embedding

        # ------------------------------------------
        # Classification
        # ------------------------------------------
        logits = self.classifier(cls_embedding)

        if return_attention:
            return logits, attention_maps

        return logits


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    model = FTTransformer()

    sample = torch.randn(
        32,
        NUM_FEATURES
    )

    logits, attention = model(
        sample,
        return_attention=True
    )

    print("=" * 60)
    print("FT-TRANSFORMER TEST")
    print("=" * 60)

    print(f"Input Shape       : {sample.shape}")
    print(f"Logits Shape      : {logits.shape}")
    print(f"Encoder Layers    : {len(attention)}")
    print(f"Attention Shape   : {attention[0].shape}")

    print("\nExpected Shapes")
    print(f"Logits            : (32, {NUM_CLASSES})")
    print(
        f"Attention         : "
        f"(32, {NUM_HEADS}, {NUM_FEATURES+1}, {NUM_FEATURES+1})"
    )