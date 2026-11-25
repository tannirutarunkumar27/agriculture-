"""
File: cls_token.py

Purpose:
Learnable CLS (Classification) token for the FT-Transformer.

The CLS token is prepended to the feature tokens and learns to
aggregate information from all features through self-attention.
The final CLS representation is used for classification.
"""

import torch
import torch.nn as nn


class CLSToken(nn.Module):
    """
    Learnable CLS Token.

    Input:
        (batch_size, num_features, embedding_dim)

    Output:
        (batch_size, num_features + 1, embedding_dim)
    """

    def __init__(self, embedding_dim):
        super().__init__()

        # Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape:
            (batch_size, num_features, embedding_dim)

        Returns
        -------
        Tensor
            Shape:
            (batch_size, num_features + 1, embedding_dim)
        """

        batch_size = x.size(0)

        # Expand CLS token for every sample in batch
        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        # Prepend CLS token
        x = torch.cat(
            [cls, x],
            dim=1
        )

        return x


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    sample = torch.randn(
        32,
        7,
        64
    )

    cls_layer = CLSToken(
        embedding_dim=64
    )

    output = cls_layer(sample)

    print("=" * 60)
    print("CLS TOKEN TEST")
    print("=" * 60)

    print(f"Input Shape  : {sample.shape}")
    print(f"Output Shape : {output.shape}")

    print("\nExpected Output Shape:")
    print("(32, 8, 64)")