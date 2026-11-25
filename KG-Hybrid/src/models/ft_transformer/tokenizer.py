"""
File: tokenizer.py

Purpose:
Public-facing tokenizer module for the FT-Transformer package.

Re-exports FeatureTokenizer from feature_tokenizer.py, providing
the canonical import path used by model.py:

    from .tokenizer import FeatureTokenizer

Architecture Role:
    Each raw numerical feature (N, P, K, temperature, humidity,
    ph, rainfall) is projected into a dense embedding vector of
    dimension `embedding_dim` using a learnable linear layer.

    This converts the flat (batch × 7) input into a sequence of
    token embeddings (batch × 7 × embedding_dim), which the
    Transformer encoder can then process with self-attention.

    Input  : (batch_size, num_features)         e.g. (32, 7)
    Output : (batch_size, num_features, d_model) e.g. (32, 7, 64)

Example:
    >>> import torch
    >>> from tokenizer import FeatureTokenizer
    >>> tokenizer = FeatureTokenizer(num_features=7, embedding_dim=64)
    >>> x = torch.randn(32, 7)
    >>> tokens = tokenizer(x)
    >>> print(tokens.shape)   # torch.Size([32, 7, 64])

See Also:
    feature_tokenizer.py  — full FeatureTokenizer implementation
    cls_token.py          — prepends [CLS] token to the token sequence
    encoder.py            — Transformer encoder that processes the tokens
    model.py              — assembles all components into FTTransformer
"""

from .feature_tokenizer import FeatureTokenizer

__all__ = ["FeatureTokenizer"]
