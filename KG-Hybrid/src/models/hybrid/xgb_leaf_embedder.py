import torch
import torch.nn as nn
import numpy as np

class XGBLeafEmbedder(nn.Module):
    """
    Converts XGBoost leaf indices to trainable embeddings.
    This bridges the gap between tree-based models and neural networks.
    """
    def __init__(
        self,
        num_unique_leaves,
        num_trees=200,
        embedding_dim=128,
        sparse: bool = False,
    ):
        super().__init__()
        self.num_trees = num_trees
        self.embedding_dim = embedding_dim
        
        # EmbeddingBag: averages embeddings of all leaves per sample
        self.embeddings = nn.EmbeddingBag(
            num_embeddings=num_unique_leaves,
            embedding_dim=embedding_dim,
            mode='mean',  # Mean across all trees -> stable scale
            sparse=sparse,
        )
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, leaf_indices):
        """
        Args:
            leaf_indices: torch.LongTensor of shape (batch, num_trees)
        Returns:
            embeddings: torch.Tensor of shape (batch, embedding_dim)
        """
        out = self.embeddings(leaf_indices)
        return self.norm(out)
