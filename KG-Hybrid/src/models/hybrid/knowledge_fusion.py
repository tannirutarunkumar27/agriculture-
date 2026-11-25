import math
import torch
import torch.nn as nn


SUPPORTED_KNOWLEDGE_POLICIES = {"fixed", "anchored_trainable"}

class KnowledgeGuidedFusion(nn.Module):
    """
    Core novelty: Gated fusion of XGBoost and FT-Transformer features
    with sample-conditioned agronomic knowledge context.

    Data Flow:
      concat = [f_xgb, f_ft]                            # (batch, 192)
      gate = sigmoid(MLP(concat))                       # (batch, 192)
      H_k = knowledge_encoder(K)                        # (num_crops, dim_k) = (22, 64)
      query = query_proj(concat)                        # (batch, dim_k) = (batch, 64)
      scores = (query @ H_k.T) / sqrt(dim_k)            # (batch, num_crops) = (batch, 22)
      attn = softmax(scores, dim=-1)                    # (batch, num_crops) = (batch, 22)
      context = attn @ H_k                              # (batch, dim_k) = (batch, 64)
      k_bias = context_proj(context)                    # (batch, 192)
      f_pre = gate * concat + beta * k_bias             # (batch, 192)
      f_fused = LayerNorm(f_pre)                        # (batch, 192)
    """
    def __init__(
        self,
        dim_xgb: int = 128,
        dim_ft: int = 64,
        num_crops: int = 22,
        num_features: int = 7,
        knowledge_matrix = None,
        dim_k: int = 64,
        beta: float = 0.1,
        knowledge_policy: str = "anchored_trainable",
    ):
        super().__init__()
        self.dim_xgb = dim_xgb
        self.dim_ft = dim_ft
        self.dim_total = dim_xgb + dim_ft
        self.num_crops = num_crops
        self.num_features = num_features
        self.dim_k = dim_k
        self.beta = beta
        if knowledge_policy not in SUPPORTED_KNOWLEDGE_POLICIES:
            raise ValueError(
                f"Unsupported knowledge_policy={knowledge_policy!r}. "
                f"Expected one of {sorted(SUPPORTED_KNOWLEDGE_POLICIES)}."
            )
        self.knowledge_policy = knowledge_policy

        # Gated network: decides how much to trust each branch
        self.gate = nn.Sequential(
            nn.Linear(self.dim_total, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.dim_total),
            nn.Sigmoid()
        )

        # Keep the validated agronomic matrix as a persistent prior.  A training
        # run must explicitly choose whether this prior is fixed or adapted with
        # an L2 anchor; unconstrained adaptation is intentionally unsupported.
        if knowledge_matrix is None:
            raise ValueError(
                "Authoritative agronomic knowledge_matrix is required for KnowledgeGuidedFusion. "
                "Silent random fallback is disallowed; numeric validation does not establish source provenance. "
                "Provide a validated (num_crops, num_features) matrix or load via load_authoritative_knowledge_matrix()."
            )
        init_tensor = torch.as_tensor(knowledge_matrix, dtype=torch.float32).clone()
        expected_shape = (self.num_crops, self.num_features)
        if tuple(init_tensor.shape) != expected_shape:
            raise ValueError(
                f"knowledge_matrix must have shape {expected_shape}, got {tuple(init_tensor.shape)}."
            )
        self.register_buffer("knowledge_prior", init_tensor.clone(), persistent=True)
        if not torch.isfinite(init_tensor).all() or (init_tensor < 0).any() or (init_tensor > 1).any():
            raise ValueError("knowledge_matrix must contain finite prior weights in [0, 1]")
        if knowledge_policy == "fixed":
            self.register_buffer("knowledge_matrix", init_tensor.clone(), persistent=True)
        else:
            self.knowledge_matrix = nn.Parameter(init_tensor.clone(), requires_grad=True)

        # Step B: Encode each crop's agronomic profile independently
        self.knowledge_encoder = nn.Linear(num_features, self.dim_k)

        # Step C: Sample query projection from multimodal concatenated representation
        self.query_proj = nn.Linear(self.dim_total, self.dim_k)

        # Step E: Project aggregated sample-specific knowledge context back to bias vector
        self.context_proj = nn.Linear(self.dim_k, self.dim_total)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.dim_total)

    @property
    def query_projection(self):
        """Alias for query projection module."""
        return self.query_proj

    @property
    def knowledge_context_projection(self):
        """Alias for knowledge context projection module."""
        return self.context_proj

    def knowledge_anchor_loss(self) -> torch.Tensor:
        """Mean-squared deviation of an adapted matrix from its agronomic prior.

        The caller applies the explicitly recorded coefficient lambda_K.  For a
        fixed matrix this is identically zero and has no gradient path.
        """
        return torch.mean((self.knowledge_matrix - self.knowledge_prior) ** 2)

    def knowledge_drift_metrics(self) -> dict[str, float | str]:
        """Return serializable diagnostics for a checkpoint or training log."""
        with torch.no_grad():
            delta = (self.knowledge_matrix - self.knowledge_prior).abs()
            return {
                "knowledge_policy": self.knowledge_policy,
                "mean_abs_change": float(delta.mean().item()),
                "max_abs_change": float(delta.max().item()),
                "current_min": float(self.knowledge_matrix.min().item()),
                "current_max": float(self.knowledge_matrix.max().item()),
            }

    def _compute_knowledge_bias(self, concat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute an input-conditioned knowledge bias and crop attention."""
        knowledge_profiles = self.knowledge_encoder(self.knowledge_matrix)
        query = self.query_proj(concat)
        scores = torch.matmul(query, knowledge_profiles.t()) / math.sqrt(self.dim_k)
        attention_weights = torch.softmax(scores, dim=-1)
        knowledge_context = torch.matmul(attention_weights, knowledge_profiles)
        return self.context_proj(knowledge_context), attention_weights

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Load legacy checkpoints that predate the persisted ``knowledge_prior``.

        The supplied prior remains the reference for drift diagnostics. Loading
        an older checkpoint cannot establish its historical regularization policy.
        """
        prior_key = prefix + "knowledge_prior"
        matrix_key = prefix + "knowledge_matrix"
        if prior_key not in state_dict and matrix_key in state_dict:
            state_dict[prior_key] = self.knowledge_prior.detach().clone()
            self.checkpoint_prior_provenance = "legacy_checkpoint_prior_not_persisted"
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, f_xgb, f_ft, return_attention=False):
        """
        Args:
            f_xgb: (batch, dim_xgb) - XGBoost decision features
            f_ft: (batch, dim_ft) - FT-Transformer context features
            return_attention: bool - Whether to return attention weights
        Returns:
            f_fused: (batch, dim_total) - Fused representation
            gate: (batch, dim_total) - Gate values for analysis
            (optional) attention_weights: (batch, num_crops) - Crop compatibility weights
        """
        # Step A: Multimodal concatenated representation
        concat = torch.cat([f_xgb, f_ft], dim=1)  # (batch, dim_total)

        # 1. Compute adaptive gate (Problem 1 fixed)
        gate = self.gate(concat)  # (batch, dim_total)

        # 2-5. Sample-conditioned crop attention over the knowledge profiles.
        k_bias, attention_weights = self._compute_knowledge_bias(concat)

        # 6. Step F: Fused representation with sample-specific knowledge contribution
        f_pre = gate * concat + self.beta * k_bias

        # Layer norm for stability
        f_fused = self.layer_norm(f_pre)

        self.last_attention_weights = attention_weights.detach()

        if return_attention:
            return f_fused, gate, attention_weights
        return f_fused, gate
