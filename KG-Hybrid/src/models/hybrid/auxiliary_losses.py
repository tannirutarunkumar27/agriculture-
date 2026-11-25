import torch
import torch.nn.functional as F

def distillation_loss(logits_xgb, logits_ft, logits_fused, temperature=4.0):
    """
    Knowledge Distillation Loss:
    Transfer knowledge from both branches to the fused model.
    """
    # Soften the distributions with temperature
    p_xgb = F.log_softmax(logits_xgb / temperature, dim=1)
    p_ft = F.log_softmax(logits_ft / temperature, dim=1)
    p_fused = F.log_softmax(logits_fused / temperature, dim=1)
    
    # KL Divergence
    loss_xgb = F.kl_div(p_fused, p_xgb, reduction='batchmean', log_target=True)
    loss_ft = F.kl_div(p_fused, p_ft, reduction='batchmean', log_target=True)
    
    # Scale by temperature^2 to keep gradient magnitude consistent
    return (loss_xgb + loss_ft) * (temperature ** 2)

def contrastive_loss(f_fused, labels, temperature=0.5):
    """
    Supervised Contrastive Learning:
    Pull same-crop embeddings closer, push different-crop embeddings apart.
    """
    batch_size = f_fused.size(0)
    f_norm = F.normalize(f_fused, dim=1)  # (batch, dim)
    
    # Similarity matrix
    sim_matrix = f_norm @ f_norm.T / temperature  # (batch, batch)
    
    # Create mask: 1 for same class, 0 for different
    labels = labels.view(-1, 1)
    mask = (labels == labels.T).float()  # (batch, batch)
    
    # Remove self-similarity
    mask = mask - torch.eye(batch_size, device=mask.device)
    
    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    sum_exp = exp_sim.sum(dim=1, keepdim=True)  # (batch, 1)
    log_prob = sim_matrix - torch.log(sum_exp + 1e-8)
    
    # Mean over positive pairs
    positive_log_prob = (mask * log_prob).sum(dim=1)
    positive_count = mask.sum(dim=1)
    loss = -(positive_log_prob / (positive_count + 1e-8)).mean()
    
    return loss

def combined_loss(logits_fused, logits_xgb, logits_ft, labels,
                  f_fused, alpha_ce=1.0, alpha_distill=0.5, alpha_contrast=0.1,
                  knowledge_anchor_loss=None, alpha_knowledge=0.0):
    """
    Combined loss function for KG-Hybrid training.

    ``knowledge_anchor_loss`` is the mean squared drift of a trainable
    agronomic matrix from its validated initialization.  Passing it explicitly
    keeps the policy visible in each training configuration and avoids silently
    treating a learned matrix as a fixed prior.
    """
    # Cross-entropy loss
    loss_ce = F.cross_entropy(logits_fused, labels)
    
    # Distillation loss (if branches are available)
    if logits_xgb is not None and logits_ft is not None:
        loss_distill = distillation_loss(logits_xgb, logits_ft, logits_fused)
    else:
        loss_distill = torch.tensor(0.0, device=logits_fused.device)
    
    # Contrastive loss
    loss_contrast = contrastive_loss(f_fused, labels)
    
    if knowledge_anchor_loss is None:
        knowledge_anchor_loss = torch.zeros((), device=logits_fused.device, dtype=loss_ce.dtype)
    if not torch.is_tensor(knowledge_anchor_loss):
        raise TypeError("knowledge_anchor_loss must be a torch.Tensor or None")
    if alpha_knowledge < 0.0:
        raise ValueError("alpha_knowledge must be non-negative")

    total_loss = (
        alpha_ce * loss_ce
        + alpha_distill * loss_distill
        + alpha_contrast * loss_contrast
        + alpha_knowledge * knowledge_anchor_loss
    )
    
    return total_loss, loss_ce, loss_distill, loss_contrast
