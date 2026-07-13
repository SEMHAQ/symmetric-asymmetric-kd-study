"""Knowledge Distillation loss functions.

Taxonomy:
  Symmetric KD — direct one-to-one alignment between T and S representations:
    KD (KL div), FitNet (L2 feature), AT (attention)

  Asymmetric KD — structural/relational knowledge, not direct one-to-one mapping:
    SP (pairwise similarity), CC (correlation congruence), RKD (relational)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# Symmetric KD Methods
# ====================================================================

class KDLoss(nn.Module):
    """Hinton's Knowledge Distillation loss (output-level, symmetric).

    L = alpha * KL(soft_t || soft_s) + (1-alpha) * CE(hard_s, target)
    """
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, logits_s, logits_t, targets):
        # KD loss: KL divergence between softened distributions
        kl_loss = F.kl_div(
            F.log_softmax(logits_s / self.T, dim=1),
            F.softmax(logits_t / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)

        # CE with hard targets
        ce_loss = F.cross_entropy(logits_s, targets)

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss


class FitNetLoss(nn.Module):
    """FitNet: Hint-based feature learning (feature-level, symmetric).

    Regresses student intermediate features to match teacher features
    after a learned projection.
    """
    def __init__(self, beta=1000.0):
        super().__init__()
        self.beta = beta

    def forward(self, feat_s, feat_t):
        # feat_s, feat_t: (N, C, H, W) feature maps
        # Align via L2 after adaptive pooling to same spatial size
        # (feature maps may differ in spatial dims)
        n, c_s, h_s, w_s = feat_s.shape
        n, c_t, h_t, w_t = feat_t.shape

        if h_s != h_t or w_s != w_t:
            # Adaptive pool to a fixed size
            size = min(h_s, h_t)
            feat_s_pool = F.adaptive_avg_pool2d(feat_s, size)
            feat_t_pool = F.adaptive_avg_pool2d(feat_t, size)
        else:
            feat_s_pool = feat_s
            feat_t_pool = feat_t

        # L2 loss
        loss = F.mse_loss(feat_s_pool, feat_t_pool) * self.beta
        return loss


class ATLoss(nn.Module):
    """Attention Transfer loss (attention-level, symmetric).

    Minimizes MSE between normalized attention maps of teacher and student.
    Attention: sum of squared activations across channels.
    """
    def __init__(self, beta=1000.0):
        super().__init__()
        self.beta = beta

    @staticmethod
    def _attention_map(x):
        """Sum of squared activations across channels, then normalize."""
        # x: (N, C, H, W)
        attn = x.pow(2).sum(dim=1)  # (N, H, W)
        # Normalize by max
        n, h, w = attn.shape
        attn_flat = attn.view(n, -1)
        norm = attn_flat.max(dim=1, keepdim=True)[0] + 1e-8
        attn_flat = attn_flat / norm
        return attn_flat.view(n, 1, h, w)

    def forward(self, feat_s, feat_t):
        attn_s = self._attention_map(feat_s)
        attn_t = self._attention_map(feat_t)
        loss = F.mse_loss(attn_s, attn_t) * self.beta
        return loss


# ====================================================================
# Asymmetric KD Methods
# ====================================================================

class SPLoss(nn.Module):
    """Similarity Preserving loss (relational, asymmetric).

    Preserves pairwise similarity matrices of samples in a mini-batch
    between teacher and student feature spaces.
    """
    def __init__(self, beta=3000.0):
        super().__init__()
        self.beta = beta

    @staticmethod
    def _similarity(feats):
        """Compute pairwise similarity matrix (N x N)."""
        # feats: (N, D) — flattened features
        feats = feats.view(feats.size(0), -1)
        # Normalize each sample
        feats_norm = F.normalize(feats, p=2, dim=1)
        sim = feats_norm @ feats_norm.T  # (N, N)
        return sim

    def forward(self, feat_s, feat_t):
        # feat_s, feat_t could be feature maps or pooled features
        if feat_s.dim() == 4:
            feat_s = F.adaptive_avg_pool2d(feat_s, 1).view(feat_s.size(0), -1)
            feat_t = F.adaptive_avg_pool2d(feat_t, 1).view(feat_t.size(0), -1)

        G_s = self._similarity(feat_s)
        G_t = self._similarity(feat_t)

        # Normalize similarity matrices
        G_s = G_s / G_s.norm(p=2, dim=1, keepdim=True)
        G_t = G_t / G_t.norm(p=2, dim=1, keepdim=True)

        loss = F.mse_loss(G_s, G_t) * self.beta
        return loss


class CCLoss(nn.Module):
    """Correlation Congruence loss (structural, asymmetric).

    Aligns second-order statistics (correlation matrices) between
    teacher and student feature spaces.
    """
    def __init__(self, beta=0.02, gamma=0.4):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _gram_matrix(x):
        """Compute Gram (correlation) matrix."""
        x = x.view(x.size(0), -1)
        x = x - x.mean(dim=0, keepdim=True)
        # Normalize by Frobenius norm
        gram = x.T @ x / (x.size(0) - 1)
        gram = gram / (gram.norm(p='fro') + 1e-8)
        return gram

    def forward(self, feat_s, feat_t):
        if feat_s.dim() == 4:
            feat_s = F.adaptive_avg_pool2d(feat_s, 1).view(feat_s.size(0), -1)
            feat_t = F.adaptive_avg_pool2d(feat_t, 1).view(feat_t.size(0), -1)

        gram_s = self._gram_matrix(feat_s)
        gram_t = self._gram_matrix(feat_t)

        cc_loss = F.mse_loss(gram_s, gram_t) * self.beta

        return cc_loss


class RKDILoss(nn.Module):
    """Relational Knowledge Distillation — Distance-wise loss (relational, asymmetric).

    Preserves pairwise distances between samples.
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    @staticmethod
    def _pairwise_distance(x):
        """Compute pairwise L2 distances (N x N)."""
        x = x.view(x.size(0), -1)
        x_norm = x.pow(2).sum(dim=1, keepdim=True)
        dist = x_norm + x_norm.T - 2.0 * (x @ x.T)
        dist = F.relu(dist).sqrt()
        # Normalize
        dist = dist / (dist.mean(dim=1, keepdim=True) + 1e-8)
        return dist

    def forward(self, feat_s, feat_t):
        if feat_s.dim() == 4:
            feat_s = F.adaptive_avg_pool2d(feat_s, 1).view(feat_s.size(0), -1)
            feat_t = F.adaptive_avg_pool2d(feat_t, 1).view(feat_t.size(0), -1)

        d_s = self._pairwise_distance(feat_s)
        d_t = self._pairwise_distance(feat_t)
        loss = F.mse_loss(d_s, d_t) * self.beta
        return loss


# ====================================================================
# Loss factory
# ====================================================================

KD_METHODS = {
    'kd':    {'cls': KDLoss,    'category': 'symmetric',   'params': ['temperature', 'alpha']},
    'fitnet':{'cls': FitNetLoss,'category': 'symmetric',   'params': ['beta']},
    'at':    {'cls': ATLoss,    'category': 'symmetric',   'params': ['beta']},
    'sp':    {'cls': SPLoss,    'category': 'asymmetric',  'params': ['beta']},
    'cc':    {'cls': CCLoss,    'category': 'asymmetric',  'params': ['beta', 'gamma']},
    'rkdi':  {'cls': RKDILoss,  'category': 'asymmetric',  'params': ['beta']},
}


def get_kd_loss(method_name, **kwargs):
    """Get KD loss function by name.

    Args:
        method_name: one of 'kd', 'fitnet', 'at', 'sp', 'cc', 'rkdi'
        **kwargs: override default hyperparameters (temperature, alpha, beta, gamma)

    Returns:
        loss_fn: nn.Module
        category: str ('symmetric' or 'asymmetric')
    """
    info = KD_METHODS[method_name]
    # Default params from config
    from config import (KD_TEMPERATURE, KD_ALPHA, FITNET_BETA, AT_BETA,
                        SP_BETA, CC_BETA, CC_GAMMA)

    param_defaults = {
        'temperature': KD_TEMPERATURE,
        'alpha': KD_ALPHA,
        'beta': {
            'fitnet': FITNET_BETA,
            'at': AT_BETA,
            'sp': SP_BETA,
            'cc': CC_BETA,
            'rkdi': 1.0,
        }[method_name],
        'gamma': CC_GAMMA,
    }

    # Override with provided kwargs
    merged = {**param_defaults, **kwargs}
    # Filter only relevant params
    relevant_params = {k: merged[k] for k in info['params']}

    loss_fn = info['cls'](**relevant_params)
    return loss_fn, info['category']
