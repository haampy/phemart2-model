from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_BCE = nn.BCEWithLogitsLoss()
_CE = nn.CrossEntropyLoss()


def _build_positive_pairs(
    positive_disease_ids_per_variant: List[List[int]],
    num_cols: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_parts: List[torch.Tensor] = []
    col_parts: List[torch.Tensor] = []
    valid_rows: List[int] = []
    for row_idx, positives in enumerate(positive_disease_ids_per_variant):
        if not positives:
            continue
        idx = torch.as_tensor(positives, device=device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < num_cols)]
        if idx.numel() == 0:
            continue
        row_parts.append(torch.full((idx.numel(),), row_idx, device=device, dtype=torch.long))
        col_parts.append(idx)
        valid_rows.append(int(row_idx))

    if not col_parts:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, empty

    return (
        torch.cat(row_parts, dim=0),
        torch.cat(col_parts, dim=0),
        torch.tensor(valid_rows, device=device, dtype=torch.long),
    )


def main_multi_positive_bce_loss(
    variant_emb: torch.Tensor,
    all_disease_emb: torch.Tensor,
    positive_disease_ids_per_variant: List[List[int]],
    temperature: float = 0.15,
    logit_scale: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Main task loss with logits [B, D] and multi-hot targets [B, D]."""
    logits = variant_emb @ all_disease_emb.t()
    if logit_scale is None:
        logits = logits / max(temperature, 1e-6)
    else:
        logits = logits * logit_scale
    row_idx, col_idx, valid_rows = _build_positive_pairs(
        positive_disease_ids_per_variant,
        num_cols=logits.shape[1],
        device=logits.device,
    )
    if valid_rows.numel() == 0:
        return torch.zeros([], device=logits.device, dtype=logits.dtype)
    logits_f = logits.float()
    per_row_sum = F.softplus(logits_f).sum(dim=1)
    per_row_sum.index_add_(0, row_idx, -logits_f[row_idx, col_idx])
    per_row = per_row_sum.index_select(0, valid_rows) / max(logits.shape[1], 1)
    if sample_weights is None:
        return per_row.mean()
    w = sample_weights.index_select(0, valid_rows).clamp_min(0.0)
    return (per_row * w).sum() / w.sum().clamp_min(1e-8)


def main_multi_positive_softmax_loss(
    variant_emb: torch.Tensor,
    all_disease_emb: torch.Tensor,
    positive_disease_ids_per_variant: List[List[int]],
    temperature: float = 0.15,
    logit_scale: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Multi-positive retrieval loss with logits [B, D] over full disease bank."""
    logits = variant_emb @ all_disease_emb.t()
    if logit_scale is None:
        logits = logits / max(temperature, 1e-6)
    else:
        logits = logits * logit_scale
    row_idx, col_idx, valid_rows = _build_positive_pairs(
        positive_disease_ids_per_variant,
        num_cols=logits.shape[1],
        device=logits.device,
    )
    if valid_rows.numel() == 0:
        return torch.zeros([], device=logits.device, dtype=logits.dtype)
    logits_f = logits.float()
    den = torch.logsumexp(logits_f, dim=1)
    pos_row_max = torch.full(
        (logits.shape[0],),
        float("-inf"),
        device=logits.device,
        dtype=torch.float32,
    )
    pos_row_max.scatter_reduce_(
        0,
        row_idx,
        logits_f[row_idx, col_idx],
        reduce="amax",
        include_self=True,
    )
    pos_exp = torch.exp(logits_f[row_idx, col_idx] - pos_row_max.index_select(0, row_idx))
    pos_exp_sum = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    pos_exp_sum.index_add_(0, row_idx, pos_exp)
    num = pos_row_max.index_select(0, valid_rows) + torch.log(
        pos_exp_sum.index_select(0, valid_rows).clamp_min(1e-12)
    )
    per_row = -(num - den.index_select(0, valid_rows))

    # Label smoothing: mix hard target loss with uniform distribution penalty
    if label_smoothing > 0.0:
        n_classes = logits_f.shape[1]
        uniform_loss = -logits_f.mean(dim=1) + den  # -mean(logits) + logsumexp
        uniform_term = uniform_loss.index_select(0, valid_rows)
        per_row = (1.0 - label_smoothing) * per_row + label_smoothing * uniform_term

    if sample_weights is None:
        return per_row.mean()
    w = sample_weights.index_select(0, valid_rows).clamp_min(0.0)
    return (per_row * w).sum() / w.sum().clamp_min(1e-8)


def domain_sampled_infonce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_negatives: int = 63,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape={tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape={tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Batch size mismatch between logits and labels")
    if logits.numel() == 0:
        return torch.zeros([], device=logits.device, dtype=logits.dtype)

    num_classes = int(logits.shape[1])
    if num_classes <= 1:
        return torch.zeros([], device=logits.device, dtype=logits.dtype)

    k = min(max(int(num_negatives), 1), num_classes - 1)
    label_col = labels.view(-1, 1)

    # Sample a different without-replacement negative prototype set for each row.
    rand = torch.rand(
        logits.shape[0],
        num_classes,
        device=logits.device,
        dtype=torch.float32,
    )
    rand.scatter_(1, label_col, -1.0)
    neg_idx = rand.topk(k=k, dim=1).indices

    pos_logits = logits.gather(1, label_col)
    neg_logits = logits.gather(1, neg_idx)
    sampled_logits = torch.cat([pos_logits, neg_logits], dim=1)
    sampled_targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return _CE(sampled_logits, sampled_targets)


def func_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    column_scales: Optional[torch.Tensor] = None,
    loss_type: str = "smooth_l1",
    smooth_l1_beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """单轴回归 loss，支持 per-column mask。"""
    diff = pred - target
    if column_scales is not None:
        diff = diff / column_scales.clamp_min(1e-6)

    if loss_type == "smooth_l1":
        err = F.smooth_l1_loss(
            diff,
            torch.zeros_like(diff),
            reduction="none",
            beta=smooth_l1_beta,
        )
    else:
        err = diff**2

    num = (err * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def func_mechanism_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MutPred2 多标签分类 loss (BCE with logits)。mask 是 per-sample (1D)。"""
    if mask.sum() < 1:
        return torch.zeros([], device=pred.device, dtype=pred.dtype)
    valid = mask > 0.5
    pred_valid = pred[valid]
    target_valid = target[valid]
    pw = torch.full([pred.shape[1]], pos_weight, device=pred.device, dtype=pred.dtype)
    return F.binary_cross_entropy_with_logits(
        pred_valid, target_valid, pos_weight=pw, reduction="mean",
    )


def func_multiaxis_loss(
    preds: Dict[str, torch.Tensor],
    regression_target: torch.Tensor,
    regression_mask: torch.Tensor,
    mechanism_target: torch.Tensor,
    mechanism_mask: torch.Tensor,
    axis_slices: Dict[str, tuple],
    column_scales: Optional[torch.Tensor] = None,
    regression_loss_type: str = "smooth_l1",
    regression_beta: float = 1.0,
    mechanism_pos_weight: float = 3.0,
) -> Dict[str, torch.Tensor]:
    """计算所有 func 轴的 loss，返回 dict。"""
    losses: Dict[str, torch.Tensor] = {}

    for axis_name, (start, end) in axis_slices.items():
        if axis_name not in preds:
            continue
        pred = preds[axis_name]
        tgt = regression_target[:, start:end]
        msk = regression_mask[:, start:end]
        cs = column_scales[start:end] if column_scales is not None else None
        losses[f"func_{axis_name}"] = func_regression_loss(
            pred, tgt, msk,
            column_scales=cs,
            loss_type=regression_loss_type,
            smooth_l1_beta=regression_beta,
        )

    if "mechanism" in preds:
        losses["func_mechanism"] = func_mechanism_loss(
            preds["mechanism"],
            mechanism_target,
            mechanism_mask,
            pos_weight=mechanism_pos_weight,
        )

    return losses


# 保留旧签名的兼容包装
def func_impact_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    column_weights: Optional[torch.Tensor] = None,
    column_scales: Optional[torch.Tensor] = None,
    loss_type: str = "mse",
    smooth_l1_beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    return func_regression_loss(pred, target, mask, column_scales, loss_type, smooth_l1_beta, eps)


def sparse_teacher_kl_from_log_probs(
    log_probs: torch.Tensor,
    teacher_indices: List[torch.Tensor],
    teacher_probs: List[torch.Tensor],
    row_weights: Optional[Sequence[torch.Tensor | float]] = None,
) -> tuple[torch.Tensor, int]:
    """Compute KL(teacher||student) when teacher is sparse.

    Args:
        log_probs: student log-probabilities with shape [B, N]
        teacher_indices: length-B list; each tensor stores selected indices in [0, N)
        teacher_probs: length-B list; each tensor stores probabilities aligned with indices
    """
    if len(teacher_indices) != len(teacher_probs):
        raise ValueError("teacher_indices and teacher_probs must have the same length")
    if len(teacher_indices) != log_probs.shape[0]:
        raise ValueError("teacher list length must match log_probs batch dimension")
    if row_weights is not None and len(row_weights) != log_probs.shape[0]:
        raise ValueError("row_weights length must match log_probs batch dimension")

    row_parts: List[torch.Tensor] = []
    col_parts: List[torch.Tensor] = []
    prob_parts: List[torch.Tensor] = []
    valid_rows: List[int] = []
    weights: List[torch.Tensor] = []
    for row_idx, (idx, prob) in enumerate(zip(teacher_indices, teacher_probs)):
        if idx.numel() == 0 or prob.numel() == 0:
            continue
        if idx.numel() != prob.numel():
            raise ValueError("Each teacher index/prob tensor pair must have same length")
        row_weight_t: Optional[torch.Tensor] = None
        if row_weights is not None:
            row_weight_t = torch.as_tensor(
                row_weights[row_idx],
                device=log_probs.device,
                dtype=log_probs.dtype,
            ).clamp_min(0.0)
            if float(row_weight_t.item()) <= 0.0:
                continue
        q = prob.clamp_min(1e-12)
        row_parts.append(torch.full((idx.numel(),), row_idx, device=log_probs.device, dtype=torch.long))
        col_parts.append(idx)
        prob_parts.append(q)
        valid_rows.append(int(row_idx))
        if row_weight_t is not None:
            weights.append(row_weight_t)

    if not row_parts:
        return torch.zeros([], device=log_probs.device, dtype=log_probs.dtype), 0

    row_idx = torch.cat(row_parts, dim=0)
    col_idx = torch.cat(col_parts, dim=0)
    probs = torch.cat(prob_parts, dim=0)
    gathered = log_probs[row_idx, col_idx]
    terms = probs * (torch.log(probs) - gathered)

    per_row = torch.zeros(log_probs.shape[0], device=log_probs.device, dtype=log_probs.dtype)
    per_row.index_add_(0, row_idx, terms.to(dtype=per_row.dtype))
    valid_row_t = torch.tensor(valid_rows, device=log_probs.device, dtype=torch.long)
    stacked = per_row.index_select(0, valid_row_t)
    if weights:
        w = torch.stack(weights).to(dtype=stacked.dtype)
        return (stacked * w).sum() / w.sum().clamp_min(1e-8), int(valid_row_t.numel())
    return stacked.mean(), int(valid_row_t.numel())


def total_loss(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    out = None
    for name, value in losses.items():
        if value is None:
            continue
        weighted = weights.get(name, 1.0) * value
        out = weighted if out is None else out + weighted
    if out is None:
        raise ValueError("No losses to aggregate")
    return out
