from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F

from model import MultiTaskModel


def _amp_cuda_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def _safe_binary_auc(scores: torch.Tensor, labels01: torch.Tensor) -> float | None:
    labels01 = labels01.to(torch.float32)
    n_pos = int(labels01.sum().item())
    n_all = int(labels01.numel())
    n_neg = n_all - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = _rankdata(scores.to(torch.float32))
    sum_rank_pos = ranks[labels01 > 0.5].sum()
    auc = (sum_rank_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc.item())


def _safe_average_precision(scores: torch.Tensor, labels01: torch.Tensor) -> float | None:
    labels01 = labels01.to(torch.float32)
    n_pos = float(labels01.sum().item())
    if n_pos <= 0:
        return None

    order = torch.argsort(scores, descending=True)
    y = labels01.index_select(0, order)
    tp_cum = torch.cumsum(y, dim=0)
    k = torch.arange(1, y.numel() + 1, device=y.device, dtype=torch.float32)
    precision_at_k = tp_cum / k
    ap = (precision_at_k * y).sum() / n_pos
    return float(ap.item())


def _safe_binary_auc_from_positive_mask(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
) -> float | None:
    positive_mask = positive_mask.to(torch.bool)
    n_pos = int(positive_mask.sum().item())
    n_all = int(positive_mask.numel())
    n_neg = n_all - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = _rankdata(scores.to(torch.float32))
    sum_rank_pos = ranks[positive_mask].sum()
    auc = (sum_rank_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc.item())


def _safe_average_precision_from_positive_mask(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
) -> float | None:
    positive_mask = positive_mask.to(torch.bool)
    n_pos = float(positive_mask.sum().item())
    if n_pos <= 0:
        return None

    order = torch.argsort(scores, descending=True)
    y = positive_mask.index_select(0, order).to(torch.float32)
    tp_cum = torch.cumsum(y, dim=0)
    k = torch.arange(1, y.numel() + 1, device=y.device, dtype=torch.float32)
    precision_at_k = tp_cum / k
    ap = (precision_at_k * y).sum() / n_pos
    return float(ap.item())


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return None
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum()).item()
    if den <= 1e-12:
        return None
    return float(((x * y).sum().item()) / den)


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.numel() == 0:
        return torch.empty_like(x, dtype=torch.float32)

    order = torch.argsort(x, dim=0)
    sorted_x = x.index_select(0, order)
    group_start_mask = torch.ones(sorted_x.shape[0], dtype=torch.bool, device=x.device)
    if sorted_x.shape[0] > 1:
        group_start_mask[1:] = sorted_x[1:] != sorted_x[:-1]
    group_starts = group_start_mask.nonzero(as_tuple=False).squeeze(-1)
    group_ends = torch.cat(
        [
            group_starts[1:],
            torch.tensor([sorted_x.shape[0]], device=x.device, dtype=torch.long),
        ]
    )
    group_sizes = group_ends - group_starts
    group_avg_ranks = (
        group_starts.to(torch.float32) + 1.0 + group_ends.to(torch.float32)
    ) / 2.0
    ranks_sorted = torch.repeat_interleave(group_avg_ranks, group_sizes)
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _best_positive_tie_counts(
    scores: torch.Tensor,
    pos_idx: torch.Tensor,
) -> tuple[int, int, int]:
    pos_scores = scores.index_select(0, pos_idx)
    best_pos_score = pos_scores.max()
    n_greater = int((scores > best_pos_score).sum().item())
    n_tied_total = int((scores == best_pos_score).sum().item())
    n_tied_pos = int((pos_scores == best_pos_score).sum().item())
    return n_greater, n_tied_total, n_tied_pos


@lru_cache(maxsize=4096)
def _expected_reciprocal_best_rank(
    n_greater: int,
    n_tied_total: int,
    n_tied_pos: int,
) -> float:
    max_min_rank_in_tie = int(n_tied_total) - int(n_tied_pos) + 1
    prob = float(n_tied_pos) / float(max(int(n_tied_total), 1))
    exp_rr = 0.0
    for min_rank_in_tie in range(1, max_min_rank_in_tie + 1):
        exp_rr += prob / float(int(n_greater) + min_rank_in_tie)
        if min_rank_in_tie < max_min_rank_in_tie:
            numer = int(n_tied_total) - min_rank_in_tie - int(n_tied_pos) + 1
            denom = int(n_tied_total) - min_rank_in_tie
            prob *= float(numer) / float(max(denom, 1))
    return exp_rr


@lru_cache(maxsize=16384)
def _prob_best_rank_at_k(
    n_greater: int,
    n_tied_total: int,
    n_tied_pos: int,
    k: int,
) -> float:
    max_min_rank_in_tie = int(n_tied_total) - int(n_tied_pos) + 1
    allowed_tie_prefix = int(k) - int(n_greater)
    if allowed_tie_prefix <= 0:
        return 0.0
    if allowed_tie_prefix >= max_min_rank_in_tie:
        return 1.0

    # P(best positive rank > k) means all best-score positives fall after the first
    # `allowed_tie_prefix` positions inside the tied block.
    prob_all_after_k = 1.0
    for offset in range(int(n_tied_pos)):
        prob_all_after_k *= float(int(n_tied_total) - allowed_tie_prefix - offset) / float(
            int(n_tied_total) - offset
        )
    return 1.0 - prob_all_after_k


def _tie_aware_best_positive_metrics(
    scores: torch.Tensor,
    pos_idx: torch.Tensor,
    recall_ks: Sequence[int],
) -> tuple[float, Dict[int, float]]:
    n_greater, n_tied_total, n_tied_pos = _best_positive_tie_counts(scores, pos_idx)
    rr = _expected_reciprocal_best_rank(n_greater, n_tied_total, n_tied_pos)
    recall_probs = {
        int(k): _prob_best_rank_at_k(n_greater, n_tied_total, n_tied_pos, int(k))
        for k in recall_ks
    }
    return rr, recall_probs


def _safe_spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return None
    rx = _rankdata(x.to(torch.float32))
    ry = _rankdata(y.to(torch.float32))
    return _safe_pearson(rx, ry)


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _empty_domain_metrics() -> Dict[str, float]:
    return {
        "top1": 0.0,
        "top5": 0.0,
        "macro_f1": 0.0,
        "balanced_acc": 0.0,
        "ovr_macro_auroc": 0.0,
        "ovr_macro_auprc": 0.0,
        "n_eval": 0.0,
    }


def _compute_domain_metrics_from_logits(
    logits_cat: torch.Tensor,
    labels_cat: torch.Tensor,
) -> Dict[str, float]:
    n = int(labels_cat.shape[0])
    if n <= 0:
        return _empty_domain_metrics()

    preds = logits_cat.argmax(dim=-1)
    top5 = logits_cat.topk(k=min(5, logits_cat.shape[1]), dim=-1).indices
    active_classes = torch.unique(labels_cat).tolist()

    recalls: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []
    aps: List[float] = []
    for c in active_classes:
        c = int(c)
        y_true = (labels_cat == c).to(torch.float32)
        support = float(y_true.sum().item())
        if support <= 0:
            continue

        y_pred = (preds == c).to(torch.float32)
        tp = float((y_true * y_pred).sum().item())
        fp = float(((1.0 - y_true) * y_pred).sum().item())
        fn = float((y_true * (1.0 - y_pred)).sum().item())
        recall = tp / max(tp + fn, 1e-8)
        precision = tp / max(tp + fp, 1e-8)
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        recalls.append(recall)
        f1s.append(f1)

        cls_scores = logits_cat[:, c]
        auc = _safe_binary_auc(cls_scores, y_true)
        ap = _safe_average_precision(cls_scores, y_true)
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)

    return {
        "top1": float((preds == labels_cat).to(torch.float32).mean().item()),
        "top5": float((top5 == labels_cat.unsqueeze(1)).any(dim=1).to(torch.float32).mean().item()),
        "macro_f1": _mean_or_zero(f1s),
        "balanced_acc": _mean_or_zero(recalls),
        "ovr_macro_auroc": _mean_or_zero(aucs),
        "ovr_macro_auprc": _mean_or_zero(aps),
        "n_eval": float(n),
    }


def _metric_bucket_token(label: str) -> str:
    return str(label).replace("-", "_").replace("+", "_plus")


def _ndcg_at_k_binary(scores: torch.Tensor, labels01: torch.Tensor, k: int = 10) -> float:
    """Compute nDCG@k for binary relevance labels (1=positive, 0=negative)."""
    if scores.numel() == 0 or labels01.numel() == 0 or scores.numel() != labels01.numel():
        return 0.0
    k_eff = int(max(1, min(int(k), int(scores.numel()))))

    labels01 = labels01.to(torch.float32)
    top_idx = torch.topk(scores, k=k_eff, dim=0, largest=True, sorted=True).indices
    rel_top = labels01.index_select(0, top_idx)

    gains = rel_top  # 2^rel - 1 for binary rel equals rel.
    positions = torch.arange(1, k_eff + 1, device=scores.device, dtype=torch.float32)
    discounts = torch.log2(positions + 1.0)
    dcg = float((gains / discounts).sum().item())

    n_pos = int(labels01.sum().item())
    if n_pos <= 0:
        return 0.0
    ideal_hits = min(n_pos, k_eff)
    ideal_gains = torch.ones(ideal_hits, device=scores.device, dtype=torch.float32)
    ideal_positions = torch.arange(1, ideal_hits + 1, device=scores.device, dtype=torch.float32)
    idcg = float((ideal_gains / torch.log2(ideal_positions + 1.0)).sum().item())
    if idcg <= 1e-12:
        return 0.0
    return float(dcg / idcg)


def _ndcg_at_k_from_positive_indices(
    scores: torch.Tensor,
    pos_idx: torch.Tensor,
    k: int = 10,
) -> float:
    if scores.numel() == 0 or pos_idx.numel() == 0:
        return 0.0
    k_eff = int(max(1, min(int(k), int(scores.numel()))))
    top_idx = torch.topk(scores, k=k_eff, dim=0, largest=True, sorted=True).indices
    hits = top_idx.unsqueeze(1).eq(pos_idx.unsqueeze(0)).any(dim=1).to(torch.float32)
    positions = torch.arange(1, k_eff + 1, device=scores.device, dtype=torch.float32)
    dcg = float((hits / torch.log2(positions + 1.0)).sum().item())

    ideal_hits = min(int(pos_idx.numel()), k_eff)
    if ideal_hits <= 0:
        return 0.0
    ideal_positions = torch.arange(1, ideal_hits + 1, device=scores.device, dtype=torch.float32)
    idcg = float((torch.ones(ideal_hits, device=scores.device) / torch.log2(ideal_positions + 1.0)).sum().item())
    if idcg <= 1e-12:
        return 0.0
    return float(dcg / idcg)


def evaluate_main(
    model: MultiTaskModel,
    loader,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    gene_graph_emb: torch.Tensor,
    trait_graph_emb: torch.Tensor,
    disease_ids: Sequence[int],
    disease_to_traits: Dict[int, List[int]],
    device: torch.device,
    disease_freq_buckets: Optional[Dict[str, Set[int]]] = None,
    gene_size_buckets: Optional[Dict[str, Set[int]]] = None,
    collect_gate_stats: bool = False,
    restrict_to_disease_ids: Optional[Sequence[int]] = None,
    gate_temperature: float = 1.0,
    compute_heavy_metrics: bool = True,
    minimal: bool = False,
) -> Dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda"
    amp_dtype = _amp_cuda_dtype() if use_amp else torch.bfloat16
    disease_ids = list(disease_ids)
    if restrict_to_disease_ids is not None:
        restrict_set = {int(d) for d in restrict_to_disease_ids}
        eval_disease_ids = [d for d in disease_ids if d in restrict_set]
    else:
        eval_disease_ids = disease_ids
    disease_id_to_col = {d: i for i, d in enumerate(eval_disease_ids)}

    if len(eval_disease_ids) == 0:
        out = {
            "mrr": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "ndcg@10": 0.0,
            "gene_macro_ndcg@10": 0.0,
            "gene_macro_n_eval_genes": 0.0,
            "auroc_query_mean": 0.0,
            "auprc_query_mean": 0.0,
            "map": 0.0,
            "n_eval": 0.0,
        }
        if disease_freq_buckets:
            for b in disease_freq_buckets.keys():
                out[f"bucket_{b}_mrr"] = 0.0
                out[f"bucket_{b}_recall@10"] = 0.0
                out[f"bucket_{b}_ndcg@10"] = 0.0
                out[f"bucket_{b}_n_eval"] = 0.0
        if gene_size_buckets:
            for b in gene_size_buckets.keys():
                token = _metric_bucket_token(b)
                out[f"size_bucket_{token}_gene_macro_ndcg@10"] = 0.0
                out[f"size_bucket_{token}_n_eval_genes"] = 0.0
        if collect_gate_stats:
            out["gate_variant_mean"] = 0.0
            out["gate_protein_mean"] = 0.0
            out["gate_gene_mean"] = 0.0
        if restrict_to_disease_ids is not None:
            out["restricted_disease_bank_size"] = 0.0
        return out

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            disease_emb = model.encode_disease_batch(
                eval_disease_ids,
                disease_to_traits,
                trait_graph_emb,
            )

    recall_ks = (1, 5, 10)
    rr_sum = 0.0
    recall_counts = {k: 0.0 for k in recall_ks}
    ndcg_sum = 0.0
    gene_query_ndcg_sum: Dict[int, float] = {}
    gene_query_ndcg_count: Dict[int, int] = {}
    size_bucket_gene_ndcg_sum: Dict[str, float] = {}
    size_bucket_gene_count: Dict[str, int] = {}
    gene_size_bucket_by_idx: Dict[int, str] = {}
    query_auroc_sum = 0.0
    query_auroc_n = 0
    query_auprc_sum = 0.0
    query_auprc_n = 0
    gate_sum = torch.zeros(3, dtype=torch.float64)
    gate_n = 0
    bucket_rr_sum: Dict[str, float] = {}
    bucket_ndcg_sum: Dict[str, float] = {}
    bucket_hits: Dict[str, Dict[int, float]] = {}
    bucket_n: Dict[str, int] = {}
    disease_bucket_by_col: List[Optional[str]] = [None] * len(eval_disease_ids)
    if disease_freq_buckets:
        for b in disease_freq_buckets.keys():
            bucket_rr_sum[b] = 0.0
            bucket_ndcg_sum[b] = 0.0
            bucket_hits[b] = {k: 0.0 for k in recall_ks}
            bucket_n[b] = 0
            for disease_id in disease_freq_buckets[b]:
                col = disease_id_to_col.get(int(disease_id))
                if col is not None:
                    disease_bucket_by_col[int(col)] = str(b)
    if gene_size_buckets:
        for b, gene_idx_set in gene_size_buckets.items():
            for gene_idx_val in gene_idx_set:
                gene_size_bucket_by_idx[int(gene_idx_val)] = str(b)
    n = 0

    with torch.no_grad():
        for batch in loader:
            gene_idx_cpu = batch["gene_idx"]
            variant_idx = batch["variant_idx"].to(device, non_blocking=True)
            gene_idx = gene_idx_cpu.to(device, non_blocking=True)
            positive_cols_batch = batch.get("positive_disease_cols")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if collect_gate_stats:
                    zv_out = model.encode_variant(
                        variant_idx,
                        gene_idx,
                        variant_x,
                        protein_x,
                        gene_graph_emb,
                        gate_temperature=gate_temperature,
                        return_gate_weights=True,
                    )
                    z_v, gate_w = zv_out
                else:
                    z_v = model.encode_variant(
                        variant_idx,
                        gene_idx,
                        variant_x,
                        protein_x,
                        gene_graph_emb,
                        gate_temperature=gate_temperature,
                    )
                z_v = F.normalize(model.clip_variant_proj(z_v), dim=-1)
                scores = z_v @ disease_emb.t()  # [B, D]
            scores = scores.float()
            if collect_gate_stats:
                gate_sum += gate_w.detach().cpu().to(torch.float64).sum(dim=0)
                gate_n += gate_w.shape[0]

            for i, positives in enumerate(batch["positive_disease_ids"]):
                if restrict_to_disease_ids is None and positive_cols_batch is not None:
                    cols = [
                        int(col)
                        for col in positive_cols_batch[i]
                        if 0 <= int(col) < len(eval_disease_ids)
                    ]
                else:
                    cols = [disease_id_to_col[d] for d in positives if d in disease_id_to_col]
                if not cols:
                    continue
                row_scores = scores[i]
                pos_idx = torch.as_tensor(cols, dtype=torch.long, device=row_scores.device)

                ndcg = _ndcg_at_k_from_positive_indices(row_scores, pos_idx, k=10)
                ndcg_sum += ndcg
                gene_key = int(gene_idx_cpu[i].item())
                gene_query_ndcg_sum[gene_key] = gene_query_ndcg_sum.get(gene_key, 0.0) + ndcg
                gene_query_ndcg_count[gene_key] = gene_query_ndcg_count.get(gene_key, 0) + 1
                rr_contrib, recall_probs = _tie_aware_best_positive_metrics(
                    row_scores, pos_idx, recall_ks
                )
                rr_sum += rr_contrib
                for k, hit_prob in recall_probs.items():
                    recall_counts[k] += hit_prob

                if not minimal:
                    if disease_freq_buckets:
                        positive_mask = torch.zeros(
                            len(eval_disease_ids),
                            dtype=torch.bool,
                            device=row_scores.device,
                        )
                        positive_mask.index_fill_(0, pos_idx, True)
                        bucket_cols_map: Dict[str, List[int]] = {}
                        for col in cols:
                            bucket_name = disease_bucket_by_col[int(col)]
                            if bucket_name is None:
                                continue
                            bucket_cols_map.setdefault(bucket_name, []).append(int(col))
                        for bucket_name, bucket_cols in bucket_cols_map.items():
                            if not bucket_cols:
                                continue
                            bucket_pos_idx = torch.as_tensor(
                                bucket_cols, dtype=torch.long, device=row_scores.device
                            )
                            bucket_rr_contrib, bucket_recall_probs = _tie_aware_best_positive_metrics(
                                row_scores,
                                bucket_pos_idx,
                                recall_ks,
                            )
                            bucket_rr_sum[bucket_name] += bucket_rr_contrib
                            for k, hit_prob in bucket_recall_probs.items():
                                bucket_hits[bucket_name][k] += hit_prob
                            bucket_ndcg_sum[bucket_name] += _ndcg_at_k_from_positive_indices(
                                row_scores,
                                bucket_pos_idx,
                                k=10,
                            )
                            bucket_n[bucket_name] += 1
                    if compute_heavy_metrics:
                        positive_mask_hm = torch.zeros(
                            len(eval_disease_ids),
                            dtype=torch.bool,
                            device=row_scores.device,
                        )
                        positive_mask_hm.index_fill_(0, pos_idx, True)
                        auc = _safe_binary_auc_from_positive_mask(row_scores, positive_mask_hm)
                        ap = _safe_average_precision_from_positive_mask(row_scores, positive_mask_hm)
                        if auc is not None:
                            query_auroc_sum += auc
                            query_auroc_n += 1
                        if ap is not None:
                            query_auprc_sum += ap
                            query_auprc_n += 1
                n += 1

    if n == 0:
        out = {
            "mrr": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "ndcg@10": 0.0,
            "gene_macro_ndcg@10": 0.0,
            "gene_macro_n_eval_genes": 0.0,
            "auroc_query_mean": 0.0,
            "auprc_query_mean": 0.0,
            "map": 0.0,
            "n_eval": 0.0,
        }
        if disease_freq_buckets:
            for b in disease_freq_buckets.keys():
                out[f"bucket_{b}_mrr"] = 0.0
                out[f"bucket_{b}_recall@10"] = 0.0
                out[f"bucket_{b}_ndcg@10"] = 0.0
                out[f"bucket_{b}_n_eval"] = 0.0
        if gene_size_buckets:
            for b in gene_size_buckets.keys():
                token = _metric_bucket_token(b)
                out[f"size_bucket_{token}_gene_macro_ndcg@10"] = 0.0
                out[f"size_bucket_{token}_n_eval_genes"] = 0.0
        if collect_gate_stats:
            out["gate_variant_mean"] = 0.0
            out["gate_protein_mean"] = 0.0
            out["gate_gene_mean"] = 0.0
        if restrict_to_disease_ids is not None:
            out["restricted_disease_bank_size"] = float(len(eval_disease_ids))
        return out

    min_gene_queries = 3
    gene_macro_ndcg_sum = 0.0
    gene_macro_ndcg_n = 0
    for gene_key, gene_ndcg_total in gene_query_ndcg_sum.items():
        gene_count = gene_query_ndcg_count.get(gene_key, 0)
        if gene_count < min_gene_queries:
            continue
        gene_mean_ndcg = float(gene_ndcg_total / gene_count)
        gene_macro_ndcg_sum += gene_mean_ndcg
        gene_macro_ndcg_n += 1
        bucket_name = gene_size_bucket_by_idx.get(int(gene_key))
        if bucket_name is not None:
            size_bucket_gene_ndcg_sum[bucket_name] = (
                size_bucket_gene_ndcg_sum.get(bucket_name, 0.0) + gene_mean_ndcg
            )
            size_bucket_gene_count[bucket_name] = size_bucket_gene_count.get(bucket_name, 0) + 1
    out = {
        "mrr": float(rr_sum / n),
        "recall@1": float(recall_counts[1] / n),
        "recall@5": float(recall_counts[5] / n),
        "recall@10": float(recall_counts[10] / n),
        "ndcg@10": float(ndcg_sum / n),
        "gene_macro_ndcg@10": float(gene_macro_ndcg_sum / max(gene_macro_ndcg_n, 1)),
        "gene_macro_n_eval_genes": float(gene_macro_ndcg_n),
        "auroc_query_mean": float(query_auroc_sum / max(query_auroc_n, 1)),
        "auprc_query_mean": float(query_auprc_sum / max(query_auprc_n, 1)),
        "map": float(query_auprc_sum / max(query_auprc_n, 1)),
        "n_eval": float(n),
    }
    if disease_freq_buckets:
        for b in disease_freq_buckets.keys():
            nb = bucket_n.get(b, 0)
            out[f"bucket_{b}_mrr"] = float(bucket_rr_sum[b] / nb) if nb > 0 else 0.0
            out[f"bucket_{b}_recall@10"] = (
                float(bucket_hits[b][10] / nb) if nb > 0 else 0.0
            )
            out[f"bucket_{b}_ndcg@10"] = (
                float(bucket_ndcg_sum[b] / nb) if nb > 0 else 0.0
            )
            out[f"bucket_{b}_n_eval"] = float(nb)
    if gene_size_buckets:
        for b in gene_size_buckets.keys():
            token = _metric_bucket_token(b)
            bucket_gene_count = size_bucket_gene_count.get(b, 0)
            out[f"size_bucket_{token}_gene_macro_ndcg@10"] = (
                float(size_bucket_gene_ndcg_sum.get(b, 0.0) / bucket_gene_count)
                if bucket_gene_count > 0
                else 0.0
            )
            out[f"size_bucket_{token}_n_eval_genes"] = float(bucket_gene_count)
    if collect_gate_stats:
        gm = gate_sum / max(gate_n, 1)
        out["gate_variant_mean"] = float(gm[0].item())
        out["gate_protein_mean"] = float(gm[1].item())
        out["gate_gene_mean"] = float(gm[2].item())
        if hasattr(model.fusion, "residual_alpha"):
            out["residual_alpha"] = float(
                torch.sigmoid(model.fusion.residual_alpha).detach().cpu().item()
            )
    if restrict_to_disease_ids is not None:
        out["restricted_disease_bank_size"] = float(len(eval_disease_ids))
    return out


def evaluate_domain(
    model: MultiTaskModel,
    loader,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    gene_graph_emb: torch.Tensor,
    domain_embeddings: torch.Tensor,
    device: torch.device,
    temperature: float = 0.15,
    gate_temperature: float = 1.0,
    seen_labels: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda"
    amp_dtype = _amp_cuda_dtype() if use_amp else torch.bfloat16
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            domain_proto = F.normalize(model.domain_transform(domain_embeddings), dim=-1)
        for batch in loader:
            variant_idx = batch["variant_idx"].to(device, non_blocking=True)
            gene_idx = batch["gene_idx"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                z_v = model.encode_variant(
                    variant_idx,
                    gene_idx,
                    variant_x,
                    protein_x,
                    gene_graph_emb,
                    gate_temperature=gate_temperature,
                )
                z_v = F.normalize(model.domain_variant_proj(z_v), dim=-1)
                logits = z_v @ domain_proto.t() / max(temperature, 1e-6)
            logits = logits.float()

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_labels:
        out = _empty_domain_metrics()
    else:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        out = _compute_domain_metrics_from_logits(logits_cat, labels_cat)
        if seen_labels is not None:
            seen_label_set = {int(v) for v in seen_labels}
            mask = torch.tensor(
                [int(v) in seen_label_set for v in labels_cat.tolist()],
                dtype=torch.bool,
            )
            seen_metrics = _compute_domain_metrics_from_logits(logits_cat[mask], labels_cat[mask])
            out.update({f"seen_label_{k}": v for k, v in seen_metrics.items()})
            out["unseen_label_n_eval"] = float((~mask).sum().item())
            out["seen_label_eval_fraction"] = (
                float(seen_metrics["n_eval"] / out["n_eval"]) if out["n_eval"] > 0 else 0.0
            )
    if seen_labels is not None and "seen_label_top1" not in out:
        seen_metrics = _empty_domain_metrics()
        out.update({f"seen_label_{k}": v for k, v in seen_metrics.items()})
        out["unseen_label_n_eval"] = float(out["n_eval"])
        out["seen_label_eval_fraction"] = 0.0
    return out


def evaluate_func(
    model: MultiTaskModel,
    loader,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    gene_graph_emb: torch.Tensor,
    device: torch.device,
    gate_temperature: float = 1.0,
) -> Dict[str, float]:
    """多轴 func 评估：每个回归轴独立 MAE/Pearson，机制轴 macro-F1。"""
    from data import FUNC_AXIS_SLICES
    model.eval()
    use_amp = device.type == "cuda"
    amp_dtype = _amp_cuda_dtype() if use_amp else torch.bfloat16

    # 每个回归轴的收集器
    axis_collectors: Dict[str, Dict[str, List[torch.Tensor]]] = {
        name: {"pred": [], "target": []}
        for name in FUNC_AXIS_SLICES
    }
    mech_pred_parts: List[torch.Tensor] = []
    mech_tgt_parts: List[torch.Tensor] = []
    n = 0

    with torch.no_grad():
        for batch in loader:
            variant_idx = batch["variant_idx"].to(device, non_blocking=True)
            gene_idx = batch["gene_idx"].to(device, non_blocking=True)
            reg_target = batch["regression_target"].to(device, non_blocking=True)
            reg_mask = batch["regression_mask"].to(device, non_blocking=True)
            mech_target = batch["mechanism_target"]
            mech_mask = batch["mechanism_mask"]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = model.forward_func(
                    variant_idx, gene_idx, variant_x, protein_x,
                    gene_graph_emb, gate_temperature=gate_temperature,
                )

            n += int(variant_idx.shape[0])

            # 回归轴：按 mask 收集有效值
            for axis_name, (start, end) in FUNC_AXIS_SLICES.items():
                pred_axis = preds[axis_name].float()
                tgt_axis = reg_target[:, start:end]
                msk_axis = reg_mask[:, start:end]
                observed = msk_axis > 0
                if observed.any():
                    axis_collectors[axis_name]["pred"].append(pred_axis[observed].detach().cpu())
                    axis_collectors[axis_name]["target"].append(tgt_axis[observed].detach().cpu())

            # 机制轴
            if "mechanism" in preds:
                valid = mech_mask > 0.5
                if valid.any():
                    mech_pred_parts.append(preds["mechanism"][valid].float().detach().cpu())
                    mech_tgt_parts.append(mech_target[valid])

    if n == 0:
        return {"n_eval": 0.0}

    metrics: Dict[str, float] = {"n_eval": float(n)}

    # 回归轴指标
    for axis_name in FUNC_AXIS_SLICES:
        coll = axis_collectors[axis_name]
        if coll["pred"]:
            obs_pred = torch.cat(coll["pred"], dim=0)
            obs_tgt = torch.cat(coll["target"], dim=0)
            diff = obs_pred - obs_tgt
            metrics[f"{axis_name}_mae"] = float(torch.abs(diff).mean().item())
            metrics[f"{axis_name}_rmse"] = float((diff ** 2).mean().sqrt().item())
            pearson = _safe_pearson(obs_pred, obs_tgt)
            metrics[f"{axis_name}_pearson"] = float(pearson if pearson is not None else 0.0)
        else:
            metrics[f"{axis_name}_mae"] = 0.0
            metrics[f"{axis_name}_rmse"] = 0.0
            metrics[f"{axis_name}_pearson"] = 0.0

    # 机制轴 macro-F1
    if mech_pred_parts:
        mech_pred_all = torch.cat(mech_pred_parts, dim=0)
        mech_tgt_all = torch.cat(mech_tgt_parts, dim=0)
        mech_pred_binary = (torch.sigmoid(mech_pred_all) > 0.5).float()
        # per-label F1
        f1_scores = []
        for j in range(mech_tgt_all.shape[1]):
            tp = ((mech_pred_binary[:, j] == 1) & (mech_tgt_all[:, j] == 1)).sum().float()
            fp = ((mech_pred_binary[:, j] == 1) & (mech_tgt_all[:, j] == 0)).sum().float()
            fn = ((mech_pred_binary[:, j] == 0) & (mech_tgt_all[:, j] == 1)).sum().float()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            f1_scores.append(float(f1.item()))
        metrics["mechanism_macro_f1"] = float(sum(f1_scores) / max(len(f1_scores), 1))
        metrics["mechanism_n_eval"] = float(mech_tgt_all.shape[0])
    else:
        metrics["mechanism_macro_f1"] = 0.0
        metrics["mechanism_n_eval"] = 0.0

    return metrics


def export_per_example_predictions(
    model: MultiTaskModel,
    loader,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    gene_graph_emb: torch.Tensor,
    trait_graph_emb: torch.Tensor,
    disease_ids: Sequence[int],
    disease_to_traits: Dict[int, List[int]],
    device: torch.device,
    output_path: str | Path,
    disease_freq_buckets: Optional[Dict[str, Set[int]]] = None,
    gate_temperature: float = 1.0,
    top_k_diseases: int = 20,
) -> int:
    """Export per-example predictions to CSV for downstream analysis.

    Each row = one (variant, gene) query. Columns:
      variant_idx, gene_idx, positive_disease_ids, n_positives,
      mrr, ndcg_at_10, best_rank, recall_at_1, recall_at_5, recall_at_10,
      bucket (if disease_freq_buckets provided),
      top_k_disease_ids, top_k_scores,
      positive_scores (score for each positive disease)

    Returns the number of rows written.
    """
    model.eval()
    use_amp = device.type == "cuda"
    amp_dtype = _amp_cuda_dtype() if use_amp else torch.bfloat16
    disease_ids = list(disease_ids)
    eval_disease_ids = disease_ids
    disease_id_to_col = {d: i for i, d in enumerate(eval_disease_ids)}

    disease_bucket_by_col: List[Optional[str]] = [None] * len(eval_disease_ids)
    if disease_freq_buckets:
        for b, disease_set in disease_freq_buckets.items():
            for disease_id in disease_set:
                col = disease_id_to_col.get(int(disease_id))
                if col is not None:
                    disease_bucket_by_col[int(col)] = str(b)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            disease_emb = model.encode_disease_batch(
                eval_disease_ids, disease_to_traits, trait_graph_emb,
            )

    recall_ks = (1, 5, 10)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant_idx", "gene_idx",
            "positive_disease_ids", "n_positives",
            "mrr", "ndcg_at_10", "best_rank",
            "recall_at_1", "recall_at_5", "recall_at_10",
            "bucket",
            "top_k_disease_ids", "top_k_scores",
            "positive_scores",
        ])

        with torch.no_grad():
            for batch in loader:
                gene_idx_cpu = batch["gene_idx"]
                variant_idx_cpu = batch["variant_idx"]
                variant_idx = variant_idx_cpu.to(device, non_blocking=True)
                gene_idx = gene_idx_cpu.to(device, non_blocking=True)
                positive_cols_batch = batch.get("positive_disease_cols")

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    z_v = model.encode_variant(
                        variant_idx, gene_idx,
                        variant_x, protein_x, gene_graph_emb,
                        gate_temperature=gate_temperature,
                    )
                    z_v = F.normalize(model.clip_variant_proj(z_v), dim=-1)
                    scores = z_v @ disease_emb.t()
                scores = scores.float()

                for i, positives in enumerate(batch["positive_disease_ids"]):
                    if positive_cols_batch is not None:
                        cols = [
                            int(col)
                            for col in positive_cols_batch[i]
                            if 0 <= int(col) < len(eval_disease_ids)
                        ]
                    else:
                        cols = [disease_id_to_col[d] for d in positives if d in disease_id_to_col]
                    if not cols:
                        continue

                    row_scores = scores[i]
                    pos_idx = torch.as_tensor(cols, dtype=torch.long, device=row_scores.device)

                    rr, recall_probs = _tie_aware_best_positive_metrics(
                        row_scores, pos_idx, recall_ks,
                    )
                    ndcg = _ndcg_at_k_from_positive_indices(row_scores, pos_idx, k=10)

                    # best rank (1-indexed, tie-aware expected)
                    n_greater, n_tied_total, n_tied_pos = _best_positive_tie_counts(row_scores, pos_idx)
                    best_rank = n_greater + (n_tied_total - n_tied_pos + 1 + n_tied_total) / 2.0

                    # bucket for this query (use first positive's bucket)
                    bucket = ""
                    if disease_freq_buckets:
                        for col in cols:
                            b = disease_bucket_by_col[int(col)]
                            if b is not None:
                                bucket = b
                                break

                    # top-k predictions
                    k_eff = min(top_k_diseases, len(eval_disease_ids))
                    topk = torch.topk(row_scores, k=k_eff, dim=0, largest=True, sorted=True)
                    top_disease_ids = [eval_disease_ids[int(idx)] for idx in topk.indices.tolist()]
                    top_scores = [f"{s:.6f}" for s in topk.values.tolist()]

                    # scores for positive diseases
                    pos_scores = [f"{row_scores[c].item():.6f}" for c in cols]
                    pos_disease_ids = [eval_disease_ids[c] for c in cols]

                    writer.writerow([
                        int(variant_idx_cpu[i].item()),
                        int(gene_idx_cpu[i].item()),
                        ";".join(str(d) for d in pos_disease_ids),
                        len(cols),
                        f"{rr:.6f}",
                        f"{ndcg:.6f}",
                        f"{best_rank:.1f}",
                        f"{recall_probs[1]:.6f}",
                        f"{recall_probs[5]:.6f}",
                        f"{recall_probs[10]:.6f}",
                        bucket,
                        ";".join(str(d) for d in top_disease_ids),
                        ";".join(top_scores),
                        ";".join(pos_scores),
                    ])
                    n_written += 1

    return n_written
