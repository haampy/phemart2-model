from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from config import default_config, ensure_output_dir
from data import (
    FUNC_REGRESSION_COLS,
    apply_disease_split,
    apply_split,
    build_disease_frequency_buckets,
    build_disease_holdout_split,
    build_gene_size_buckets,
    build_main_split_coverage_sets,
    build_query_positive_disease_lookup,
    build_disease_to_traits_map,
    build_feature_store,
    build_global_variant_split,
    build_hetero_graph,
    build_inductive_train_graph,
    build_variant_disease_kl_teacher,
    build_mappings,
    build_protocol_id,
    build_rsid_to_hgvs_map,
    build_split_artifact,
    build_within_gene_variant_split,
    compute_disease_inv_freq_weights,
    compute_func_target_scales,
    load_disease_table,
    load_domain_labels,
    load_embeddings,
    load_func_labels,
    load_gene_concept_targets,
    load_main_labels,
    load_split_artifact,
    make_dataloader_for_task,
    make_domain_records,
    make_func_records,
    make_main_records,
    make_vd_d2v_records,
    normalize_id,
    parse_hpo_ids,
    prepare_task_dataframe_for_training,
    remap_variant_ids_to_hgvs,
    save_split_artifact,
    select_domain_train_subset,
    select_func_train_subset,
    summarize_disease_holdout_split,
    summarize_gene_holdout_split,
    summarize_split,
    validate_split_consistency,
)
from eval import evaluate_main, export_per_example_predictions
from model import MultiTaskModel
from train import evaluate_all_tasks, train_multitask


TASK_MODE_TO_ENABLED = {
    "main_only": {"main"},
    "main_func": {"main", "func"},
    "main_domain": {"main", "domain"},
    "main_domain_func": {"main", "domain", "func"},
}

TASK_MODE_ALIASES = {
    "full": "main_domain_func",
}

TASK_MODE_CHOICES = sorted(set(TASK_MODE_TO_ENABLED) | set(TASK_MODE_ALIASES))


MAIN_DELTA_METRIC_KEYS = (
    "mrr",
    "recall@1",
    "recall@5",
    "recall@10",
    "ndcg@10",
    "gene_macro_ndcg@10",
    "auroc_query_mean",
    "auprc_query_mean",
    "map",
)


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _build_main_metric_delta(
    full_main: Dict[str, float],
    baseline_main: Dict[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in MAIN_DELTA_METRIC_KEYS:
        out[f"delta_{key}"] = float(full_main.get(key, 0.0) - baseline_main.get(key, 0.0))
    return out


def _build_train_selection_summary(
    train_result: Dict[str, object],
    test_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, float | int | str]:
    best_val_metrics = train_result.get("best_val_metrics", {})
    best_main = best_val_metrics.get("main", {}) if isinstance(best_val_metrics, dict) else {}
    test_main = test_metrics.get("main", {})
    summary: Dict[str, float | int | str] = {
        "best_epoch": int(train_result.get("best_epoch", 0) or 0),
        "completed_epochs": int(train_result.get("completed_epochs", 0) or 0),
        "best_metric_name": str(train_result.get("best_metric_name", "")),
        "best_metric": float(train_result.get("best_metric", 0.0) or 0.0),
    }
    for key in ("mrr", "recall@10", "ndcg@10", "gene_macro_ndcg@10"):
        best_val = float(best_main.get(key, 0.0))
        test_val = float(test_main.get(key, 0.0))
        summary[f"best_val_main_{key}"] = best_val
        summary[f"test_main_{key}"] = test_val
        summary[f"generalization_gap_{key}"] = float(best_val - test_val)
    return summary


# --- arg → config 映射表 ---
# (arg_name, config_section, config_field)
# 当 arg_name == config_field 时可以省略第三项（用 None）
_ARG_CFG_DIRECT: List[Tuple[str, str, Optional[str]]] = [
    # runtime
    ("num_workers", "runtime", None),
    ("device", "runtime", None),
    # split
    ("seed", "split", None),
    ("split_protocol", "split", "protocol"),
    ("split_mode", "split", "mode"),
    # paths
    ("output_dir", "paths", None),
    ("main_labels", "paths", None),
    ("disease_table", "paths", None),
    ("split_artifact_path", "paths", None),
    ("mvp_hpo_semantic_map", "paths", None),
    ("mvp_disease_concept_map", "paths", None),
    ("mvp_topk_indices", "paths", None),
    ("mvp_topk_values", "paths", None),
    ("mvp_topk_metadata", "paths", None),
    # loss_weights
    ("loss_weight_main", "loss_weights", "main"),
    ("loss_weight_domain", "loss_weights", "domain"),
    ("loss_weight_func", "loss_weights", "func"),
    ("loss_weight_concept", "loss_weights", "concept"),
    # model
    ("graph_mode", "model", None),
    ("residual_alpha_max", "model", None),
    ("modality_drop_variant", "model", None),
    ("modality_drop_protein", "model", None),
    ("modality_drop_gene", "model", None),
    ("trait_dropout", "model", None),
    ("fusion_type", "model", None),
    ("num_graph_layers", "model", None),
    ("hidden_dim", "model", None),
    ("out_dim", "model", None),
    # train — general
    ("epochs", "train", None),
    ("batch_size_main", "train", None),
    ("batch_size_domain", "train", None),
    ("batch_size_func", "train", None),
    ("lr", "train", None),
    ("lr_graph", "train", None),
    ("lr_disease_encoder", "train", None),
    ("lr_graph_warmup", "train", None),
    ("graph_warmup_epochs", "train", None),
    ("graph_cache_refresh_steps", "train", None),
    ("weight_decay", "train", None),
    ("grad_clip_norm", "train", None),
    ("eval_interval", "train", None),
    ("early_stopping_patience", "train", None),
    ("main_temperature", "train", None),
    ("domain_temperature", "train", None),
    ("domain_loss_type", "train", None),
    ("domain_contrastive_negatives", "train", None),
    ("domain_data_mode", "train", None),
    ("domain_train_per_label_cap", "train", None),
    ("main_early_stop_metric", "train", None),
    ("main_loss_type", "train", None),
    ("main_logit_scale_min", "train", None),
    ("main_logit_scale_max", "train", None),
    ("main_logit_scale_init", "train", None),
    ("main_logit_scale_lr_mult", "train", None),
    ("label_smoothing", "train", None),
    ("aux_domain_interval", "train", None),
    ("aux_func_interval", "train", None),
    ("main_only_warmup_epochs", "train", None),
    ("gate_entropy_weight_start", "train", None),
    ("gate_entropy_weight_end", "train", None),
    ("graph_visibility", "train", None),
    ("graph_train_mode", "train", None),
    ("func_min_valid_axes", "train", None),
    ("func_mechanism_pos_weight", "train", None),
    ("func_train_per_gene_cap", "train", None),
    ("min_train_records_func", "train", None),
    ("disease_freq_reweight", "train", None),
    ("disease_freq_weight_agg", "train", None),
    ("disease_freq_weight_clip", "train", None),
    ("main_hard_negative_k", "train", None),
    ("scheduler_t0", "train", None),
    ("scheduler_t_mult", "train", None),
    ("scheduler_eta_min", "train", None),
    # train — VD-KL
    ("vd_kl_loss_type", "train", None),
    ("vd_kl_teacher_mode", "train", None),
    ("vd_kl_lambda_v2d", "train", None),
    ("vd_kl_lambda_d2v", "train", None),
    ("vd_kl_lambda_v2d_start", "train", None),
    ("vd_kl_lambda_d2v_start", "train", None),
    ("vd_kl_lambda_ramp_epochs", "train", None),
    ("vd_kl_temperature", "train", None),
    ("vd_kl_warmup_epochs", "train", None),
    ("vd_kl_d2v_start_epoch", "train", None),
    ("vd_kl_cache_refresh_interval", "train", None),
    ("vd_kl_d2v_batch_size", "train", None),
    ("vd_kl_hpo_topk_per_hpo", "train", None),
    ("vd_kl_hpo_min_similarity", "train", None),
    ("vd_kl_concept_direct_weight", "train", None),
    ("vd_kl_concept_semantic_weight", "train", None),
    ("vd_kl_min_variant_mapped_mass", "train", None),
    ("vd_kl_max_diseases_per_variant", "train", None),
    ("vd_kl_disease_score_concentration", "train", None),
    ("vd_kl_max_diseases_per_concept", "train", None),
    ("vd_kl_max_variants_per_disease", "train", None),
    ("vd_kl_positive_smoothing", "train", None),
    ("vd_kl_d2v_positive_smoothing", "train", None),
    ("vd_kl_d2v_min_anchor_mass", "train", None),
    ("vd_kl_d2v_min_rows_per_step", "train", None),
    ("vd_kl_min_teacher_top1_prob", "train", None),
    ("vd_kl_d2v_teacher_topk", "train", None),
    ("vd_kl_d2v_random_negatives", "train", None),
    ("vd_kl_d2v_max_positive_variants", "train", None),
    ("vd_kl_adaptive_reference_scale", "train", None),
    ("vd_kl_gene_propagation_alpha", "train", None),
    ("vd_kl_gene_propagation_distance_lambda", "train", None),
    ("vd_kl_gene_propagation_max_distance", "train", None),
    ("vd_kl_gene_propagation_entropy_threshold", "train", None),
    ("vd_kl_gene_propagation_position_unknown_penalty", "train", None),
    ("vd_kl_bridge_length_penalty_b", "train", None),
    ("vd_kl_hpo_min_terms_for_full_weight", "train", None),
    ("vd_kl_teacher_max_concepts", "train", None),
    ("vd_kl_teacher_concept_temperature", "train", None),
    ("vd_kl_prior_correction_alpha", "train", None),
    ("vd_kl_concept_filter_level", "train", None),
    ("vd_kl_min_concept_corr", "train", None),
    ("vd_kl_disease_score_power", "train", None),
    ("vd_kl_slack_tau", "train", None),
]

# 需要 bool() 转换的参数 (argparse type=int choices=[0,1] → config bool)
_ARG_CFG_BOOL: List[Tuple[str, str, Optional[str]]] = [
    ("main_logit_scale_learnable", "train", None),
    ("aux_update_hgt", "train", None),
    ("enrich_trait_graph", "model", None),
    ("disease_size_embed", "model", None),
    ("enable_vd_kl", "train", None),
    ("vd_kl_adaptive_weight", "train", None),
    ("vd_kl_gene_propagation", "train", None),
    ("vd_kl_gene_propagation_adaptive_alpha", "train", None),
    ("vd_kl_gene_propagation_exclude_d2v", "train", None),
    ("vd_kl_gene_propagation_confidence_scaling", "train", None),
    ("vd_kl_quality_row_weight", "train", None),
    ("vd_kl_shuffle_teacher", "train", None),
    ("vd_kl_raw_corr_mode", "train", None),
    ("vd_kl_concept_quality_scaling", "train", None),
    ("vd_kl_concept_specificity_weighting", "train", None),
    ("enable_concept_regression", "train", None),
]


def _apply_args_to_config(args: argparse.Namespace, cfg) -> None:
    for arg_name, section, field in _ARG_CFG_DIRECT:
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(getattr(cfg, section), field or arg_name, val)
    for arg_name, section, field in _ARG_CFG_BOOL:
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(getattr(cfg, section), field or arg_name, bool(val))


def canonicalize_task_mode(task_mode: str) -> str:
    canonical = TASK_MODE_ALIASES.get(task_mode, task_mode)
    if canonical not in TASK_MODE_TO_ENABLED:
        raise ValueError(f"Unknown task_mode: {task_mode}")
    return canonical


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.map(normalize_id)
    out = out[~out.index.duplicated(keep="first")]
    return out


def remap_domain_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, List[int]]:
    out = df.copy()
    raw_labels = sorted(set(out["domain_map"].astype(int).tolist()))
    if not raw_labels:
        raise ValueError("Domain label table is empty")
    label_to_idx = {raw: idx for idx, raw in enumerate(raw_labels)}
    out["domain_map"] = out["domain_map"].map(label_to_idx).astype(int)
    return out, raw_labels


def _sorted_domain_label_ids(df: pd.DataFrame) -> List[int]:
    if len(df) == 0 or "domain_map" not in df.columns:
        return []
    return sorted(df["domain_map"].astype(int).unique().tolist())


def _build_task_feature_coverages(
    cfg,
    enabled_tasks: Set[str],
) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, Dict[str, int]]]:
    base_variant_paths = [cfg.paths.variant_x, cfg.paths.legacy_variant_x]
    base_protein_paths = [cfg.paths.protein_x, cfg.paths.legacy_protein_x]
    task_paths = {
        "main": (list(base_variant_paths), list(base_protein_paths)),
        "domain": (list(base_variant_paths), list(base_protein_paths)),
        "func": (list(base_variant_paths), list(base_protein_paths)),
    }
    if "domain" in enabled_tasks:
        task_paths["domain"][0].append(cfg.paths.domain_variant_x)
    if "func" in enabled_tasks:
        task_paths["func"][0].append(cfg.paths.mvp_hgvs_embeddings)
        task_paths["func"][1].append(cfg.paths.mvp_protein_x)

    task_coverages: Dict[str, Dict[str, Set[str]]] = {}
    task_catalog: Dict[str, Dict[str, int]] = {}
    for task_name, (variant_paths, protein_paths) in task_paths.items():
        covered_variant_ids, covered_gene_ids = build_main_split_coverage_sets(
            variant_paths=variant_paths,
            protein_paths=protein_paths,
            gene_path=cfg.paths.gene_x,
        )
        task_coverages[task_name] = {
            "variant_ids": covered_variant_ids,
            "gene_ids": covered_gene_ids,
        }
        task_catalog[task_name] = {
            "covered_variants": int(len(covered_variant_ids)),
            "covered_genes": int(len(covered_gene_ids)),
        }
    return task_coverages, task_catalog


def _build_split_input_signature(
    enabled_tasks: Set[str],
    task_feature_coverage_catalog: Dict[str, Dict[str, int]],
    task_prepare_stats: Dict[str, Dict[str, int]],
    split_domain_df: pd.DataFrame,
    split_func_df: pd.DataFrame,
) -> Dict[str, object]:
    return {
        "enabled_tasks_for_split": sorted([t for t in enabled_tasks if t in {"domain", "func"}]),
        "task_feature_coverage_catalog": task_feature_coverage_catalog,
        "task_prepare_stats": task_prepare_stats,
        "split_seed_rows": {
            "domain": int(len(split_domain_df)),
            "func": int(len(split_func_df)),
        },
    }


def _summarize_domain_split_full_random(
    domain_split: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> Dict[str, Dict[str, int]]:
    train_df, val_df, test_df = domain_split
    train_labels = set(train_df["domain_map"].astype(int).tolist()) if len(train_df) > 0 else set()
    stats: Dict[str, Dict[str, int]] = {}
    for split_name, df in zip(("train", "val", "test"), (train_df, val_df, test_df)):
        split_labels = set(df["domain_map"].astype(int).tolist()) if len(df) > 0 else set()
        stats[split_name] = {
            "rows": int(len(df)),
            "labels": int(len(split_labels)),
            "unseen_vs_train_labels": int(len(split_labels - train_labels)) if split_name != "train" else 0,
        }
    return stats


def _prepare_domain_split(
    domain_split: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    raw_domain_labels: List[int],
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], int, Dict[str, object]]:
    if not raw_domain_labels:
        raise ValueError("Domain label remap must run before preparing domain split")
    train_label_count = int(domain_split[0]["domain_map"].astype(int).nunique()) if len(domain_split[0]) > 0 else 0
    return domain_split, int(len(raw_domain_labels)), {
        "data_mode": "full_random",
        "global_domain_labels": int(len(raw_domain_labels)),
        "train_seen_domain_labels": train_label_count,
        "full_random_pool": _summarize_domain_split_full_random(domain_split),
    }


def load_domain_embedding_tensor(
    path: str,
    raw_label_ids: List[int],
    embedding_dim: int,
) -> torch.Tensor:
    num_domains = len(raw_label_ids)
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_numeric(df.index, errors="coerce")
    except Exception:
        pass

    if df.shape[1] != embedding_dim:
        df2 = pd.read_csv(path)
        numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
        if len(numeric_cols) < embedding_dim:
            raise ValueError(
                f"Domain embedding width mismatch: got {df.shape[1]}, expected {embedding_dim}"
            )
        df = df2[numeric_cols[:embedding_dim]]

    if pd.api.types.is_numeric_dtype(df.index):
        missing = [lab for lab in raw_label_ids if lab not in set(df.index.astype(int).tolist())]
        if missing:
            raise ValueError(f"Domain embedding file missing raw labels: {missing[:10]}")
        arr = df.loc[raw_label_ids, df.columns[:embedding_dim]].to_numpy(dtype=np.float32)
    else:
        arr = df.to_numpy(dtype=np.float32)
        if arr.shape[0] < num_domains:
            raise ValueError(
                f"Domain embedding rows ({arr.shape[0]}) < num_domains ({num_domains})"
            )
        arr = arr[:num_domains, :embedding_dim]

    return torch.tensor(arr, dtype=torch.float32)


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def configure_trainable_modules(
    model: MultiTaskModel,
    graph_train_mode: str,
) -> None:
    set_requires_grad(model.variant_encoder, True)
    set_requires_grad(model.protein_encoder, True)
    set_requires_grad(model.fusion, True)
    set_requires_grad(model.func_conservation_head, True)
    set_requires_grad(model.func_protein_impact_head, True)
    set_requires_grad(model.func_integrative_head, True)
    if model.func_mechanism_head is not None:
        set_requires_grad(model.func_mechanism_head, True)
    set_requires_grad(model.clip_variant_proj, True)
    set_requires_grad(model.clip_disease_proj, True)
    set_requires_grad(model.domain_variant_proj, True)
    set_requires_grad(model.domain_transform, True)
    if model.disease_encoder is not None:
        set_requires_grad(model.disease_encoder, True)
    if model.disease_id_emb is not None:
        set_requires_grad(model.disease_id_emb, True)
    if graph_train_mode == "frozen":
        set_requires_grad(model.graph_encoder, False)
    else:
        set_requires_grad(model.graph_encoder, True)


def build_optimizer_and_scheduler(
    model: MultiTaskModel,
    lr: float,
    lr_graph: float,
    weight_decay: float,
    graph_train_mode: str,
    lr_disease_encoder: float = 1e-4,
    logit_scale_lr_mult: float = 1.0,
    scheduler_t0: int = 20,
    scheduler_t_mult: int = 2,
    scheduler_eta_min: float = 1e-6,
) -> tuple[torch.optim.Optimizer, object]:
    graph_params = list(model.graph_encoder.parameters())
    graph_param_ids = {id(p) for p in graph_params}
    # DiseaseEncoder gets its own param group so it is never accidentally frozen
    # when graph_train_mode="frozen" (e.g. graph_mode="none" ablation).
    disease_enc_params: list = []
    disease_enc_param_ids: set = set()
    if model.disease_encoder is not None:
        disease_enc_params = list(model.disease_encoder.parameters())
    elif model.disease_id_emb is not None:
        disease_enc_params = list(model.disease_id_emb.parameters())
    disease_enc_param_ids = {id(p) for p in disease_enc_params}
    logit_scale_param_ids = {id(model.main_logit_scale_log)}
    reserved_ids = graph_param_ids | disease_enc_param_ids | logit_scale_param_ids
    other_params = [
        p for p in model.parameters()
        if id(p) not in reserved_ids
    ]
    graph_lr = lr_graph
    if graph_train_mode == "frozen":
        graph_lr = 0.0
    elif graph_train_mode == "weak":
        graph_lr = min(lr_graph, lr * 0.1)
    elif graph_train_mode == "full":
        graph_lr = lr_graph
    else:
        raise ValueError(f"Unknown graph_train_mode: {graph_train_mode}")
    # DiseaseEncoder has its own LR (no pretrained weights to protect).
    disease_enc_lr = lr_disease_encoder if disease_enc_params else 0.0

    logit_scale_lr = lr * logit_scale_lr_mult
    param_groups = [
        {"params": [p for p in other_params if p.requires_grad], "lr": lr},
        {"params": [p for p in graph_params if p.requires_grad], "lr": graph_lr},
    ]
    if disease_enc_params:
        param_groups.append(
            {"params": [p for p in disease_enc_params if p.requires_grad], "lr": disease_enc_lr}
        )
    if model.main_logit_scale_log.requires_grad:
        param_groups.append(
            {"params": [model.main_logit_scale_log], "lr": logit_scale_lr, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_t0,
        T_mult=scheduler_t_mult,
        eta_min=scheduler_eta_min,
    )
    return optimizer, scheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal PheMART2 multi-task runner")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--export-predictions", type=int, default=1,
                   help="Export per-example predictions CSV after test eval (0=off, 1=on)")
    p.add_argument("--eval-only", type=str, default=None,
                   help="Skip training, load checkpoint (best_model.pt), run test eval only")
    p.add_argument("--main-labels", type=str, default=None)
    p.add_argument("--disease-table", type=str, default=None)
    p.add_argument("--batch-size-main", type=int, default=None)
    p.add_argument("--batch-size-domain", type=int, default=None)
    p.add_argument("--batch-size-func", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-graph", type=float, default=None)
    p.add_argument("--lr-disease-encoder", type=float, default=None,
                   help="DiseaseEncoder LR (default 1e-4; independent of lr_graph)")
    p.add_argument("--lr-graph-warmup", type=float, default=None,
                   help="Graph LR during warmup phase (e.g. 1e-3 to match main LR)")
    p.add_argument("--graph-warmup-epochs", type=int, default=None,
                   help="Epochs to use boosted graph LR before decaying to lr_graph")
    p.add_argument("--graph-cache-refresh-steps", type=int, default=None,
                   help="Trainable graph cache refresh interval in optimizer steps (1 = recompute every step)")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--eval-interval", type=int, default=None,
                   help="Run full validation every N epochs (default 3, 1=every epoch)")
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument(
        "--task-mode",
        type=str,
        choices=TASK_MODE_CHOICES,
        default=None,
    )
    p.add_argument("--main-temperature", type=float, default=None)
    p.add_argument("--domain-temperature", type=float, default=None)
    p.add_argument(
        "--domain-loss-type",
        type=str,
        choices=["sampled_infonce"],
        default=None,
    )
    p.add_argument("--domain-contrastive-negatives", type=int, default=None)
    p.add_argument(
        "--domain-data-mode",
        type=str,
        choices=["full_random"],
        default=None,
    )
    p.add_argument("--domain-train-per-label-cap", type=int, default=None)
    p.add_argument("--main-early-stop-metric", type=str, default=None)
    p.add_argument("--main-loss-type", type=str, choices=["softmax", "bce"], default=None)
    p.add_argument("--main-logit-scale-learnable", type=int, choices=[0, 1], default=None)
    p.add_argument("--main-logit-scale-min", type=float, default=None)
    p.add_argument("--main-logit-scale-max", type=float, default=None)
    p.add_argument("--main-logit-scale-init", type=float, default=None)
    p.add_argument("--main-logit-scale-lr-mult", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--aux-update-hgt", type=int, choices=[0, 1], default=None)
    p.add_argument("--aux-domain-interval", type=int, default=None)
    p.add_argument("--aux-func-interval", type=int, default=None)
    p.add_argument("--main-only-warmup-epochs", type=int, default=None)
    p.add_argument("--gate-entropy-weight-start", type=float, default=None)
    p.add_argument("--gate-entropy-weight-end", type=float, default=None)
    p.add_argument("--graph-visibility", type=str, choices=["inductive", "transductive"], default=None)
    p.add_argument("--graph-train-mode", type=str, choices=["frozen", "weak", "full"], default=None)
    p.add_argument("--graph-mode", type=str, choices=["hgt", "none"], default=None,
                   help="Graph encoder mode: hgt (default) or none (disable message passing)")
    p.add_argument("--enrich-trait-graph", type=int, choices=[0, 1], default=None,
                   help="Use enriched trait-to-trait edges (HPO is_a + cosine kNN)")
    p.add_argument("--residual-alpha-max", type=float, default=None,
                   help="Cap for ConcatResidualFusion residual bypass (default 1.0 = no cap)")
    p.add_argument(
        "--split-protocol",
        type=str,
        choices=["gene_holdout", "within_gene", "disease_holdout"],
        default=None,
    )
    p.add_argument("--split-mode", type=str, choices=["auto", "generate", "load"], default=None)
    p.add_argument("--split-only", type=int, choices=[0, 1], default=None)
    p.add_argument("--split-artifact-path", type=str, default=None)
    p.add_argument("--func-min-valid-axes", type=int, default=None)
    p.add_argument("--func-mechanism-pos-weight", type=float, default=None)
    p.add_argument("--func-train-per-gene-cap", type=int, default=None)
    p.add_argument("--min-train-records-func", type=int, default=None)
    p.add_argument("--mvp-hpo-semantic-map", type=str, default=None)
    p.add_argument("--mvp-disease-concept-map", type=str, default=None)
    p.add_argument("--mvp-topk-indices", type=str, default=None,
                    help="Override MVP top-K indices path (e.g. raw_corr_topk_indices_EUR_k256.npy)")
    p.add_argument("--mvp-topk-values", type=str, default=None,
                    help="Override MVP top-K values path (e.g. raw_corr_topk_values_EUR_k256.npy)")
    p.add_argument("--mvp-topk-metadata", type=str, default=None,
                    help="Override MVP top-K metadata path (e.g. raw_corr_topk_metadata_EUR_k256.json)")
    p.add_argument("--vd-kl-max-diseases-per-concept", type=int, default=None)
    p.add_argument("--vd-kl-concept-specificity-weighting", type=int, choices=[0, 1], default=None)
    p.add_argument("--vd-kl-bridge-length-penalty-b", type=float, default=None)
    p.add_argument("--vd-kl-hpo-min-terms-for-full-weight", type=int, default=None)
    p.add_argument("--vd-kl-teacher-max-concepts", type=int, default=None,
                   help="Only use top-K mapped concepts per variant (0=all)")
    p.add_argument("--vd-kl-teacher-concept-temperature", type=float, default=None,
                   help="Temperature for concept prob sharpening (<1=sharper, 1=no change)")
    p.add_argument("--modality-drop-variant", type=float, default=None)
    p.add_argument("--modality-drop-protein", type=float, default=None)
    p.add_argument("--modality-drop-gene", type=float, default=None)
    p.add_argument("--trait-dropout", type=float, default=None)
    p.add_argument("--disease-size-embed", type=int, choices=[0, 1], default=None)
    p.add_argument(
        "--fusion-type",
        type=str,
        choices=["gated", "concat", "concat_residual"],
        default=None,
    )
    p.add_argument(
        "--disease-freq-reweight",
        type=str,
        choices=["none", "sqrt_inv", "log_inv"],
        default=None,
    )
    p.add_argument(
        "--disease-freq-weight-agg",
        type=str,
        choices=["max", "mean"],
        default=None,
    )
    p.add_argument("--disease-freq-weight-clip", type=float, default=None)
    p.add_argument("--enable-vd-kl", type=int, choices=[0, 1], default=None)
    p.add_argument(
        "--vd-kl-loss-type",
        type=str,
        choices=["kl", "set_infonce", "weighted_set_infonce", "slack_constraint"],
        default=None,
        help="kl: match teacher distribution (default); set_infonce: multi-positive InfoNCE over support set",
    )
    p.add_argument(
        "--vd-kl-teacher-mode",
        type=str,
        choices=["concept_map", "hpo_semantic", "hybrid"],
        default=None,
    )
    p.add_argument("--vd-kl-lambda-v2d", type=float, default=None)
    p.add_argument("--vd-kl-lambda-d2v", type=float, default=None)
    p.add_argument("--vd-kl-lambda-v2d-start", type=float, default=None)
    p.add_argument("--vd-kl-lambda-d2v-start", type=float, default=None)
    p.add_argument("--vd-kl-lambda-ramp-epochs", type=int, default=None)
    p.add_argument("--vd-kl-temperature", type=float, default=None)
    p.add_argument("--vd-kl-warmup-epochs", type=int, default=None)
    p.add_argument("--vd-kl-d2v-start-epoch", type=int, default=None)
    p.add_argument("--vd-kl-cache-refresh-interval", type=int, default=None)
    p.add_argument("--vd-kl-d2v-batch-size", type=int, default=None)
    p.add_argument("--vd-kl-hpo-topk-per-hpo", type=int, default=None)
    p.add_argument("--vd-kl-hpo-min-similarity", type=float, default=None)
    p.add_argument("--vd-kl-concept-direct-weight", type=float, default=None)
    p.add_argument("--vd-kl-concept-semantic-weight", type=float, default=None)
    p.add_argument("--vd-kl-min-variant-mapped-mass", type=float, default=None)
    p.add_argument("--vd-kl-max-diseases-per-variant", type=int, default=None)
    p.add_argument("--vd-kl-disease-score-concentration", type=float, default=None,
                   help="Keep disease in support set only if score > C × uniform_baseline (0=disabled, 2.0=recommended)")
    p.add_argument("--vd-kl-max-variants-per-disease", type=int, default=None)
    p.add_argument("--vd-kl-positive-smoothing", type=float, default=None)
    p.add_argument("--vd-kl-d2v-positive-smoothing", type=float, default=None)
    p.add_argument("--vd-kl-d2v-min-anchor-mass", type=float, default=None)
    p.add_argument("--vd-kl-d2v-min-rows-per-step", type=int, default=None)
    p.add_argument("--vd-kl-min-teacher-top1-prob", type=float, default=None)
    p.add_argument("--vd-kl-d2v-teacher-topk", type=int, default=None)
    p.add_argument("--vd-kl-d2v-random-negatives", type=int, default=None)
    p.add_argument("--vd-kl-d2v-max-positive-variants", type=int, default=None)
    p.add_argument("--vd-kl-adaptive-weight", type=int, choices=[0, 1], default=None)
    p.add_argument("--vd-kl-adaptive-reference-scale", type=float, default=None)
    p.add_argument("--vd-kl-gene-propagation", type=int, choices=[0, 1], default=None,
                    help="Propagate teacher signal to same-gene variants without direct MVP teacher")
    p.add_argument("--vd-kl-gene-propagation-alpha", type=float, default=None,
                    help="KL weight decay for gene-propagated variants (default 0.3)")
    p.add_argument("--vd-kl-gene-propagation-distance-lambda", type=float, default=None,
                    help="Distance decay for position-weighted propagation (default 0.01, 0=uniform)")
    p.add_argument("--vd-kl-gene-propagation-max-distance", type=int, default=None,
                    help="Max aa distance for propagation sources (default 0=unlimited)")
    p.add_argument("--vd-kl-gene-propagation-adaptive-alpha", type=int, choices=[0, 1], default=None,
                    help="Alpha decays with distance to nearest source (default 0)")
    p.add_argument("--vd-kl-gene-propagation-exclude-d2v", type=int, choices=[0, 1], default=None,
                    help="Exclude propagated variants from D2V reverse teacher (default 1)")
    p.add_argument("--vd-kl-gene-propagation-entropy-threshold", type=float, default=None,
                    help="Filter propagated variants with normalized entropy > threshold (default 0=disabled)")
    p.add_argument("--vd-kl-gene-propagation-position-unknown-penalty", type=float, default=None,
                    help="Weight multiplier when protein position cannot be parsed (default 0.3)")
    p.add_argument("--vd-kl-gene-propagation-confidence-scaling", type=int, choices=[0, 1], default=None,
                    help="Scale alpha by source-count confidence: 1-exp(-0.5*n) (default 0)")
    p.add_argument("--vd-kl-quality-row-weight", type=int, choices=[0, 1], default=None,
                    help="Use mapped_mass as row weight for direct-mapped variants (default 0=uniform 1.0)")
    p.add_argument("--vd-kl-prior-correction-alpha", type=float, default=None,
                    help="Prior-correct disease scores: q'_d ∝ q_d/π(d)^α; 0=disabled; >0 boosts rare diseases")
    p.add_argument("--vd-kl-slack-tau", type=float, default=None,
                    help="Target mass-on-support for slack_constraint loss (default 0.5)")
    p.add_argument("--vd-kl-raw-corr-mode", type=int, choices=[0, 1], default=None,
                    help="Raw correlation pipeline: skip concept temp/renorm, use |corr| threshold (default 0)")
    p.add_argument("--vd-kl-min-concept-corr", type=float, default=None,
                    help="Min |correlation| to include a concept (only in raw_corr_mode, default 0.01)")
    p.add_argument("--vd-kl-disease-score-power", type=float, default=None,
                    help="Power transform on disease_scores before normalization; >1 sharpens (default 1.0)")
    p.add_argument("--vd-kl-concept-quality-scaling", type=int, choices=[0, 1], default=None,
                    help="Scale per-concept weights by max(original_scores) to preserve sigmoid calibration (default 0)")
    p.add_argument("--main-hard-negative-k", type=int, default=None,
                    help="Hard negative mining: keep top-k hardest negatives per sample (0=disabled)")
    p.add_argument("--vd-kl-concept-filter-level", type=int, choices=[0, 1, 2], default=None,
                    help="Teacher concept filter: 0=none, 1=remove PheCodes+exact disease names, 2=L1+fuzzy 70%%")
    p.add_argument("--vd-kl-shuffle-teacher", type=int, choices=[0, 1], default=None,
                    help="Shuffle teacher concept indices (negative control)")
    p.add_argument("--loss-weight-main", type=float, default=None)
    p.add_argument("--loss-weight-domain", type=float, default=None)
    p.add_argument("--loss-weight-func", type=float, default=None)
    p.add_argument("--loss-weight-concept", type=float, default=None)
    p.add_argument("--enable-concept-regression", type=int, choices=[0, 1], default=None,
                    help="Enable gene-level concept profile regression (0=off, 1=on)")
    p.add_argument("--subsample-fraction", type=float, default=None,
                    help="Subsample training main-task labels (0.0-1.0) for label-efficiency experiments")
    p.add_argument("--num-graph-layers", type=int, default=None,
                    help="Override number of HGT graph layers (0 disables graph encoding)")
    p.add_argument("--disease-encoder-type", type=str, default=None,
                    choices=["hpo_attention", "disease_id"],
                    help="Disease encoder type: hpo_attention (default) or disease_id (learnable embedding)")
    p.add_argument("--data-root", type=str, default=None,
                    help="Override base data directory (replaces ../data/ prefix in all default paths)")
    p.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader num_workers (default: 2 for GPU, 0 for CPU)")
    p.add_argument("--hidden-dim", type=int, default=None,
                    help="Model hidden dimension (default 256)")
    p.add_argument("--out-dim", type=int, default=None,
                    help="Model output / CLIP embedding dimension (default 128)")
    p.add_argument("--scheduler-t0", type=int, default=None,
                    help="CosineAnnealingWarmRestarts T_0 (default 20)")
    p.add_argument("--scheduler-t-mult", type=int, default=None,
                    help="CosineAnnealingWarmRestarts T_mult (default 2)")
    p.add_argument("--scheduler-eta-min", type=float, default=None,
                    help="CosineAnnealingWarmRestarts eta_min (default 1e-6)")
    return p.parse_args()


def _prepare_task_records_by_mode(
    records: Dict[str, Dict[str, List[Dict[str, object]]]],
    enabled_tasks: Set[str],
) -> None:
    for split_name in ["train", "val", "test"]:
        for task_name in ["domain", "func"]:
            if task_name not in enabled_tasks:
                records[split_name][task_name] = []


def main() -> None:
    args = parse_args()
    cfg = default_config()
    # --data-root: rewrite all ../data/ prefixed paths in PathsConfig
    if args.data_root is not None:
        data_root = args.data_root.rstrip("/")
        old_prefix = "../data/"
        for field_name in vars(cfg.paths):
            val = getattr(cfg.paths, field_name)
            if isinstance(val, str) and val.startswith(old_prefix):
                setattr(cfg.paths, field_name, val.replace(old_prefix, data_root + "/", 1))
    _apply_args_to_config(args, cfg)
    split_only = bool(args.split_only) if args.split_only is not None else False

    disease_encoder_type = args.disease_encoder_type or "hpo_attention"

    subsample_fraction = args.subsample_fraction

    task_mode_arg = args.task_mode or "main_domain_func"
    task_mode = canonicalize_task_mode(task_mode_arg)
    enabled_tasks = TASK_MODE_TO_ENABLED[task_mode]
    if cfg.train.batch_size_main <= 0 or cfg.train.batch_size_domain <= 0 or cfg.train.batch_size_func <= 0:
        raise ValueError("batch_size_main/batch_size_domain/batch_size_func must be > 0")
    if cfg.train.lr <= 0 or cfg.train.lr_graph <= 0:
        raise ValueError("lr/lr_graph must be > 0")
    if cfg.train.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if cfg.train.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be > 0")
    if cfg.train.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if cfg.train.domain_temperature <= 0:
        raise ValueError("domain_temperature must be > 0")
    if cfg.train.domain_loss_type != "sampled_infonce":
        raise ValueError("domain_loss_type must be sampled_infonce")
    if cfg.train.domain_contrastive_negatives <= 0:
        raise ValueError("domain_contrastive_negatives must be > 0")
    if cfg.train.domain_data_mode != "full_random":
        raise ValueError("domain_data_mode must be full_random")
    if cfg.train.domain_train_per_label_cap < 0:
        raise ValueError("domain_train_per_label_cap must be >= 0")
    if cfg.loss_weights.main < 0 or cfg.loss_weights.domain < 0 or cfg.loss_weights.func < 0:
        raise ValueError("loss_weight_main/loss_weight_domain/loss_weight_func must be >= 0")
    if cfg.train.main_loss_type != "softmax":
        print("warning: main_loss_type is not softmax; this is not recommended for primary conclusions")
    if cfg.train.vd_kl_lambda_v2d < 0 or cfg.train.vd_kl_lambda_d2v < 0:
        raise ValueError("vd_kl_lambda_v2d/vd_kl_lambda_d2v must be >= 0")
    if cfg.train.vd_kl_lambda_v2d_start < 0 or cfg.train.vd_kl_lambda_d2v_start < 0:
        raise ValueError("vd_kl_lambda_v2d_start/vd_kl_lambda_d2v_start must be >= 0")
    if cfg.train.vd_kl_lambda_ramp_epochs < 0:
        raise ValueError("vd_kl_lambda_ramp_epochs must be >= 0")
    if cfg.train.vd_kl_temperature <= 0:
        raise ValueError("vd_kl_temperature must be > 0")
    if cfg.train.vd_kl_cache_refresh_interval <= 0:
        raise ValueError("vd_kl_cache_refresh_interval must be > 0")
    if cfg.train.vd_kl_d2v_batch_size <= 0:
        raise ValueError("vd_kl_d2v_batch_size must be > 0")
    if cfg.train.vd_kl_loss_type not in {"kl", "set_infonce", "weighted_set_infonce", "slack_constraint"}:
        raise ValueError(f"Unknown vd_kl_loss_type: {cfg.train.vd_kl_loss_type}")
    if cfg.train.vd_kl_slack_tau <= 0 or cfg.train.vd_kl_slack_tau >= 1:
        raise ValueError(f"vd_kl_slack_tau must be in (0, 1), got {cfg.train.vd_kl_slack_tau}")
    # --- set_infonce / weighted_set_infonce: 不再强制覆盖 max_diseases_per_variant ---
    # 旧逻辑把 max_diseases_per_variant 强制设为 0（无截断），导致 |S|≈700，
    # gradient vanishing。现在保留用户设置（默认 32），允许公平对比。
    if cfg.train.enable_vd_kl and cfg.train.vd_kl_loss_type in ("set_infonce", "weighted_set_infonce"):
        if cfg.train.vd_kl_teacher_mode != "concept_map" and cfg.train.vd_kl_max_diseases_per_variant == 0:
            print(f"[{cfg.train.vd_kl_loss_type}] WARNING: teacher_mode={cfg.train.vd_kl_teacher_mode} "
                  f"+ max_diseases_per_variant=0 → |S|≈700 (44% of diseases)。"
                  f"建议设置 --vd-kl-max-diseases-per-variant 32 或使用 --vd-kl-teacher-mode concept_map。")
    if cfg.train.vd_kl_teacher_mode not in {"concept_map", "hpo_semantic", "hybrid"}:
        raise ValueError(f"Unknown vd_kl_teacher_mode: {cfg.train.vd_kl_teacher_mode}")
    if cfg.train.vd_kl_hpo_topk_per_hpo <= 0:
        raise ValueError("vd_kl_hpo_topk_per_hpo must be > 0")
    if cfg.train.vd_kl_hpo_min_similarity < 0:
        raise ValueError("vd_kl_hpo_min_similarity must be >= 0")
    if cfg.train.vd_kl_concept_direct_weight < 0 or cfg.train.vd_kl_concept_semantic_weight < 0:
        raise ValueError("vd_kl_concept_direct_weight/vd_kl_concept_semantic_weight must be >= 0")
    if cfg.train.vd_kl_min_variant_mapped_mass < 0 or cfg.train.vd_kl_min_variant_mapped_mass > 1.0:
        raise ValueError("vd_kl_min_variant_mapped_mass must be in [0, 1]")
    if cfg.train.vd_kl_max_diseases_per_variant < 0:
        raise ValueError("vd_kl_max_diseases_per_variant must be >= 0")
    if cfg.train.vd_kl_max_variants_per_disease < 0:
        raise ValueError("vd_kl_max_variants_per_disease must be >= 0")
    if cfg.train.vd_kl_positive_smoothing < 0 or cfg.train.vd_kl_positive_smoothing > 1.0:
        raise ValueError("vd_kl_positive_smoothing must be in [0, 1]")
    if cfg.train.vd_kl_d2v_positive_smoothing < 0 or cfg.train.vd_kl_d2v_positive_smoothing > 1.0:
        raise ValueError("vd_kl_d2v_positive_smoothing must be in [0, 1]")
    if cfg.train.vd_kl_d2v_min_anchor_mass < 0 or cfg.train.vd_kl_d2v_min_anchor_mass > 1.0:
        raise ValueError("vd_kl_d2v_min_anchor_mass must be in [0, 1]")
    if cfg.train.vd_kl_d2v_min_rows_per_step <= 0:
        raise ValueError("vd_kl_d2v_min_rows_per_step must be > 0")
    if cfg.train.vd_kl_d2v_teacher_topk <= 0:
        raise ValueError("vd_kl_d2v_teacher_topk must be > 0")
    if cfg.train.vd_kl_d2v_random_negatives < 0:
        raise ValueError("vd_kl_d2v_random_negatives must be >= 0")
    if cfg.train.vd_kl_d2v_max_positive_variants <= 0:
        raise ValueError("vd_kl_d2v_max_positive_variants must be > 0")
    if cfg.train.vd_kl_min_teacher_top1_prob < 0 or cfg.train.vd_kl_min_teacher_top1_prob > 1.0:
        raise ValueError("vd_kl_min_teacher_top1_prob must be in [0, 1]")
    if cfg.train.main_only_warmup_epochs < 0:
        raise ValueError("main_only_warmup_epochs must be >= 0")
    if cfg.train.graph_warmup_epochs < 0:
        raise ValueError("graph_warmup_epochs must be >= 0")
    if cfg.train.lr_graph_warmup < 0:
        raise ValueError("lr_graph_warmup must be >= 0")
    if cfg.train.graph_cache_refresh_steps <= 0:
        raise ValueError("graph_cache_refresh_steps must be > 0")
    if cfg.train.disease_freq_weight_agg not in {"max", "mean"}:
        raise ValueError(f"Unknown disease_freq_weight_agg: {cfg.train.disease_freq_weight_agg}")
    if cfg.train.disease_freq_weight_clip < 0:
        raise ValueError("disease_freq_weight_clip must be >= 0")
    if cfg.model.residual_alpha_max <= 0 or cfg.model.residual_alpha_max > 1.0:
        raise ValueError("residual_alpha_max must be in (0, 1]")

    if cfg.model.graph_mode == "none" and cfg.train.graph_train_mode == "frozen":
        # NoGraphEncoder with frozen = random projections permanently frozen.
        # This is almost certainly unintentional; force to "weak".
        print(f"WARNING: graph_mode=none + graph_train_mode=frozen would freeze NoGraphEncoder "
              f"at random init → overriding to weak")
        cfg.train.graph_train_mode = "weak"
    elif cfg.model.graph_mode == "none" and cfg.train.graph_train_mode == "full":
        print(f"graph_mode=none → capping graph_train_mode to weak (was full)")
        cfg.train.graph_train_mode = "weak"
    _use_inductive_graph_train = cfg.train.graph_visibility == "inductive"
    set_seed(cfg.split.seed)
    out_dir = ensure_output_dir(cfg)
    device = torch.device(cfg.runtime.device)
    if task_mode_arg != task_mode:
        print(f"task_mode_alias={task_mode_arg}->{task_mode}")
    print(f"task_mode={task_mode} enabled_tasks={sorted(enabled_tasks)}")
    print(
        f"split_protocol={cfg.split.protocol} "
        f"split_mode={cfg.split.mode} split_only={split_only} "
        f"graph_visibility={cfg.train.graph_visibility} "
        f"graph_cache_refresh_steps={cfg.train.graph_cache_refresh_steps} "
        f"main_temp={cfg.train.main_temperature:.4f} early_stop={cfg.train.main_early_stop_metric} "
        f"domain_loss={cfg.train.domain_loss_type} domain_data={cfg.train.domain_data_mode} "
        f"domain_cap={cfg.train.domain_train_per_label_cap} "
        f"main_warmup={cfg.train.main_only_warmup_epochs} "
        f"freq_reweight={cfg.train.disease_freq_reweight}:{cfg.train.disease_freq_weight_agg}:{cfg.train.disease_freq_weight_clip} "
        f"vd_kl={cfg.train.enable_vd_kl} vd_loss_type={cfg.train.vd_kl_loss_type} vd_teacher={cfg.train.vd_kl_teacher_mode}"
    )
    print(f"device={device}")

    print("[1/8] loading labels")
    main_df = load_main_labels(cfg.paths.main_labels)
    disease_df = load_disease_table(cfg.paths.disease_table)
    domain_df = load_domain_labels(cfg.paths.domain_labels)
    print("func_regression_cols=" + json.dumps(FUNC_REGRESSION_COLS, ensure_ascii=False))

    raw_domain_labels: List[int] = []
    num_domains = 0
    func_df = load_func_labels(cfg.paths.func_labels)
    split_seed_df = pd.DataFrame(columns=["variant_id", "gene_id"])

    rsid_to_hgvs = build_rsid_to_hgvs_map(
        hgvs_embed_csv=cfg.paths.mvp_hgvs_embeddings,
        rsid_embed_csv=cfg.paths.mvp_rsid_embeddings,
    )
    print(f"rsid_to_hgvs_size={len(rsid_to_hgvs)}")
    task_coverages, task_feature_coverage_catalog = _build_task_feature_coverages(cfg, enabled_tasks)
    main_covered_variant_ids = task_coverages["main"]["variant_ids"]
    main_covered_gene_ids = task_coverages["main"]["gene_ids"]
    domain_covered_variant_ids = task_coverages["domain"]["variant_ids"]
    domain_covered_gene_ids = task_coverages["domain"]["gene_ids"]
    func_covered_variant_ids = task_coverages["func"]["variant_ids"]
    func_covered_gene_ids = task_coverages["func"]["gene_ids"]
    print(
        "feature_coverage_catalog="
        + json.dumps(task_feature_coverage_catalog, ensure_ascii=False)
    )

    func_df = remap_variant_ids_to_hgvs(
        func_df,
        rsid_to_hgvs,
        task_name="func",
        preserve_ids=func_covered_variant_ids,
    )
    func_df = func_df.drop_duplicates(subset=["variant_id", "gene_id"], keep="first").copy()
    main_df, main_prepare_stats = prepare_task_dataframe_for_training(
        main_df,
        task_name="main",
        covered_variant_ids=main_covered_variant_ids,
        covered_gene_ids=main_covered_gene_ids,
    )
    domain_df, domain_prepare_stats = prepare_task_dataframe_for_training(
        domain_df,
        task_name="domain",
        covered_variant_ids=domain_covered_variant_ids,
        covered_gene_ids=domain_covered_gene_ids,
    )
    func_df, func_prepare_stats = prepare_task_dataframe_for_training(
        func_df,
        task_name="func",
        covered_variant_ids=func_covered_variant_ids,
        covered_gene_ids=func_covered_gene_ids,
    )
    task_prepare_stats = {
        "main": main_prepare_stats,
        "domain": domain_prepare_stats,
        "func": func_prepare_stats,
    }
    print("task_prepare_stats=" + json.dumps(task_prepare_stats, ensure_ascii=False))
    if "domain" in enabled_tasks:
        domain_df, raw_domain_labels = remap_domain_labels(domain_df)
        num_domains = len(raw_domain_labels)
        print(f"domain_label_remap=num_domains={num_domains}")
    split_domain_df = domain_df if "domain" in enabled_tasks else domain_df.iloc[0:0].copy()
    split_func_df = func_df if "func" in enabled_tasks else func_df.iloc[0:0].copy()
    split_input_signature = _build_split_input_signature(
        enabled_tasks,
        task_feature_coverage_catalog,
        task_prepare_stats,
        split_domain_df,
        split_func_df,
    )

    print("[2/8] preparing split artifact")
    split_artifact_path = Path(cfg.paths.split_artifact_path)
    if split_artifact_path.suffix == "":
        split_artifact_path = split_artifact_path / f"{cfg.split.protocol}_seed{cfg.split.seed}.json"
    if not split_artifact_path.is_absolute():
        split_artifact_path = Path.cwd() / split_artifact_path

    artifact = None
    if cfg.split.protocol == "gene_holdout":
        split_strategy_tag = "gene_holdout_stratified_size_v5"
    elif cfg.split.protocol == "within_gene":
        split_strategy_tag = "within_gene_variant_v2"
    elif cfg.split.protocol == "disease_holdout":
        split_strategy_tag = "disease_holdout_stratified_disease_v3"
    else:
        raise ValueError(f"Unsupported split protocol: {cfg.split.protocol}")
    if cfg.split.mode == "load" and split_artifact_path.exists():
        artifact = load_split_artifact(str(split_artifact_path))
        metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
        artifact_strategy = ""
        if isinstance(metadata, dict):
            artifact_strategy = str(metadata.get("split_strategy", ""))
        if artifact_strategy != split_strategy_tag:
            raise ValueError(
                "Loaded split artifact strategy mismatch: "
                f"expected={split_strategy_tag} got={artifact_strategy or 'missing'} "
                f"path={split_artifact_path}"
            )
        artifact_signature = metadata.get("split_input_signature") if isinstance(metadata, dict) else None
        if artifact_signature != split_input_signature:
            raise ValueError(
                "Loaded split artifact input signature mismatch; regenerate this artifact with "
                "--split-mode generate under the current preprocessing/task-mode semantics "
                f"path={split_artifact_path}"
            )
        print(f"split_artifact_load={split_artifact_path}")
    elif cfg.split.mode == "load":
        raise FileNotFoundError(f"split artifact required but missing: {split_artifact_path}")
    elif cfg.split.mode in {"auto", "generate"}:
        print(
            "split_artifact_refresh="
            f"{split_artifact_path} mode={cfg.split.mode}; regenerating deterministically"
        )
    else:
        raise ValueError(f"Unsupported split mode: {cfg.split.mode}")

    if artifact is None:
        disease_split_map: Dict[int, str] = {}
        if cfg.split.protocol == "disease_holdout":
            split_map, gene_split_map, disease_split_map = build_disease_holdout_split(
                main_df=main_df,
                domain_df=split_domain_df,
                aux_df=split_seed_df,
                func_df=split_func_df,
                seed=cfg.split.seed,
                ratios=(cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio),
            )
        elif cfg.split.protocol == "gene_holdout":
            split_map, gene_split_map = build_global_variant_split(
                main_df=main_df,
                domain_df=split_domain_df,
                aux_df=split_seed_df,
                func_df=split_func_df,
                seed=cfg.split.seed,
                ratios=(cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio),
                protocol=cfg.split.protocol,
                return_gene_split=True,
                covered_variant_ids=main_covered_variant_ids,
                covered_gene_ids=main_covered_gene_ids,
            )
        else:
            split_map, gene_split_map = build_within_gene_variant_split(
                main_df=main_df,
                domain_df=split_domain_df,
                aux_df=split_seed_df,
                func_df=split_func_df,
                seed=cfg.split.seed,
                ratios=(cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio),
            )
        gene_split_diagnostics = {}
        disease_split_diagnostics = {}
        if cfg.split.protocol == "gene_holdout":
            gene_split_diagnostics = summarize_gene_holdout_split(main_df, gene_split_map)
        elif cfg.split.protocol == "disease_holdout":
            disease_split_diagnostics = summarize_disease_holdout_split(main_df, disease_split_map)
        artifact = build_split_artifact(
            split_map=split_map,
            gene_split_map=gene_split_map,
            disease_split_map=disease_split_map,
            protocol=cfg.split.protocol,
            seed=cfg.split.seed,
            ratios=(cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio),
            metadata={
                "main_rows": int(len(main_df)),
                "domain_rows": int(len(split_domain_df)),
                "aux_rows": 0,
                "func_rows": int(len(split_func_df)),
                "task_feature_coverage_catalog": task_feature_coverage_catalog,
                "task_prepare_stats": task_prepare_stats,
                "split_strategy": split_strategy_tag,
                "split_input_signature": split_input_signature,
                "gene_split_diagnostics": gene_split_diagnostics,
                "disease_split_diagnostics": disease_split_diagnostics,
            },
        )
        save_split_artifact(artifact, str(split_artifact_path))
        print(f"split_artifact_generate={split_artifact_path}")

    split_map = artifact["split_map"]
    gene_split_map = artifact.get("gene_split_map", {})
    disease_split_map = {
        int(k): v for k, v in artifact.get("disease_split_map", {}).items()
    }
    print(f"split_artifact_id={artifact.get('artifact_id', '')}")

    if cfg.split.protocol == "disease_holdout":
        main_split = apply_disease_split(main_df, "disease_index", disease_split_map)
    else:
        main_split = apply_split(main_df, "variant_id", split_map)
    domain_split = apply_split(domain_df, "variant_id", split_map)
    func_split = apply_split(func_df, "variant_id", split_map)
    domain_aux_stats: Dict[str, object] = {}
    if "domain" in enabled_tasks:
        domain_split, num_domains, domain_aux_stats = _prepare_domain_split(
            domain_split,
            raw_domain_labels,
        )
    split_summary = summarize_split(
        {
            "main": main_split,
            "domain": domain_split,
            "func": func_split,
        }
    )
    consistency_task_splits = {
        "domain": domain_split,
        "func": func_split,
    }
    if cfg.split.protocol != "disease_holdout":
        consistency_task_splits["main"] = main_split
    consistency = validate_split_consistency(
        split_map=split_map,
        task_splits=consistency_task_splits,
    )
    if cfg.split.protocol == "disease_holdout":
        main_train_variants = {normalize_id(v) for v in main_split[0]["variant_id"].tolist() if normalize_id(v)}
        main_val_variants = {normalize_id(v) for v in main_split[1]["variant_id"].tolist() if normalize_id(v)}
        main_test_variants = {normalize_id(v) for v in main_split[2]["variant_id"].tolist() if normalize_id(v)}
        consistency["main_train_rows"] = int(len(main_split[0]))
        consistency["main_val_rows"] = int(len(main_split[1]))
        consistency["main_test_rows"] = int(len(main_split[2]))
        consistency["main_train_val_variant_overlap"] = int(
            len(main_train_variants & main_val_variants)
        )
        consistency["main_train_test_variant_overlap"] = int(
            len(main_train_variants & main_test_variants)
        )
        consistency["main_val_test_variant_overlap"] = int(
            len(main_val_variants & main_test_variants)
        )
    print("split_summary=" + json.dumps(split_summary, ensure_ascii=False))
    print("split_consistency=" + json.dumps(consistency, ensure_ascii=False))
    if split_only:
        split_only_result = {
            "split_artifact_path": str(split_artifact_path),
            "split_artifact_id": str(artifact.get("artifact_id", "")),
            "split_protocol": cfg.split.protocol,
            "split_mode": cfg.split.mode,
            "task_mode": task_mode,
            "split_summary": split_summary,
            "split_consistency": consistency,
        }
        with open(out_dir / "split_only_result.json", "w", encoding="utf-8") as f:
            json.dump(split_only_result, f, ensure_ascii=False, indent=2)
        print("split_only_result=" + json.dumps(split_only_result, ensure_ascii=False))
        print("[split-only] finished after split artifact preparation")
        return

    print("[3/8] loading graph data")
    gene_x_df = _normalize_index(load_embeddings(cfg.paths.gene_x))
    trait_x_df = _normalize_index(load_embeddings(cfg.paths.trait_x))
    mappings = build_mappings(gene_x_df, trait_x_df, disease_df)
    gene_size_buckets = build_gene_size_buckets(main_df, mappings["gene_to_idx"])
    print(
        "gene_size_buckets="
        + json.dumps({k: int(len(v)) for k, v in gene_size_buckets.items()}, ensure_ascii=False)
    )
    trait_to_trait_path = cfg.paths.trait_to_trait
    if cfg.model.enrich_trait_graph:
        enriched = cfg.paths.trait_to_trait_enriched
        if not Path(enriched).exists():
            raise FileNotFoundError(
                f"enrich_trait_graph=True but enriched trait graph not found: {enriched}"
            )
        trait_to_trait_path = enriched
        print(f"trait_to_trait=enriched ({enriched})")
    graph = build_hetero_graph(
        gene_x_df=gene_x_df,
        trait_x_df=trait_x_df,
        edge_files={
            "gene_to_gene": cfg.paths.gene_to_gene,
            "gene_to_trait": cfg.paths.gene_to_trait,
            "trait_to_trait": trait_to_trait_path,
        },
        gene_mapping=mappings["gene_to_idx"],
        trait_mapping=mappings["trait_to_idx"],
    )
    if _use_inductive_graph_train:
        train_gene_indices = {
            mappings["gene_to_idx"][g]
            for g, split_name in gene_split_map.items()
            if split_name == "train" and g in mappings["gene_to_idx"]
        }
        train_graph = build_inductive_train_graph(graph, train_gene_indices)
        print(
            "graph_mode=inductive "
            f"train_genes={len(train_gene_indices)} "
            f"full_gene_edges={graph[('gene', 'to', 'gene')].edge_index.shape[1]} "
            f"train_gene_edges={train_graph[('gene', 'to', 'gene')].edge_index.shape[1]}"
        )
    else:
        train_graph = graph
        print("graph_mode=transductive")

    disease_to_traits = build_disease_to_traits_map(disease_df, mappings["trait_to_idx"])
    all_disease_ids: List[int] = sorted(
        set(disease_df["disease_index"].tolist()) & set(disease_to_traits.keys())
    )

    print("[4/8] loading variant/protein embeddings")
    required_variants: Set[str] = set()
    active_splits = []
    if "main" in enabled_tasks:
        active_splits.append(main_split)
    if "domain" in enabled_tasks:
        active_splits.append(domain_split)
    if "func" in enabled_tasks:
        active_splits.append(func_split)
    for split in active_splits:
        for part in split:
            required_variants.update(part["variant_id"].tolist())

    variant_x_df = _normalize_index(load_embeddings(cfg.paths.variant_x, required_ids=required_variants))
    protein_x_df = _normalize_index(load_embeddings(cfg.paths.protein_x, required_ids=required_variants))
    n_main_vx, n_main_px = len(variant_x_df), len(protein_x_df)

    # Supplement with task-specific embeddings for variants not in main files
    supplementary_variant_sources = [
        ("legacy", cfg.paths.legacy_variant_x),
    ]
    supplementary_protein_sources = [
        ("legacy", cfg.paths.legacy_protein_x),
    ]
    if "func" in enabled_tasks:
        supplementary_variant_sources.append(("mvp", cfg.paths.mvp_hgvs_embeddings))
        supplementary_protein_sources.append(("mvp", cfg.paths.mvp_protein_x))
    if "domain" in enabled_tasks:
        supplementary_variant_sources.append(("domain", cfg.paths.domain_variant_x))
    for src_name, src_path in supplementary_variant_sources:
        missing = required_variants - set(variant_x_df.index)
        if not missing:
            break
        sup = _normalize_index(load_embeddings(src_path, required_ids=missing))
        if len(sup) > 0:
            variant_x_df = pd.concat([variant_x_df, sup])
            print(f"variant_x_supplement_{src_name}=+{len(sup)}")
    for src_name, src_path in supplementary_protein_sources:
        missing = required_variants - set(protein_x_df.index)
        if not missing:
            break
        sup = _normalize_index(load_embeddings(src_path, required_ids=missing))
        if len(sup) > 0:
            protein_x_df = pd.concat([protein_x_df, sup])
            print(f"protein_x_supplement_{src_name}=+{len(sup)}")

    print(f"embedding_coverage=variant_x:{n_main_vx}+{len(variant_x_df)-n_main_vx}={len(variant_x_df)} protein_x:{n_main_px}+{len(protein_x_df)-n_main_px}={len(protein_x_df)}")
    feature_store = build_feature_store(variant_x_df, protein_x_df)

    print("[5/8] preparing records/loaders")
    main_train, main_val, main_test = main_split
    domain_train, domain_val, domain_test = domain_split
    func_train, func_val, func_test = func_split
    domain_seen_labels: List[int] = []
    if "domain" in enabled_tasks:
        domain_train, domain_train_subset_stats = select_domain_train_subset(
            domain_train,
            per_label_cap=cfg.train.domain_train_per_label_cap,
            seed=cfg.split.seed,
        )
        domain_split = (domain_train, domain_val, domain_test)
        domain_seen_labels = _sorted_domain_label_ids(domain_train)
        domain_aux_stats["train_subset"] = domain_train_subset_stats
        print("domain_aux_stats=" + json.dumps(domain_aux_stats, ensure_ascii=False))
    train_seen_diseases = sorted(
        set(main_train["disease_index"].astype(int).tolist()) & set(all_disease_ids)
    )
    if not train_seen_diseases:
        raise ValueError("No train diseases with trait mappings remain after preprocessing")
    # Transductive: all diseases in CLIP bank regardless of split.
    # Novel diseases participate as negatives; their labels are never used.
    train_disease_ids = all_disease_ids
    train_seen_disease_set = set(train_seen_diseases)
    val_heldout_disease_ids: List[int] = []
    if cfg.split.protocol == "disease_holdout":
        val_diseases = set(main_val["disease_index"].astype(int).tolist()) & set(all_disease_ids)
        val_heldout_disease_ids = sorted(val_diseases - train_seen_disease_set)
    print(
        "disease_bank="
        + json.dumps(
            {
                "train_bank_size": int(len(train_disease_ids)),
                "eval_bank_size": int(len(all_disease_ids)),
                "heldout_eval_only": int(len(set(all_disease_ids) - set(train_disease_ids))),
                "val_heldout_diseases": int(len(val_heldout_disease_ids)),
            },
            ensure_ascii=False,
        )
    )

    if "func" in enabled_tasks:
        func_train = select_func_train_subset(
            func_train,
            min_valid_axes=cfg.train.func_min_valid_axes,
            per_gene_cap=cfg.train.func_train_per_gene_cap,
            seed=cfg.split.seed,
        )
        func_target_scales = compute_func_target_scales(func_train)
    else:
        func_train = func_train.iloc[0:0].copy()
        func_val = func_val.iloc[0:0].copy()
        func_test = func_test.iloc[0:0].copy()
        func_target_scales = np.ones(len(FUNC_REGRESSION_COLS), dtype=np.float32)
    print("func_target_scales=" + json.dumps(func_target_scales.tolist(), ensure_ascii=False))

    disease_freq_weights = compute_disease_inv_freq_weights(
        main_train, method=cfg.train.disease_freq_reweight,
    )
    if disease_freq_weights:
        top5 = sorted(disease_freq_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        bot5 = sorted(disease_freq_weights.items(), key=lambda x: x[1])[:5]
        print(
            f"disease_freq_reweight={cfg.train.disease_freq_reweight} "
            f"agg={cfg.train.disease_freq_weight_agg} clip={cfg.train.disease_freq_weight_clip} "
            f"n_diseases={len(disease_freq_weights)} "
            f"top5_weight={[(d, round(w, 3)) for d, w in top5]} "
            f"bot5_weight={[(d, round(w, 3)) for d, w in bot5]}"
        )

    empty_record_stats = {
        "input_rows": 0,
        "records_emitted": 0,
        "dropped_missing_variant": 0,
        "dropped_missing_gene": 0,
    }
    train_disease_id_to_col = {int(d): i for i, d in enumerate(train_disease_ids)}
    eval_disease_id_to_col = {int(d): i for i, d in enumerate(all_disease_ids)}
    eval_positive_disease_lookup = (
        build_query_positive_disease_lookup(main_df)
        if cfg.split.protocol == "disease_holdout"
        else {}
    )
    records: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    record_build_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    split_frames = {
        "train": {"main": main_train, "domain": domain_train, "func": func_train},
        "val": {"main": main_val, "domain": domain_val, "func": func_val},
        "test": {"main": main_test, "domain": domain_test, "func": func_test},
    }
    for split_name, split_tasks in split_frames.items():
        records[split_name] = {}
        record_build_stats[split_name] = {}
        split_disease_id_to_col = train_disease_id_to_col if split_name == "train" else eval_disease_id_to_col
        main_positive_lookup = (
            eval_positive_disease_lookup
            if cfg.split.protocol == "disease_holdout" and split_name != "train"
            else None
        )
        main_rows, main_stats = make_main_records(
            split_tasks["main"],
            feature_store.variant_to_idx,
            mappings["gene_to_idx"],
            disease_id_to_col=split_disease_id_to_col,
            positive_disease_lookup=main_positive_lookup,
            disease_freq_weights=disease_freq_weights if split_name == "train" else None,
            freq_weight_agg=cfg.train.disease_freq_weight_agg,
            freq_weight_clip=cfg.train.disease_freq_weight_clip,
            return_stats=True,
        )
        records[split_name]["main"] = main_rows
        record_build_stats[split_name]["main"] = main_stats

        if "domain" in enabled_tasks:
            domain_rows, domain_stats = make_domain_records(
                split_tasks["domain"],
                feature_store.variant_to_idx,
                mappings["gene_to_idx"],
                return_stats=True,
            )
        else:
            domain_rows, domain_stats = [], dict(empty_record_stats)
        records[split_name]["domain"] = domain_rows
        record_build_stats[split_name]["domain"] = domain_stats

        if "func" in enabled_tasks:
            func_rows, func_stats = make_func_records(
                split_tasks["func"],
                feature_store.variant_to_idx,
                mappings["gene_to_idx"],
                return_stats=True,
            )
        else:
            func_rows, func_stats = [], dict(empty_record_stats)
        records[split_name]["func"] = func_rows
        record_build_stats[split_name]["func"] = func_stats

    print("record_build_stats=" + json.dumps(record_build_stats, ensure_ascii=False))
    post_split_drop_issues: List[str] = []
    for split_name, split_stats in record_build_stats.items():
        for task_name, stats in split_stats.items():
            dropped_keys = [
                key for key, value in stats.items()
                if key.startswith("dropped_") and int(value) > 0
            ]
            if dropped_keys:
                summary = {key: int(stats[key]) for key in dropped_keys}
                post_split_drop_issues.append(
                    f"{split_name}.{task_name}={json.dumps(summary, ensure_ascii=False)}"
                )
    if post_split_drop_issues:
        raise ValueError(
            "Post-split record drops remain after coverage alignment: "
            + "; ".join(post_split_drop_issues)
        )
    if records["train"]["main"]:
        main_w = np.asarray([float(r.get("confidence", 1.0)) for r in records["train"]["main"]], dtype=np.float64)
        q = np.quantile(main_w, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist()
        print(
            "main_train_weight_stats="
            + json.dumps(
                {
                    "mean": float(main_w.mean()),
                    "std": float(main_w.std()),
                    "q0": float(q[0]),
                    "q50": float(q[1]),
                    "q90": float(q[2]),
                    "q99": float(q[3]),
                    "q100": float(q[4]),
                },
                ensure_ascii=False,
            )
        )
    _prepare_task_records_by_mode(records, enabled_tasks)

    vd_kl_teacher = None
    disease_hpo_map: Dict[int, List[str]] = {
        int(row.disease_index): [h.upper() for h in parse_hpo_ids(row.hpo_ids)]
        for row in disease_df[["disease_index", "hpo_ids"]].itertuples(index=False)
    }
    if cfg.train.enable_vd_kl:
        print("building_vd_kl_teacher=1")
        # Collect disease names for concept filtering (read from raw CSV since
        # load_disease_table() only keeps disease_index + hpo_ids)
        _disease_names_for_filter: Optional[List[str]] = None
        if cfg.train.vd_kl_concept_filter_level > 0:
            try:
                _raw_disease_csv = pd.read_csv(cfg.paths.disease_table, usecols=["representative_name"])
                _disease_names_for_filter = _raw_disease_csv["representative_name"].dropna().tolist()
            except Exception:
                _disease_names_for_filter = []
        # Resolve concept metadata path
        _concept_meta_path = str(
            Path(cfg.paths.mvp_disease_concept_map).resolve().parent / "concept_to_disease_mapping.json"
        )
        # Compute disease frequency from training data for prior correction
        _train_disease_freq: Optional[Dict[int, int]] = None
        if cfg.train.vd_kl_prior_correction_alpha > 0:
            from collections import Counter
            _disease_counter: Counter = Counter()
            for rec in records["train"]["main"]:
                for did in rec.get("positive_disease_ids", []):
                    _disease_counter[int(did)] += 1
            _train_disease_freq = dict(_disease_counter)
            print(f"prior_correction: alpha={cfg.train.vd_kl_prior_correction_alpha}, "
                  f"n_diseases_in_train={len(_train_disease_freq)}")
        vd_kl_teacher = build_variant_disease_kl_teacher(
            train_main_records=records["train"]["main"],
            idx_to_variant=feature_store.idx_to_variant,
            rsid_to_hgvs=rsid_to_hgvs,
            disease_ids=train_disease_ids,
            topk_indices_path=cfg.paths.mvp_topk_indices,
            topk_values_path=cfg.paths.mvp_topk_values,
            topk_metadata_path=cfg.paths.mvp_topk_metadata,
            disease_concept_map_csv=cfg.paths.mvp_disease_concept_map,
            disease_hpo_map=disease_hpo_map,
            hpo_semantic_map_json=cfg.paths.mvp_hpo_semantic_map,
            teacher_mode=cfg.train.vd_kl_teacher_mode,
            hpo_topk_per_hpo=cfg.train.vd_kl_hpo_topk_per_hpo,
            hpo_min_similarity=cfg.train.vd_kl_hpo_min_similarity,
            concept_direct_weight=cfg.train.vd_kl_concept_direct_weight,
            concept_semantic_weight=cfg.train.vd_kl_concept_semantic_weight,
            min_variant_mapped_mass=cfg.train.vd_kl_min_variant_mapped_mass,
            max_diseases_per_variant=cfg.train.vd_kl_max_diseases_per_variant,
            max_variants_per_disease=cfg.train.vd_kl_max_variants_per_disease,
            concept_filter_level=cfg.train.vd_kl_concept_filter_level,
            concept_metadata_path=_concept_meta_path,
            disease_names=_disease_names_for_filter,
            shuffle_teacher=bool(cfg.train.vd_kl_shuffle_teacher),
            shuffle_seed=args.seed if args.seed else 42,
            gene_propagation=cfg.train.vd_kl_gene_propagation,
            gene_propagation_alpha=cfg.train.vd_kl_gene_propagation_alpha,
            gene_propagation_distance_lambda=cfg.train.vd_kl_gene_propagation_distance_lambda,
            gene_propagation_max_distance=cfg.train.vd_kl_gene_propagation_max_distance,
            gene_propagation_adaptive_alpha=cfg.train.vd_kl_gene_propagation_adaptive_alpha,
            gene_propagation_exclude_d2v=cfg.train.vd_kl_gene_propagation_exclude_d2v,
            gene_propagation_entropy_threshold=cfg.train.vd_kl_gene_propagation_entropy_threshold,
            gene_propagation_position_unknown_penalty=cfg.train.vd_kl_gene_propagation_position_unknown_penalty,
            gene_propagation_confidence_scaling=cfg.train.vd_kl_gene_propagation_confidence_scaling,
            max_diseases_per_concept=cfg.train.vd_kl_max_diseases_per_concept,
            concept_specificity_weighting=cfg.train.vd_kl_concept_specificity_weighting,
            bridge_length_penalty_b=cfg.train.vd_kl_bridge_length_penalty_b,
            hpo_min_terms_for_full_weight=cfg.train.vd_kl_hpo_min_terms_for_full_weight,
            teacher_max_concepts=cfg.train.vd_kl_teacher_max_concepts,
            teacher_concept_temperature=cfg.train.vd_kl_teacher_concept_temperature,
            disease_score_concentration=cfg.train.vd_kl_disease_score_concentration,
            quality_row_weight=cfg.train.vd_kl_quality_row_weight,
            prior_correction_alpha=cfg.train.vd_kl_prior_correction_alpha,
            train_disease_freq=_train_disease_freq,
            raw_corr_mode=cfg.train.vd_kl_raw_corr_mode,
            min_concept_corr=cfg.train.vd_kl_min_concept_corr,
            disease_score_power=cfg.train.vd_kl_disease_score_power,
            concept_quality_scaling=cfg.train.vd_kl_concept_quality_scaling,
        )
        print("vd_kl_teacher_stats=" + json.dumps(vd_kl_teacher.stats, ensure_ascii=False))
        if int(vd_kl_teacher.stats.get("mapped_variants", 0.0)) <= 0:
            print("warning: vd_kl_teacher has zero mapped variants; disabling vd_kl")
            vd_kl_teacher = None
    vd_d2v_empty_stats = {
        "input_queries": 0.0,
        "mapped_queries": 0.0,
        "candidate_diseases": 0.0,
        "records_emitted": 0.0,
        "dropped_no_teacher": 0.0,
        "dropped_no_positive_pool": 0.0,
        "dropped_no_anchor_overlap": 0.0,
        "positive_pool_mean": 0.0,
        "positive_pool_median": 0.0,
        "anchor_pool_mean": 0.0,
        "anchor_pool_median": 0.0,
        "teacher_support_mean": 0.0,
        "teacher_support_median": 0.0,
    }
    for split_name in ["train", "val", "test"]:
        records[split_name]["vd_d2v"] = []
        record_build_stats[split_name]["vd_d2v"] = dict(vd_d2v_empty_stats)
    if cfg.train.enable_vd_kl and vd_kl_teacher is not None and cfg.train.vd_kl_lambda_d2v > 0:
        vd_d2v_rows, vd_d2v_stats = make_vd_d2v_records(
            train_main_records=records["train"]["main"],
            disease_ids=train_disease_ids,
            vd_kl_teacher=vd_kl_teacher,
            return_stats=True,
        )
        records["train"]["vd_d2v"] = vd_d2v_rows
        record_build_stats["train"]["vd_d2v"] = vd_d2v_stats
        print("vd_d2v_record_stats=" + json.dumps(vd_d2v_stats, ensure_ascii=False))
    for split_name, task_records in records.items():
        counts = {k: len(v) for k, v in task_records.items()}
        print(f"records_{split_name}=" + json.dumps(counts, ensure_ascii=False))

    if "func" in enabled_tasks and len(records["train"]["func"]) < cfg.train.min_train_records_func:
        raise ValueError(
            f"Insufficient FUNC train records: {len(records['train']['func'])} < {cfg.train.min_train_records_func}"
        )

    if subsample_fraction is not None and 0.0 < subsample_fraction < 1.0:
        _sub_rng = random.Random(int(cfg.split.seed))
        orig_n = len(records["train"]["main"])
        keep_n = max(1, int(orig_n * subsample_fraction))
        records["train"]["main"] = _sub_rng.sample(records["train"]["main"], keep_n)
        print(f"subsample_fraction={subsample_fraction} -> main train {orig_n} -> {keep_n}")

    loader_seed_base = int(cfg.split.seed) * 1000
    train_loaders = {
        "main": make_dataloader_for_task(
            "main",
            records["train"]["main"],
            cfg.train.batch_size_main,
            True,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 11,
        ),
        "domain": make_dataloader_for_task(
            "domain",
            records["train"]["domain"],
            cfg.train.batch_size_domain,
            True,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 13,
        ),
        "func": make_dataloader_for_task(
            "func",
            records["train"]["func"],
            cfg.train.batch_size_func,
            True,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 17,
        ),
        "vd_d2v": make_dataloader_for_task(
            "vd_d2v",
            records["train"]["vd_d2v"],
            cfg.train.vd_kl_d2v_batch_size,
            True,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 19,
        ) if records["train"]["vd_d2v"] else None,
    }
    val_loaders = {
        "main": make_dataloader_for_task(
            "main",
            records["val"]["main"],
            cfg.train.batch_size_main,
            False,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 101,
        ),
        "domain": make_dataloader_for_task(
            "domain",
            records["val"]["domain"],
            cfg.train.batch_size_domain,
            False,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 103,
        ),
        "func": make_dataloader_for_task(
            "func",
            records["val"]["func"],
            cfg.train.batch_size_func,
            False,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 107,
        ),
        "vd_d2v": None,
    }
    test_loaders = {
        "main": make_dataloader_for_task(
            "main",
            records["test"]["main"],
            cfg.train.batch_size_main,
            False,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 201,
        ),
        "domain": make_dataloader_for_task(
            "domain",
            records["test"]["domain"],
            cfg.train.batch_size_domain,
            False,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 203,
        ),
        "func": make_dataloader_for_task(
            "func",
            records["test"]["func"],
            cfg.train.batch_size_func,
            False,
            num_workers=cfg.runtime.num_workers,
            seed=loader_seed_base + 207,
        ),
        "vd_d2v": None,
    }
    print("runtime_dataloader=" + json.dumps({"num_workers": int(cfg.runtime.num_workers)}, ensure_ascii=False))
    disease_freq_buckets = build_disease_frequency_buckets(main_train)
    print(
        "disease_freq_buckets="
        + json.dumps({k: len(v) for k, v in disease_freq_buckets.items()}, ensure_ascii=False)
    )

    if "domain" in enabled_tasks:
        if num_domains > 0:
            domain_embeddings = load_domain_embedding_tensor(
                cfg.paths.domain_embeddings,
                raw_label_ids=raw_domain_labels,
                embedding_dim=cfg.model.domain_embedding_dim,
            )
        else:
            domain_embeddings = torch.empty((0, cfg.model.domain_embedding_dim), dtype=torch.float32)
    else:
        domain_embeddings = torch.empty((0, cfg.model.domain_embedding_dim), dtype=torch.float32)

    # 加载 gene-level concept SVD targets
    concept_targets_t: Optional[torch.Tensor] = None
    gene_to_concept_row: Optional[Dict[str, int]] = None
    concept_svd_dim_actual = 0
    if cfg.train.enable_concept_regression:
        svd_path = cfg.paths.gene_concept_svd
        svd_meta_path = cfg.paths.gene_concept_svd_metadata
        if Path(svd_path).exists() and Path(svd_meta_path).exists():
            concept_emb_np, gene_to_concept_row = load_gene_concept_targets(
                svd_path, svd_meta_path,
            )
            concept_targets_t = torch.tensor(concept_emb_np, dtype=torch.float32)
            concept_svd_dim_actual = concept_emb_np.shape[1]
            # 统计覆盖率
            covered = sum(1 for g in mappings["gene_to_idx"] if g.upper() in gene_to_concept_row)
            total_genes = len(mappings["gene_to_idx"])
            print(f"concept_regression: loaded {concept_emb_np.shape[0]} gene targets, "
                  f"dim={concept_svd_dim_actual}, "
                  f"coverage={covered}/{total_genes} ({100*covered/max(total_genes,1):.1f}%)")
        else:
            print(f"concept_regression: SVD files not found ({svd_path}), disabling")
            cfg.train.enable_concept_regression = False

    print("[6/8] init model")
    set_seed(cfg.split.seed)
    model = MultiTaskModel(
        metadata=graph.metadata(),
        gene_in_dim=cfg.model.graph_gene_in_dim,
        trait_in_dim=cfg.model.graph_trait_in_dim,
        variant_in_dim=cfg.model.variant_in_dim,
        protein_in_dim=cfg.model.protein_in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_heads=cfg.model.num_heads,
        num_graph_layers=cfg.model.num_graph_layers,
        dropout=cfg.model.dropout,
        num_domains=num_domains,
        domain_embedding_dim=cfg.model.domain_embedding_dim,
        func_conservation_dim=cfg.model.func_conservation_dim,
        func_protein_impact_dim=cfg.model.func_protein_impact_dim,
        func_integrative_dim=cfg.model.func_integrative_dim,
        func_mechanism_dim=cfg.model.func_mechanism_dim,
        modality_drop_variant=cfg.model.modality_drop_variant,
        modality_drop_protein=cfg.model.modality_drop_protein,
        modality_drop_gene=cfg.model.modality_drop_gene,
        main_temperature=cfg.train.main_temperature,
        main_logit_scale_learnable=cfg.train.main_logit_scale_learnable,
        main_logit_scale_init=cfg.train.main_logit_scale_init,
        trait_dropout=cfg.model.trait_dropout,
        disease_size_embed=cfg.model.disease_size_embed,
        fusion_type=cfg.model.fusion_type,
        disease_encoder_type=disease_encoder_type,
        num_diseases=len(all_disease_ids) if disease_encoder_type == "disease_id" else 0,
        graph_mode=cfg.model.graph_mode,
        residual_alpha_max=cfg.model.residual_alpha_max,
        concept_svd_dim=concept_svd_dim_actual if cfg.train.enable_concept_regression else 0,
    ).to(device)
    print(
        f"fusion_type={cfg.model.fusion_type} "
        f"disease_encoder: type={disease_encoder_type} "
        f"trait_dropout={cfg.model.trait_dropout} "
        f"size_embed={cfg.model.disease_size_embed} "
        f"freq_reweight={cfg.train.disease_freq_reweight}"
    )

    variant_x_t = feature_store.variant_x
    protein_x_t = feature_store.protein_x
    train_result: Dict[str, object]

    _eval_only_ckpt = args.eval_only
    if _eval_only_ckpt:
        cfg.train.epochs = 0  # skip training loop, just build infrastructure

    print("[7/8] training")
    configure_trainable_modules(
        model,
        graph_train_mode=cfg.train.graph_train_mode,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=cfg.train.lr,
        lr_graph=cfg.train.lr_graph,
        weight_decay=cfg.train.weight_decay,
        graph_train_mode=cfg.train.graph_train_mode,
        lr_disease_encoder=cfg.train.lr_disease_encoder,
        logit_scale_lr_mult=cfg.train.main_logit_scale_lr_mult,
        scheduler_t0=cfg.train.scheduler_t0,
        scheduler_t_mult=cfg.train.scheduler_t_mult,
        scheduler_eta_min=cfg.train.scheduler_eta_min,
    )
    train_result = train_multitask(
        model=model,
        graph=train_graph,
        eval_graph=graph,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        variant_x=variant_x_t,
        protein_x=protein_x_t,
        domain_embeddings=domain_embeddings,
        train_disease_ids=train_disease_ids,
        disease_to_traits=disease_to_traits,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_weights={
            "main": cfg.loss_weights.main,
            "domain": cfg.loss_weights.domain,
            "func_conservation": cfg.loss_weights.func * 0.30,
            "func_protein_impact": cfg.loss_weights.func * 0.30,
            "func_integrative": cfg.loss_weights.func * 0.25,
            "func_mechanism": cfg.loss_weights.func * 0.15,
            "concept": cfg.loss_weights.concept,
        },
        epochs=cfg.train.epochs,
        grad_clip_norm=cfg.train.grad_clip_norm,
        early_stopping_patience=cfg.train.early_stopping_patience,
        main_temperature=cfg.train.main_temperature,
        main_logit_scale_learnable=cfg.train.main_logit_scale_learnable,
        main_logit_scale_min=cfg.train.main_logit_scale_min,
        main_logit_scale_max=cfg.train.main_logit_scale_max,
        domain_temperature=cfg.train.domain_temperature,
        domain_contrastive_negatives=cfg.train.domain_contrastive_negatives,
        domain_seen_labels=domain_seen_labels,
        main_loss_type=cfg.train.main_loss_type,
        aux_update_hgt=cfg.train.aux_update_hgt,
        aux_domain_interval=cfg.train.aux_domain_interval,
        aux_func_interval=cfg.train.aux_func_interval,
        main_only_warmup_epochs=cfg.train.main_only_warmup_epochs,
        func_regression_loss_type=cfg.train.func_regression_loss_type,
        func_regression_smooth_l1_beta=cfg.train.func_regression_smooth_l1_beta,
        func_mechanism_pos_weight=cfg.train.func_mechanism_pos_weight,
        gate_entropy_weight_start=cfg.train.gate_entropy_weight_start,
        gate_entropy_weight_end=cfg.train.gate_entropy_weight_end,
        func_column_scales=torch.tensor(func_target_scales, dtype=torch.float32),
        early_stop_metric=cfg.train.main_early_stop_metric,
        checkpoint_name="best_model.pt",
        disease_freq_buckets=disease_freq_buckets,
        gene_size_buckets=gene_size_buckets,
        device=device,
        output_dir=str(out_dir),
        eval_disease_ids=all_disease_ids,
        val_heldout_disease_ids=val_heldout_disease_ids if val_heldout_disease_ids else None,
        vd_kl_teacher=vd_kl_teacher,
        enable_vd_kl=(cfg.train.enable_vd_kl and vd_kl_teacher is not None),
        vd_kl_loss_type=cfg.train.vd_kl_loss_type,
        vd_kl_lambda_v2d=cfg.train.vd_kl_lambda_v2d,
        vd_kl_lambda_d2v=cfg.train.vd_kl_lambda_d2v,
        vd_kl_lambda_v2d_start=cfg.train.vd_kl_lambda_v2d_start,
        vd_kl_lambda_d2v_start=cfg.train.vd_kl_lambda_d2v_start,
        vd_kl_lambda_ramp_epochs=cfg.train.vd_kl_lambda_ramp_epochs,
        vd_kl_temperature=cfg.train.vd_kl_temperature,
        vd_kl_warmup_epochs=cfg.train.vd_kl_warmup_epochs,
        vd_kl_d2v_start_epoch=cfg.train.vd_kl_d2v_start_epoch,
        vd_kl_cache_refresh_interval=cfg.train.vd_kl_cache_refresh_interval,
        vd_kl_min_variant_mapped_mass=cfg.train.vd_kl_min_variant_mapped_mass,
        vd_kl_positive_smoothing=cfg.train.vd_kl_positive_smoothing,
        vd_kl_d2v_positive_smoothing=cfg.train.vd_kl_d2v_positive_smoothing,
        vd_kl_d2v_min_anchor_mass=cfg.train.vd_kl_d2v_min_anchor_mass,
        vd_kl_d2v_min_rows_per_step=cfg.train.vd_kl_d2v_min_rows_per_step,
        vd_kl_min_teacher_top1_prob=cfg.train.vd_kl_min_teacher_top1_prob,
        vd_kl_d2v_teacher_topk=cfg.train.vd_kl_d2v_teacher_topk,
        vd_kl_d2v_random_negatives=cfg.train.vd_kl_d2v_random_negatives,
        vd_kl_d2v_max_positive_variants=cfg.train.vd_kl_d2v_max_positive_variants,
        vd_kl_adaptive_weight=cfg.train.vd_kl_adaptive_weight,
        vd_kl_adaptive_reference_scale=cfg.train.vd_kl_adaptive_reference_scale,
        vd_kl_slack_tau=cfg.train.vd_kl_slack_tau,
        label_smoothing=cfg.train.label_smoothing,
        graph_warmup_epochs=cfg.train.graph_warmup_epochs,
        graph_warmup_lr=cfg.train.lr_graph_warmup,
        graph_cache_refresh_steps=cfg.train.graph_cache_refresh_steps,
        eval_interval=cfg.train.eval_interval,
        main_hard_negative_k=cfg.train.main_hard_negative_k,
        concept_targets=concept_targets_t,
        gene_to_concept_row=gene_to_concept_row,
        idx_to_gene=mappings.get("idx_to_gene"),
        enable_concept_regression=cfg.train.enable_concept_regression,
    )

    best_gate_temperature = float(train_result.get("best_gate_temperature", 1.0))
    print(
        "train_result="
        + json.dumps(
            {
                "best_metric": train_result.get("best_metric", 0.0),
                "best_epoch": int(train_result.get("best_epoch", 0) or 0),
                "best_gate_temperature": best_gate_temperature,
            },
            ensure_ascii=False,
        )
    )
    with open(out_dir / "train_result.json", "w", encoding="utf-8") as f:
        json.dump(train_result, f, ensure_ascii=False, indent=2)

    # --- eval-only: load checkpoint, override model weights ---
    if _eval_only_ckpt:
        _ckpt_path = Path(_eval_only_ckpt)
        if not _ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {_ckpt_path}")
        print(f"[eval-only] loading {_ckpt_path}")
        _ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model_state_dict"])
        best_gate_temperature = float(_ckpt.get("best_gate_temperature", 1.0))
        print(f"  epoch={_ckpt.get('epoch', '?')} val_metric={_ckpt.get('best_metric', 0):.4f} gate_temp={best_gate_temperature:.4f}")

    print("[8/8] test eval")
    eval_graph = graph.to(device)
    variant_x_eval = feature_store.variant_x.to(device)
    protein_x_eval = feature_store.protein_x.to(device)
    domain_embeddings_eval = domain_embeddings.to(device)
    test_metrics = evaluate_all_tasks(
        model=model,
        graph=eval_graph,
        loaders=test_loaders,
        variant_x=variant_x_eval,
        protein_x=protein_x_eval,
        domain_embeddings=domain_embeddings_eval,
        disease_ids=all_disease_ids,
        disease_to_traits=disease_to_traits,
        device=device,
        domain_temperature=cfg.train.domain_temperature,
        domain_seen_labels=domain_seen_labels,
        disease_freq_buckets=disease_freq_buckets,
        gene_size_buckets=gene_size_buckets,
        collect_gate_stats=True,
        gate_temperature=best_gate_temperature,
    )

    # Gene-only baseline by masking variant/protein features.
    main_loader = test_loaders.get("main")
    if main_loader is not None and len(main_loader) > 0:
        zero_variant = torch.zeros_like(variant_x_eval)
        zero_protein = torch.zeros_like(protein_x_eval)
        with torch.no_grad():
            gene_graph_emb, trait_graph_emb = model.forward_graph(
                eval_graph.x_dict, eval_graph.edge_index_dict
            )
        main_gene_only = evaluate_main(
            model,
            main_loader,
            zero_variant,
            zero_protein,
            gene_graph_emb,
            trait_graph_emb,
            all_disease_ids,
            disease_to_traits,
            device,
            disease_freq_buckets=disease_freq_buckets,
            gene_size_buckets=gene_size_buckets,
            collect_gate_stats=False,
            gate_temperature=best_gate_temperature,
        )
        test_metrics["main_gene_only"] = main_gene_only
        full_main = test_metrics.get("main", {})
        test_metrics["main_variant_delta"] = _build_main_metric_delta(full_main, main_gene_only)

        test_diseases = set(main_test["disease_index"].astype(int).tolist())
        heldout_diseases = sorted(test_diseases - train_seen_disease_set)
        main_disease_heldout = evaluate_main(
            model,
            main_loader,
            variant_x_eval,
            protein_x_eval,
            gene_graph_emb,
            trait_graph_emb,
            all_disease_ids,
            disease_to_traits,
            device,
            collect_gate_stats=False,
            restrict_to_disease_ids=heldout_diseases,
            gene_size_buckets=gene_size_buckets,
            gate_temperature=best_gate_temperature,
        )
        main_disease_heldout["n_heldout_diseases"] = float(len(heldout_diseases))
        main_disease_heldout["n_train_seen_diseases"] = float(len(train_seen_disease_set))
        test_metrics["main_disease_heldout"] = main_disease_heldout
    train_selection_summary = _build_train_selection_summary(train_result, test_metrics)
    test_metrics["train_selection_summary"] = train_selection_summary

    artifact_id = artifact.get("artifact_id", "")
    split_metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
    loader_counts = {
        split_name: {
            task_name: int(len(loader))
            for task_name, loader in loader_group.items()
            if loader is not None
        }
        for split_name, loader_group in {
            "train": train_loaders,
            "val": val_loaders,
            "test": test_loaders,
        }.items()
    }
    record_counts = {
        split_name: {task_name: int(len(rows)) for task_name, rows in task_rows.items()}
        for split_name, task_rows in records.items()
    }
    gene_size_bucket_counts = {k: int(len(v)) for k, v in gene_size_buckets.items()}
    feature_stats = {
        "required_variants": int(len(required_variants)),
        "main_catalog_feature_variants": int(len(main_covered_variant_ids)),
        "main_catalog_gene_ids": int(len(main_covered_gene_ids)),
        "task_feature_coverage_catalog": task_feature_coverage_catalog,
        "variant_x_loaded": int(len(variant_x_df)),
        "protein_x_loaded": int(len(protein_x_df)),
        "shared_feature_variants": int(len(feature_store.variant_to_idx)),
    }
    protocol_id = build_protocol_id(
        protocol=cfg.split.protocol,
        seed=cfg.split.seed,
        ratios=(cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio),
        artifact_id=artifact_id,
        graph_visibility=cfg.train.graph_visibility,
    )
    run_meta = {
        "protocol_id": protocol_id,
        "split_artifact_id": artifact_id,
        "split_artifact_path": str(split_artifact_path),
        "split_strategy": split_strategy_tag,
        "task_mode": task_mode,
        "task_mode_arg": task_mode_arg,
        "split_protocol": cfg.split.protocol,
        "graph_visibility": cfg.train.graph_visibility,
        "graph_train_mode": cfg.train.graph_train_mode,
        "graph_cache_refresh_steps": int(cfg.train.graph_cache_refresh_steps),
        "graph_mode": cfg.model.graph_mode,
        "split_input_signature": split_input_signature,
        "enable_vd_kl": bool(cfg.train.enable_vd_kl and vd_kl_teacher is not None),
        "vd_kl_loss_type": cfg.train.vd_kl_loss_type,
        "vd_kl_teacher_mode": cfg.train.vd_kl_teacher_mode,
        "vd_kl_lambda_v2d": float(cfg.train.vd_kl_lambda_v2d),
        "vd_kl_lambda_d2v": float(cfg.train.vd_kl_lambda_d2v),
        "vd_kl_lambda_v2d_start": float(cfg.train.vd_kl_lambda_v2d_start),
        "vd_kl_lambda_d2v_start": float(cfg.train.vd_kl_lambda_d2v_start),
        "vd_kl_lambda_ramp_epochs": int(cfg.train.vd_kl_lambda_ramp_epochs),
        "vd_kl_temperature": float(cfg.train.vd_kl_temperature),
        "vd_kl_warmup_epochs": int(cfg.train.vd_kl_warmup_epochs),
        "vd_kl_d2v_start_epoch": int(cfg.train.vd_kl_d2v_start_epoch),
        "vd_kl_d2v_batch_size": int(cfg.train.vd_kl_d2v_batch_size),
        "vd_kl_cache_refresh_interval": int(cfg.train.vd_kl_cache_refresh_interval),
        "vd_kl_hpo_topk_per_hpo": int(cfg.train.vd_kl_hpo_topk_per_hpo),
        "vd_kl_hpo_min_similarity": float(cfg.train.vd_kl_hpo_min_similarity),
        "vd_kl_concept_direct_weight": float(cfg.train.vd_kl_concept_direct_weight),
        "vd_kl_concept_semantic_weight": float(cfg.train.vd_kl_concept_semantic_weight),
        "vd_kl_min_variant_mapped_mass": float(cfg.train.vd_kl_min_variant_mapped_mass),
        "vd_kl_max_diseases_per_variant": int(cfg.train.vd_kl_max_diseases_per_variant),
        "vd_kl_max_variants_per_disease": int(cfg.train.vd_kl_max_variants_per_disease),
        "vd_kl_positive_smoothing": float(cfg.train.vd_kl_positive_smoothing),
        "vd_kl_d2v_positive_smoothing": float(cfg.train.vd_kl_d2v_positive_smoothing),
        "vd_kl_d2v_min_anchor_mass": float(cfg.train.vd_kl_d2v_min_anchor_mass),
        "vd_kl_d2v_min_rows_per_step": int(cfg.train.vd_kl_d2v_min_rows_per_step),
        "vd_kl_min_teacher_top1_prob": float(cfg.train.vd_kl_min_teacher_top1_prob),
        "vd_kl_d2v_teacher_topk": int(cfg.train.vd_kl_d2v_teacher_topk),
        "vd_kl_d2v_random_negatives": int(cfg.train.vd_kl_d2v_random_negatives),
        "vd_kl_d2v_max_positive_variants": int(cfg.train.vd_kl_d2v_max_positive_variants),
        "vd_kl_gene_propagation": bool(cfg.train.vd_kl_gene_propagation),
        "vd_kl_gene_propagation_alpha": float(cfg.train.vd_kl_gene_propagation_alpha),
        "vd_kl_gene_propagation_distance_lambda": float(cfg.train.vd_kl_gene_propagation_distance_lambda),
        "vd_kl_gene_propagation_max_distance": int(cfg.train.vd_kl_gene_propagation_max_distance),
        "vd_kl_gene_propagation_adaptive_alpha": bool(cfg.train.vd_kl_gene_propagation_adaptive_alpha),
        "vd_kl_gene_propagation_exclude_d2v": bool(cfg.train.vd_kl_gene_propagation_exclude_d2v),
        "vd_kl_gene_propagation_entropy_threshold": float(cfg.train.vd_kl_gene_propagation_entropy_threshold),
        "vd_kl_gene_propagation_position_unknown_penalty": float(cfg.train.vd_kl_gene_propagation_position_unknown_penalty),
        "vd_kl_gene_propagation_confidence_scaling": bool(cfg.train.vd_kl_gene_propagation_confidence_scaling),
        "main_hard_negative_k": int(cfg.train.main_hard_negative_k),
        "vd_kl_bridge_length_penalty_b": float(cfg.train.vd_kl_bridge_length_penalty_b),
        "vd_kl_hpo_min_terms_for_full_weight": int(cfg.train.vd_kl_hpo_min_terms_for_full_weight),
        "vd_kl_teacher_max_concepts": int(cfg.train.vd_kl_teacher_max_concepts),
        "vd_kl_teacher_concept_temperature": float(cfg.train.vd_kl_teacher_concept_temperature),
        "main_only_warmup_epochs": int(cfg.train.main_only_warmup_epochs),
        "seed": cfg.split.seed,
        "enabled_tasks": sorted(enabled_tasks),
        "main_counts": {
            "train": int(len(records["train"]["main"])),
            "val": int(len(records["val"]["main"])),
            "test": int(len(records["test"]["main"])),
        },
        "domain_counts": {
            "train": int(len(records["train"]["domain"])),
            "val": int(len(records["val"]["domain"])),
            "test": int(len(records["test"]["domain"])),
        },
        "func_counts": {
            "train": int(len(records["train"]["func"])),
            "val": int(len(records["val"]["func"])),
            "test": int(len(records["test"]["func"])),
        },
        "record_counts": record_counts,
        "task_prepare_stats": task_prepare_stats,
        "domain_aux_stats": domain_aux_stats,
        "record_build_stats": record_build_stats,
        "loader_counts": loader_counts,
        "split_summary": split_summary,
        "gene_size_bucket_counts": gene_size_bucket_counts,
        "feature_stats": feature_stats,
        "device": str(cfg.runtime.device),
        "epochs": int(cfg.train.epochs),
        "lr": float(cfg.train.lr),
        "lr_graph": float(cfg.train.lr_graph),
        "lr_disease_encoder": float(cfg.train.lr_disease_encoder),
        "weight_decay": float(cfg.train.weight_decay),
        "grad_clip_norm": float(cfg.train.grad_clip_norm),
        "batch_size_main": int(cfg.train.batch_size_main),
        "batch_size_domain": int(cfg.train.batch_size_domain),
        "batch_size_func": int(cfg.train.batch_size_func),
        "early_stopping_patience": int(cfg.train.early_stopping_patience),
        "scheduler_t0": int(cfg.train.scheduler_t0),
        "scheduler_t_mult": int(cfg.train.scheduler_t_mult),
        "scheduler_eta_min": float(cfg.train.scheduler_eta_min),
        "hidden_dim": int(cfg.model.hidden_dim),
        "out_dim": int(cfg.model.out_dim),
        "num_heads": int(cfg.model.num_heads),
        "num_graph_layers": int(cfg.model.num_graph_layers),
        "dropout": float(cfg.model.dropout),
        "residual_alpha_max": float(cfg.model.residual_alpha_max),
        "modality_drop_variant": float(cfg.model.modality_drop_variant),
        "modality_drop_protein": float(cfg.model.modality_drop_protein),
        "modality_drop_gene": float(cfg.model.modality_drop_gene),
        "trait_dropout": float(cfg.model.trait_dropout),
        "disease_size_embed": bool(cfg.model.disease_size_embed),
        "enrich_trait_graph": bool(cfg.model.enrich_trait_graph),
        "main_logit_scale_learnable": bool(cfg.train.main_logit_scale_learnable),
        "main_logit_scale_init": float(cfg.train.main_logit_scale_init),
        "main_logit_scale_min": float(cfg.train.main_logit_scale_min),
        "main_logit_scale_max": float(cfg.train.main_logit_scale_max),
        "main_logit_scale_lr_mult": float(cfg.train.main_logit_scale_lr_mult),
        "main_loss_type": cfg.train.main_loss_type,
        "label_smoothing": float(cfg.train.label_smoothing),
        "loss_weight_main": float(cfg.loss_weights.main),
        "loss_weight_domain": float(cfg.loss_weights.domain),
        "loss_weight_func": float(cfg.loss_weights.func),
        "loss_weight_concept": float(cfg.loss_weights.concept),
        "enable_concept_regression": bool(cfg.train.enable_concept_regression),
        "vd_kl_adaptive_weight": bool(cfg.train.vd_kl_adaptive_weight),
        "vd_kl_adaptive_reference_scale": float(cfg.train.vd_kl_adaptive_reference_scale),
        "vd_kl_quality_row_weight": bool(cfg.train.vd_kl_quality_row_weight),
        "vd_kl_prior_correction_alpha": float(cfg.train.vd_kl_prior_correction_alpha),
        "vd_kl_slack_tau": float(cfg.train.vd_kl_slack_tau),
        "vd_kl_raw_corr_mode": bool(cfg.train.vd_kl_raw_corr_mode),
        "vd_kl_min_concept_corr": float(cfg.train.vd_kl_min_concept_corr),
        "vd_kl_disease_score_power": float(cfg.train.vd_kl_disease_score_power),
        "vd_kl_concept_quality_scaling": bool(cfg.train.vd_kl_concept_quality_scaling),
        "vd_kl_concept_filter_level": int(cfg.train.vd_kl_concept_filter_level),
        "vd_kl_shuffle_teacher": bool(cfg.train.vd_kl_shuffle_teacher),
        "main_temperature": float(cfg.train.main_temperature),
        "domain_loss_type": cfg.train.domain_loss_type,
        "domain_contrastive_negatives": int(cfg.train.domain_contrastive_negatives),
        "domain_data_mode": cfg.train.domain_data_mode,
        "domain_train_per_label_cap": int(cfg.train.domain_train_per_label_cap),
        "main_early_stop_metric": cfg.train.main_early_stop_metric,
        "disease_freq_reweight": cfg.train.disease_freq_reweight,
        "disease_freq_weight_agg": cfg.train.disease_freq_weight_agg,
        "disease_freq_weight_clip": float(cfg.train.disease_freq_weight_clip),
        "fusion_type": cfg.model.fusion_type,
        "train_selection_summary": train_selection_summary,
    }
    if isinstance(split_metadata, dict) and split_metadata.get("gene_split_diagnostics"):
        run_meta["gene_split_diagnostics"] = split_metadata.get("gene_split_diagnostics")
    if isinstance(split_metadata, dict) and split_metadata.get("disease_split_diagnostics"):
        run_meta["disease_split_diagnostics"] = split_metadata.get("disease_split_diagnostics")
    if vd_kl_teacher is not None:
        run_meta["vd_kl_teacher_stats"] = vd_kl_teacher.stats
    test_metrics["__meta__"] = run_meta

    print("test_main=" + json.dumps(test_metrics.get("main", {}), ensure_ascii=False))
    print("test_domain=" + json.dumps(test_metrics.get("domain", {}), ensure_ascii=False))
    print("test_func=" + json.dumps(test_metrics.get("func", {}), ensure_ascii=False))
    print("test_main_gene_only=" + json.dumps(test_metrics.get("main_gene_only", {}), ensure_ascii=False))
    print("test_main_variant_delta=" + json.dumps(test_metrics.get("main_variant_delta", {}), ensure_ascii=False))
    print("test_main_disease_heldout=" + json.dumps(test_metrics.get("main_disease_heldout", {}), ensure_ascii=False))
    print("train_selection_summary=" + json.dumps(train_selection_summary, ensure_ascii=False))
    print("run_meta=" + json.dumps(run_meta, ensure_ascii=False))

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # Save index mappings for inference reproducibility
    torch.save(
        {
            "variant_to_idx": feature_store.variant_to_idx,
            "idx_to_variant": feature_store.idx_to_variant,
            "gene_to_idx": mappings["gene_to_idx"],
            "trait_to_idx": mappings["trait_to_idx"],
            "all_disease_ids": all_disease_ids,
            "disease_to_traits": disease_to_traits,
        },
        out_dir / "inference_context.pt",
    )

    if getattr(args, "export_predictions", 1) and main_loader is not None and len(main_loader) > 0:
        pred_path = out_dir / "per_example_predictions.csv"
        print(f"exporting per-example predictions → {pred_path}")
        # Reuse gene_graph_emb / trait_graph_emb already computed above for gene-only baseline
        n_pred = export_per_example_predictions(
            model=model,
            loader=main_loader,
            variant_x=variant_x_eval,
            protein_x=protein_x_eval,
            gene_graph_emb=gene_graph_emb,
            trait_graph_emb=trait_graph_emb,
            disease_ids=all_disease_ids,
            disease_to_traits=disease_to_traits,
            device=device,
            output_path=pred_path,
            disease_freq_buckets=disease_freq_buckets,
            gate_temperature=best_gate_temperature,
        )
        print(f"exported {n_pred} per-example predictions")


if __name__ == "__main__":
    main()
