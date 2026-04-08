from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import math
import random

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from data import VariantDiseaseKLTeacher
from eval import evaluate_domain, evaluate_func, evaluate_main
from data import FUNC_AXIS_SLICES
from losses import (
    domain_sampled_infonce_loss,
    func_multiaxis_loss,
    main_multi_positive_bce_loss,
    main_multi_positive_softmax_loss,
    sparse_teacher_kl_from_log_probs,
    total_loss,
)
from model import MultiTaskModel


def _amp_cuda_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def _build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _forward_graph_embeddings(
    model: MultiTaskModel,
    graph,
    device: torch.device,
    requires_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_ctx = nullcontext() if requires_grad else torch.no_grad()
    with grad_ctx:
        # HGTConv/segment_matmul path in PyG currently requires FP32 here.
        with torch.autocast(device_type=device.type, enabled=False):
            return model.forward_graph(graph.x_dict, graph.edge_index_dict)


def _compute_detached_graph_cache(
    model: MultiTaskModel,
    graph,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    was_training = model.training
    model.eval()
    gene_graph_emb, trait_graph_emb = _forward_graph_embeddings(
        model=model,
        graph=graph,
        device=device,
        requires_grad=False,
    )
    if was_training:
        model.train()
    return gene_graph_emb, trait_graph_emb


def _refresh_vd_kl_variant_cache(
    model: MultiTaskModel,
    graph,
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    pool_variant_idx: torch.Tensor,
    pool_gene_idx: torch.Tensor,
    gate_temperature: float,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Encode teacher-pool variants into CLIP variant space once per cache refresh."""
    if pool_variant_idx.numel() == 0:
        return torch.empty((0, model.clip_variant_proj[-1].out_features), device=variant_x.device)
    was_training = model.training
    model.eval()
    use_amp = variant_x.device.type == "cuda"
    amp_dtype = _amp_cuda_dtype() if use_amp else torch.bfloat16
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        # HGTConv/segment_matmul path in PyG currently requires FP32 here.
        with torch.autocast(device_type=variant_x.device.type, enabled=False):
            gene_graph_emb, _ = model.forward_graph(
                graph.x_dict,
                graph.edge_index_dict,
            )
        total = int(pool_variant_idx.numel())
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            v_idx = pool_variant_idx[start:end]
            g_idx = pool_gene_idx[start:end]
            with torch.autocast(device_type=variant_x.device.type, dtype=amp_dtype, enabled=use_amp):
                z_v = model.encode_variant(
                    v_idx,
                    g_idx,
                    variant_x,
                    protein_x,
                    gene_graph_emb,
                    gate_temperature=gate_temperature,
                    return_gate_weights=False,
                )
                z_v = F.normalize(model.clip_variant_proj(z_v), dim=-1)
            z_v = z_v.float()
            outs.append(z_v.detach())
    if was_training:
        model.train()
    return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), device=variant_x.device)


def _blend_sparse_teacher_with_positives(
    teacher_idx: torch.Tensor,
    teacher_prob: torch.Tensor,
    positive_cols: Sequence[int],
    alpha: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blend sparse teacher with uniform mass on known positives."""
    alpha = float(alpha)
    if alpha <= 0.0:
        return teacher_idx, teacher_prob
    pos_unique = sorted({int(p) for p in positive_cols if int(p) >= 0})
    if not pos_unique:
        return teacher_idx, teacher_prob

    merged: Dict[int, float] = {}
    keep_ratio = max(0.0, min(1.0, 1.0 - alpha))
    if teacher_idx.numel() > 0:
        for idx_val, prob_val in zip(teacher_idx.tolist(), teacher_prob.tolist()):
            merged[int(idx_val)] = merged.get(int(idx_val), 0.0) + float(prob_val) * keep_ratio
    pos_mass = alpha / max(len(pos_unique), 1)
    for p in pos_unique:
        merged[int(p)] = merged.get(int(p), 0.0) + float(pos_mass)

    out_idx = torch.tensor(sorted(merged.keys()), dtype=torch.long, device=device)
    out_prob = torch.tensor(
        [merged[int(i)] for i in out_idx.tolist()],
        dtype=torch.float32,
        device=device,
    )
    denom = float(out_prob.sum().item())
    if denom > 0:
        out_prob = out_prob / denom
    return out_idx, out_prob


def _sparse_prob_mass_on_indices(
    teacher_idx: torch.Tensor,
    teacher_prob: torch.Tensor,
    selected_indices: Sequence[int],
) -> float:
    if teacher_idx.numel() == 0 or teacher_prob.numel() == 0:
        return 0.0
    selected = {int(v) for v in selected_indices if int(v) >= 0}
    if not selected:
        return 0.0
    mass = 0.0
    for idx_val, prob_val in zip(teacher_idx.tolist(), teacher_prob.tolist()):
        if int(idx_val) in selected:
            mass += float(prob_val)
    return mass


def _cap_and_sample_unique_ints(values: Sequence[int], max_items: int) -> List[int]:
    uniq = sorted({int(v) for v in values if int(v) >= 0})
    if max_items <= 0 or len(uniq) <= max_items:
        return uniq
    return sorted(random.sample(uniq, max_items))


def _sample_random_pool_positions(
    pool_size: int,
    excluded: Set[int],
    num_samples: int,
) -> List[int]:
    if pool_size <= 0 or num_samples <= 0:
        return []
    candidates = [idx for idx in range(pool_size) if idx not in excluded]
    if not candidates:
        return []
    if len(candidates) <= num_samples:
        return candidates
    return random.sample(candidates, num_samples)


def _linear_ramp_value(
    epoch: int,
    start_epoch_exclusive: int,
    start_value: float,
    end_value: float,
    ramp_epochs: int,
) -> float:
    if epoch <= int(start_epoch_exclusive):
        return float(start_value)
    if int(ramp_epochs) <= 0:
        return float(end_value)
    progress = min(
        max((float(epoch) - float(start_epoch_exclusive)) / float(max(int(ramp_epochs), 1)), 0.0),
        1.0,
    )
    return float(start_value) + (float(end_value) - float(start_value)) * float(progress)


def _cosine_restart_lr(base_lr: float, eta_min: float, t_0: int, t_mult: int, epoch: int) -> float:
    """Match CosineAnnealingWarmRestarts for one param group at an explicit epoch."""
    epoch = max(int(epoch), 0)
    if t_0 <= 0:
        raise ValueError("t_0 must be > 0")
    if t_mult <= 0:
        raise ValueError("t_mult must be > 0")

    if t_mult == 1:
        t_i = t_0
        t_cur = epoch % t_0
    else:
        t_i = t_0
        t_cur = epoch
        while t_cur >= t_i:
            t_cur -= t_i
            t_i *= t_mult

    return float(eta_min) + 0.5 * (float(base_lr) - float(eta_min)) * (
        1.0 + math.cos(math.pi * float(t_cur) / float(t_i))
    )


def _graph_group_lr_for_epoch(
    scheduler,
    base_lr: float,
    epoch: int,
) -> float:
    if scheduler is None:
        return float(base_lr)
    if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
        raise TypeError(
            "graph LR warmup currently supports CosineAnnealingWarmRestarts only"
        )
    return _cosine_restart_lr(
        base_lr=float(base_lr),
        eta_min=float(scheduler.eta_min),
        t_0=int(scheduler.T_0),
        t_mult=int(scheduler.T_mult),
        epoch=int(epoch),
    )


def evaluate_all_tasks(
    model: MultiTaskModel,
    graph,
    loaders: Dict[str, object],
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    domain_embeddings: torch.Tensor,
    disease_ids: Sequence[int],
    disease_to_traits: Dict[int, List[int]],
    device: torch.device,
    domain_temperature: float,
    domain_seen_labels: Optional[Sequence[int]] = None,
    disease_freq_buckets: Optional[Dict[str, Set[int]]] = None,
    gene_size_buckets: Optional[Dict[str, Set[int]]] = None,
    collect_gate_stats: bool = False,
    gate_temperature: float = 1.0,
    compute_heavy_metrics: bool = True,
    minimal: bool = False,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    use_amp = device.type == "cuda"
    amp_dtype = _amp_cuda_dtype() if use_amp else torch.bfloat16
    with torch.no_grad():
        # Keep graph encoder in FP32 to avoid bf16 segment_matmul dtype mismatch.
        with torch.autocast(device_type=device.type, enabled=False):
            gene_graph_emb, trait_graph_emb = model.forward_graph(
                graph.x_dict,
                graph.edge_index_dict,
            )

    out: Dict[str, Dict[str, float]] = {}
    if loaders.get("main") is not None and len(loaders["main"]) > 0:
        out["main"] = evaluate_main(
            model,
            loaders["main"],
            variant_x,
            protein_x,
            gene_graph_emb,
            trait_graph_emb,
            disease_ids,
            disease_to_traits,
            device,
            disease_freq_buckets=disease_freq_buckets,
            gene_size_buckets=gene_size_buckets,
            collect_gate_stats=collect_gate_stats,
            gate_temperature=gate_temperature,
            compute_heavy_metrics=compute_heavy_metrics,
            minimal=minimal,
        )
    if not minimal:
        if loaders.get("domain") is not None and len(loaders["domain"]) > 0:
            out["domain"] = evaluate_domain(
                model,
                loaders["domain"],
                variant_x,
                protein_x,
                gene_graph_emb,
                domain_embeddings,
                device,
                temperature=domain_temperature,
                gate_temperature=gate_temperature,
                seen_labels=domain_seen_labels,
            )
        if loaders.get("func") is not None and len(loaders["func"]) > 0:
            out["func"] = evaluate_func(
                model,
                loaders["func"],
                variant_x,
                protein_x,
                gene_graph_emb,
                device,
                gate_temperature=gate_temperature,
            )
    return out


def train_multitask(
    model: MultiTaskModel,
    graph,
    eval_graph,
    train_loaders: Dict[str, object],
    val_loaders: Dict[str, object],
    variant_x: torch.Tensor,
    protein_x: torch.Tensor,
    domain_embeddings: torch.Tensor,
    train_disease_ids: Sequence[int],
    disease_to_traits: Dict[int, List[int]],
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_weights: Dict[str, float],
    epochs: int,
    grad_clip_norm: float,
    early_stopping_patience: int,
    main_temperature: float,
    main_logit_scale_learnable: bool,
    main_logit_scale_min: float,
    main_logit_scale_max: float,
    domain_temperature: float,
    domain_contrastive_negatives: int,
    domain_seen_labels: Optional[Sequence[int]],
    main_loss_type: str,
    aux_update_hgt: bool,
    aux_domain_interval: int,
    aux_func_interval: int,
    main_only_warmup_epochs: int,
    func_regression_loss_type: str,
    func_regression_smooth_l1_beta: float,
    func_mechanism_pos_weight: float,
    gate_entropy_weight_start: float,
    gate_entropy_weight_end: float,
    func_column_scales: torch.Tensor,
    early_stop_metric: str,
    checkpoint_name: str,
    disease_freq_buckets: Optional[Dict[str, Set[int]]],
    gene_size_buckets: Optional[Dict[str, Set[int]]],
    device: torch.device,
    output_dir: str,
    eval_disease_ids: Optional[Sequence[int]] = None,
    vd_kl_teacher: Optional[VariantDiseaseKLTeacher] = None,
    enable_vd_kl: bool = False,
    vd_kl_lambda_v2d: float = 1.0,
    vd_kl_lambda_d2v: float = 0.1,
    vd_kl_lambda_v2d_start: float = 0.0,
    vd_kl_lambda_d2v_start: float = 0.0,
    vd_kl_lambda_ramp_epochs: int = 0,
    vd_kl_temperature: float = 0.15,
    vd_kl_warmup_epochs: int = 3,
    vd_kl_d2v_start_epoch: int = 3,
    vd_kl_cache_refresh_interval: int = 1,
    vd_kl_min_variant_mapped_mass: float = 0.05,
    vd_kl_positive_smoothing: float = 0.10,
    vd_kl_d2v_positive_smoothing: float = 0.20,
    vd_kl_d2v_min_anchor_mass: float = 0.01,
    vd_kl_d2v_min_rows_per_step: int = 4,
    vd_kl_min_teacher_top1_prob: float = 0.0,
    vd_kl_d2v_teacher_topk: int = 32,
    vd_kl_d2v_random_negatives: int = 128,
    vd_kl_d2v_max_positive_variants: int = 8,
    vd_kl_adaptive_weight: bool = True,
    vd_kl_adaptive_reference_scale: float = 15.0,
    label_smoothing: float = 0.0,
    graph_warmup_epochs: int = 0,
    graph_warmup_lr: float = 0.0,
    graph_cache_refresh_steps: int = 1,
    graph_param_group_index: int = 1,
    val_heldout_disease_ids: Optional[Sequence[int]] = None,
    eval_interval: int = 1,
) -> Dict[str, object]:
    main_loss_type = main_loss_type.lower()
    func_regression_loss_type = func_regression_loss_type.lower()
    train_disease_ids = list(train_disease_ids)
    if eval_disease_ids is None:
        eval_disease_ids = list(train_disease_ids)
    else:
        eval_disease_ids = list(eval_disease_ids)
    disease_id_to_col = {d: i for i, d in enumerate(train_disease_ids)}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / checkpoint_name
    aux_domain_interval = max(int(aux_domain_interval), 1)
    aux_func_interval = max(int(aux_func_interval), 1)
    domain_contrastive_negatives = max(int(domain_contrastive_negatives), 1)
    main_only_warmup_epochs = max(int(main_only_warmup_epochs), 0)
    vd_kl_lambda_ramp_epochs = max(int(vd_kl_lambda_ramp_epochs), 0)
    vd_kl_d2v_min_rows_per_step = max(int(vd_kl_d2v_min_rows_per_step), 1)
    vd_kl_d2v_positive_smoothing = max(0.0, min(1.0, float(vd_kl_d2v_positive_smoothing)))
    vd_kl_d2v_min_anchor_mass = max(0.0, min(1.0, float(vd_kl_d2v_min_anchor_mass)))
    vd_kl_min_teacher_top1_prob = max(0.0, min(1.0, float(vd_kl_min_teacher_top1_prob)))
    vd_kl_d2v_teacher_topk = max(int(vd_kl_d2v_teacher_topk), 1)
    vd_kl_d2v_random_negatives = max(int(vd_kl_d2v_random_negatives), 0)
    vd_kl_d2v_max_positive_variants = max(int(vd_kl_d2v_max_positive_variants), 1)
    graph_cache_refresh_steps = max(int(graph_cache_refresh_steps), 1)
    eval_interval = max(int(eval_interval), 1)
    func_column_scales = func_column_scales.to(device)

    metric_lower_is_better = any(
        token in early_stop_metric.lower() for token in ["loss", "mae", "rmse", "error"]
    )
    best_metric_value = float("inf") if metric_lower_is_better else float("-inf")
    best_state = None
    best_epoch = 0
    best_val_metrics: Dict[str, Dict[str, float]] = {}
    wait = 0
    history: List[Dict[str, object]] = []
    gate_temperature = 5.0
    best_gate_temperature = gate_temperature

    variant_x = variant_x.to(device)
    protein_x = protein_x.to(device)
    domain_embeddings = domain_embeddings.to(device)
    graph = graph.to(device)
    eval_graph = eval_graph.to(device)
    use_amp = device.type == "cuda"
    amp_dtype = _amp_cuda_dtype() if use_amp else torch.bfloat16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = _build_grad_scaler(enabled=True) if use_grad_scaler else None
    graph_encoder_trainable = any(p.requires_grad for p in model.graph_encoder.parameters())
    amp_name = "fp32"
    if use_amp:
        amp_name = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
    print(f"amp_mode={amp_name}")

    # Graph embedding caching strategy:
    # - frozen: compute once before training, never recompute
    # - weak/full: recompute with gradients every K optimizer steps, and
    #   rebuild a detached cache after each graph update for the interim steps
    static_train_gene_graph_emb = None
    static_train_trait_graph_emb = None
    cached_train_gene_graph_emb = None
    cached_train_trait_graph_emb = None
    steps_until_graph_refresh = 0
    if not graph_encoder_trainable:
        static_train_gene_graph_emb, static_train_trait_graph_emb = _compute_detached_graph_cache(
            model=model,
            graph=graph,
            device=device,
        )
    if graph_encoder_trainable:
        graph_cache_mode = (
            "per_step" if graph_cache_refresh_steps == 1 else f"refresh_every_{graph_cache_refresh_steps}_steps"
        )
    else:
        graph_cache_mode = "static_frozen"
    print(f"graph_cache_mode={graph_cache_mode}")

    vd_pool_variant_idx_t = None
    vd_pool_gene_idx_t = None
    vd_variant_idx_to_pool: Dict[int, int] = {}
    vd_variant_to_disease = {}
    vd_variant_mapped_mass: Dict[int, float] = {}
    vd_variant_to_disease_tensor_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    vd_cache: Optional[torch.Tensor] = None
    if enable_vd_kl and vd_kl_teacher is not None and len(vd_kl_teacher.pool_variant_idx) > 0:
        vd_pool_variant_idx_t = torch.tensor(
            vd_kl_teacher.pool_variant_idx, dtype=torch.long, device=device
        )
        vd_pool_gene_idx_t = torch.tensor(
            vd_kl_teacher.pool_gene_idx, dtype=torch.long, device=device
        )
        vd_variant_idx_to_pool = {
            int(k): int(v) for k, v in vd_kl_teacher.variant_idx_to_pool.items()
        }
        vd_variant_to_disease = vd_kl_teacher.variant_to_disease
        vd_variant_mapped_mass = {
            int(k): float(v) for k, v in vd_kl_teacher.variant_mapped_mass.items()
        }

    has_main_loader = train_loaders.get("main") is not None and len(train_loaders["main"]) > 0
    has_vd_d2v_loader = train_loaders.get("vd_d2v") is not None and len(train_loaders["vd_d2v"]) > 0
    main_warmup_active = bool(has_main_loader and main_only_warmup_epochs > 0)
    tracked_params: Dict[str, Dict[str, torch.Tensor]] = {}
    for mod_name, mod in [
        ("graph_enc", model.graph_encoder),
        ("variant_enc", model.variant_encoder),
        ("protein_enc", model.protein_encoder),
        ("fusion", model.fusion),
        ("clip_v", model.clip_variant_proj),
        ("clip_d", model.clip_disease_proj),
    ]:
        p0 = next(mod.parameters(), None)
        if p0 is None:
            continue
        tracked_params[mod_name] = {
            "param_ref": p0,
            "init": p0.detach().float().clone(),
        }

    graph_warmup_active = bool(graph_warmup_epochs > 0 and graph_warmup_lr > 0.0)
    if graph_warmup_active:
        if graph_param_group_index < 0 or graph_param_group_index >= len(optimizer.param_groups):
            raise IndexError(f"Invalid graph_param_group_index: {graph_param_group_index}")
        _graph_base_lr = float(optimizer.param_groups[graph_param_group_index]["lr"])
        graph_warmup_active = _graph_base_lr > 0.0
    else:
        _graph_base_lr = 0.0

    for epoch in range(1, epochs + 1):
        # Keep the graph param group on its own LR schedule so warmup does not
        # distort the scheduler state for the rest of the model.
        if graph_warmup_active and epoch <= graph_warmup_epochs:
            optimizer.param_groups[graph_param_group_index]["lr"] = graph_warmup_lr
        elif graph_warmup_active:
            optimizer.param_groups[graph_param_group_index]["lr"] = _graph_group_lr_for_epoch(
                scheduler=scheduler,
                base_lr=_graph_base_lr,
                epoch=epoch - graph_warmup_epochs - 1,
            )

        model.train()
        task_iters = {
            name: iter(loader)
            for name, loader in train_loaders.items()
            if loader is not None and len(loader) > 0
        }
        if has_main_loader:
            max_steps = len(train_loaders["main"])
        else:
            max_steps = max((len(loader) for loader in train_loaders.values() if loader is not None), default=0)

        epoch_loss = 0.0
        step_count = 0
        epoch_gate_entropy = 0.0
        epoch_gate_entropy_steps = 0
        epoch_vd_kl_v2d = 0.0
        epoch_vd_kl_d2v = 0.0
        epoch_vd_kl_v2d_rows = 0
        epoch_vd_kl_d2v_rows = 0
        epoch_vd_cache_refreshed = 0
        epoch_vd_d2v_candidate_diseases = 0
        epoch_vd_d2v_valid_diseases = 0
        epoch_vd_d2v_anchor_mass_sum = 0.0
        epoch_vd_d2v_anchor_mass_count = 0
        epoch_vd_d2v_skipped_steps = 0
        epoch_vd_d2v_low_row_steps = 0
        epoch_grad_norm_sum = 0.0
        epoch_grad_norm_steps = 0
        epoch_graph_refreshes = 0
        epoch_task_steps = {"main": 0, "domain": 0, "func": 0, "vd_d2v": 0}
        epoch_task_rows = {"main": 0, "domain": 0, "func": 0, "vd_d2v": 0}
        epoch_loss_term_sum: Dict[str, float] = {}
        epoch_loss_term_weighted_sum: Dict[str, float] = {}
        epoch_loss_term_count: Dict[str, int] = {}
        progress = (epoch - 1) / max(epochs - 1, 1)
        gate_entropy_weight = (
            gate_entropy_weight_start
            + (gate_entropy_weight_end - gate_entropy_weight_start) * progress
        )
        aux_tasks_active = not (main_warmup_active and epoch <= main_only_warmup_epochs)

        effective_vd_warmup = int(max(vd_kl_warmup_epochs, main_only_warmup_epochs if has_main_loader else 0, 0))
        effective_vd_d2v_start = int(max(effective_vd_warmup, vd_kl_d2v_start_epoch))
        cur_vd_lambda_v2d = 0.0
        cur_vd_lambda_d2v = 0.0
        if enable_vd_kl and vd_pool_variant_idx_t is not None:
            cur_vd_lambda_v2d = _linear_ramp_value(
                epoch=epoch,
                start_epoch_exclusive=effective_vd_warmup,
                start_value=float(vd_kl_lambda_v2d_start),
                end_value=float(vd_kl_lambda_v2d),
                ramp_epochs=vd_kl_lambda_ramp_epochs,
            )
            cur_vd_lambda_d2v = _linear_ramp_value(
                epoch=epoch,
                start_epoch_exclusive=effective_vd_d2v_start,
                start_value=float(vd_kl_lambda_d2v_start),
                end_value=float(vd_kl_lambda_d2v),
                ramp_epochs=vd_kl_lambda_ramp_epochs,
            )
        vd_kl_v2d_active = (
            enable_vd_kl
            and vd_pool_variant_idx_t is not None
            and epoch > effective_vd_warmup
            and cur_vd_lambda_v2d > 0.0
        )
        vd_kl_d2v_active = (
            enable_vd_kl
            and vd_pool_variant_idx_t is not None
            and has_vd_d2v_loader
            and epoch > effective_vd_warmup
            and epoch > effective_vd_d2v_start
            and cur_vd_lambda_d2v > 0.0
        )
        refresh_interval = max(int(vd_kl_cache_refresh_interval), 1)
        if vd_kl_v2d_active and (
            vd_cache is None or ((epoch - 1) % refresh_interval == 0)
        ):
            vd_cache = _refresh_vd_kl_variant_cache(
                model=model,
                graph=graph,
                variant_x=variant_x,
                protein_x=protein_x,
                pool_variant_idx=vd_pool_variant_idx_t,
                pool_gene_idx=vd_pool_gene_idx_t,
                gate_temperature=gate_temperature,
            )
            epoch_vd_cache_refreshed = 1

        # During graph warmup (high LR), refresh every step to avoid stale cache.
        # After warmup, use the configured refresh interval.
        effective_graph_refresh = (
            1 if (graph_warmup_active and epoch <= graph_warmup_epochs)
            else graph_cache_refresh_steps
        )

        for step_idx in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            main_disease_emb = None

            refresh_graph_this_step = False
            if static_train_gene_graph_emb is not None and static_train_trait_graph_emb is not None:
                gene_graph_emb = static_train_gene_graph_emb
                trait_graph_emb = static_train_trait_graph_emb
            else:
                refresh_graph_this_step = (
                    cached_train_gene_graph_emb is None
                    or cached_train_trait_graph_emb is None
                    or steps_until_graph_refresh <= 0
                )
                if refresh_graph_this_step:
                    gene_graph_emb, trait_graph_emb = _forward_graph_embeddings(
                        model=model,
                        graph=graph,
                        device=device,
                        requires_grad=True,
                    )
                else:
                    gene_graph_emb = cached_train_gene_graph_emb
                    trait_graph_emb = cached_train_trait_graph_emb
            aux_gene_graph_emb = gene_graph_emb if aux_update_hgt else gene_graph_emb.detach()

            losses = {}
            gate_weight_batches: List[torch.Tensor] = []

            # Main
            if "main" in task_iters:
                try:
                    batch = next(task_iters["main"])
                except StopIteration:
                    pass
                else:
                    variant_idx_cpu = batch["variant_idx"]
                    gene_idx_cpu = batch["gene_idx"]
                    epoch_task_steps["main"] += 1
                    epoch_task_rows["main"] += int(variant_idx_cpu.shape[0])
                    variant_idx = variant_idx_cpu.to(device, non_blocking=True)
                    gene_idx = gene_idx_cpu.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        main_out = model.forward_main(
                            variant_idx,
                            gene_idx,
                            variant_x,
                            protein_x,
                            gene_graph_emb,
                            trait_graph_emb,
                            train_disease_ids,
                            disease_to_traits,
                            gate_temperature=gate_temperature,
                            return_gate_weights=(gate_entropy_weight > 0),
                        )
                    if gate_entropy_weight > 0:
                        z_v, z_d, gate_weights = main_out
                        gate_weight_batches.append(gate_weights)
                    else:
                        z_v, z_d = main_out
                    main_disease_emb = z_d
                    positives = batch.get("positive_disease_cols")
                    if positives is None:
                        positives = [
                            [disease_id_to_col[d] for d in p if d in disease_id_to_col]
                            for p in batch["positive_disease_ids"]
                        ]
                    sample_weights = batch["confidence"].to(device, non_blocking=True)
                    main_logit_scale = model.get_main_logit_scale(
                        min_scale=main_logit_scale_min,
                        max_scale=main_logit_scale_max,
                    )
                    if not main_logit_scale_learnable:
                        main_logit_scale = main_logit_scale.detach()
                    if main_loss_type == "bce":
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            losses["main"] = main_multi_positive_bce_loss(
                                z_v,
                                z_d,
                                positives,
                                temperature=main_temperature,
                                logit_scale=main_logit_scale,
                                sample_weights=sample_weights,
                            )
                    else:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            losses["main"] = main_multi_positive_softmax_loss(
                                z_v,
                                z_d,
                                positives,
                                temperature=main_temperature,
                                logit_scale=main_logit_scale,
                                sample_weights=sample_weights,
                                label_smoothing=label_smoothing,
                            )

                    if vd_kl_v2d_active and vd_cache is not None:
                        batch_variant_ids = [int(v) for v in variant_idx_cpu.tolist()]
                        empty_idx = torch.empty(0, dtype=torch.long, device=device)
                        empty_prob = torch.empty(0, dtype=torch.float32, device=device)
                        teacher_indices: List[torch.Tensor] = []
                        teacher_probs: List[torch.Tensor] = []
                        for batch_row, v_idx in enumerate(batch_variant_ids):
                            item = vd_variant_to_disease.get(v_idx)
                            if item is None:
                                teacher_indices.append(empty_idx)
                                teacher_probs.append(empty_prob)
                                continue
                            mapped_mass = float(vd_variant_mapped_mass.get(v_idx, 0.0))
                            if mapped_mass < float(vd_kl_min_variant_mapped_mass):
                                teacher_indices.append(empty_idx)
                                teacher_probs.append(empty_prob)
                                continue
                            cached = vd_variant_to_disease_tensor_cache.get(v_idx)
                            if cached is None:
                                d_cols_np, d_probs_np = item
                                cached = (
                                    torch.as_tensor(d_cols_np, dtype=torch.long, device=device),
                                    torch.as_tensor(d_probs_np, dtype=torch.float32, device=device),
                                )
                                vd_variant_to_disease_tensor_cache[v_idx] = cached
                            d_cols_raw_t, d_probs_raw_t = cached
                            if d_probs_raw_t.numel() == 0:
                                teacher_indices.append(empty_idx)
                                teacher_probs.append(empty_prob)
                                continue
                            if float(d_probs_raw_t.max().item()) < vd_kl_min_teacher_top1_prob:
                                teacher_indices.append(empty_idx)
                                teacher_probs.append(empty_prob)
                                continue
                            d_cols_t, d_probs_t = d_cols_raw_t, d_probs_raw_t
                            if vd_kl_positive_smoothing > 0:
                                d_cols_t, d_probs_t = _blend_sparse_teacher_with_positives(
                                    teacher_idx=d_cols_t,
                                    teacher_prob=d_probs_t,
                                    positive_cols=positives[batch_row],
                                    alpha=float(vd_kl_positive_smoothing),
                                    device=device,
                                )
                            teacher_indices.append(d_cols_t)
                            teacher_probs.append(d_probs_t)

                        if cur_vd_lambda_v2d > 0:
                            logits_v2d = (z_v.float() @ z_d.float().t()) / max(float(vd_kl_temperature), 1e-6)
                            log_probs_v2d = F.log_softmax(logits_v2d, dim=1)
                            v2d_kl, n_rows_v2d = sparse_teacher_kl_from_log_probs(
                                log_probs_v2d,
                                teacher_indices,
                                teacher_probs,
                            )
                            if n_rows_v2d > 0:
                                effective_lambda_v2d = float(cur_vd_lambda_v2d)
                                if vd_kl_adaptive_weight:
                                    cur_scale = float(main_logit_scale.detach().item()) if main_logit_scale is not None else 1.0
                                    ref_scale = float(vd_kl_adaptive_reference_scale)
                                    if cur_scale > 0 and ref_scale > 0:
                                        adaptive_factor = max(cur_scale / ref_scale, 1.0)
                                        effective_lambda_v2d *= adaptive_factor
                                weighted_v2d = v2d_kl * effective_lambda_v2d
                                losses["vd_kl_v2d"] = weighted_v2d
                                epoch_vd_kl_v2d += float(weighted_v2d.item())
                                epoch_vd_kl_v2d_rows += int(n_rows_v2d)

            # VD d2v auxiliary
            if vd_kl_d2v_active and "vd_d2v" in task_iters:
                batch = None
                try:
                    batch = next(task_iters["vd_d2v"])
                except StopIteration:
                    vd_d2v_loader = train_loaders.get("vd_d2v")
                    if vd_d2v_loader is not None and len(vd_d2v_loader) > 0:
                        task_iters["vd_d2v"] = iter(vd_d2v_loader)
                        batch = next(task_iters["vd_d2v"])
                if batch is not None:
                    disease_ids_batch = [int(d) for d in batch["disease_id"]]
                    disease_cols_batch = [int(c) for c in batch["disease_col"]]
                    positive_pool_batch = [
                        _cap_and_sample_unique_ints(pos_list, vd_kl_d2v_max_positive_variants)
                        for pos_list in batch["positive_pool_pos"]
                    ]
                    anchor_pool_batch = [
                        _cap_and_sample_unique_ints(pos_list, vd_kl_d2v_max_positive_variants)
                        for pos_list in batch["anchor_pool_pos"]
                    ]
                    teacher_pool_batch = [
                        [int(v) for v in row]
                        for row in batch["teacher_pool_pos"]
                    ]
                    teacher_prob_batch = [
                        [float(v) for v in row]
                        for row in batch["teacher_probs"]
                    ]
                    epoch_task_steps["vd_d2v"] += 1
                    epoch_task_rows["vd_d2v"] += int(len(disease_cols_batch))
                    epoch_vd_d2v_candidate_diseases += int(len(disease_cols_batch))

                    candidate_pool_set: Set[int] = set()
                    for pos_list, teacher_idx_row in zip(positive_pool_batch, teacher_pool_batch):
                        candidate_pool_set.update(int(p) for p in pos_list)
                        candidate_pool_set.update(int(v) for v in teacher_idx_row[:vd_kl_d2v_teacher_topk])
                    candidate_pool_set.update(
                        _sample_random_pool_positions(
                            pool_size=int(vd_pool_variant_idx_t.shape[0]) if vd_pool_variant_idx_t is not None else 0,
                            excluded=candidate_pool_set,
                            num_samples=vd_kl_d2v_random_negatives,
                        )
                    )
                    if not candidate_pool_set:
                        epoch_vd_d2v_skipped_steps += 1
                    else:
                        candidate_pool = sorted(candidate_pool_set)
                        candidate_local_col = {int(pool_pos): idx for idx, pool_pos in enumerate(candidate_pool)}
                        teacher_d2v_indices: List[torch.Tensor] = []
                        teacher_d2v_probs: List[torch.Tensor] = []
                        teacher_d2v_row_weights: List[torch.Tensor] = []
                        valid_disease_ids: List[int] = []

                        for disease_id, _d_col, positive_pool_pos, anchor_pool_pos, teacher_idx_row, teacher_prob_row in zip(
                            disease_ids_batch,
                            disease_cols_batch,
                            positive_pool_batch,
                            anchor_pool_batch,
                            teacher_pool_batch,
                            teacher_prob_batch,
                        ):
                            if not positive_pool_pos or not anchor_pool_pos or not teacher_idx_row or not teacher_prob_row:
                                continue
                            positive_anchor_mass = 0.0
                            positive_anchor_set = {int(p) for p in anchor_pool_pos}
                            local_pairs: List[Tuple[int, float]] = []
                            for pool_pos, prob in zip(teacher_idx_row, teacher_prob_row):
                                pool_pos_int = int(pool_pos)
                                prob_f = float(prob)
                                if pool_pos_int in positive_anchor_set:
                                    positive_anchor_mass += prob_f
                                local_col = candidate_local_col.get(pool_pos_int)
                                if local_col is not None and prob_f > 0.0:
                                    local_pairs.append((int(local_col), prob_f))
                            if positive_anchor_mass < float(vd_kl_d2v_min_anchor_mass):
                                continue
                            if not local_pairs:
                                continue
                            candidate_mass = float(sum(prob for _, prob in local_pairs))
                            if candidate_mass <= 0.0:
                                continue
                            epoch_vd_d2v_anchor_mass_sum += float(positive_anchor_mass)
                            epoch_vd_d2v_anchor_mass_count += 1
                            local_idx_t = torch.tensor(
                                [idx for idx, _ in local_pairs],
                                dtype=torch.long,
                                device=device,
                            )
                            local_prob_t = torch.tensor(
                                [prob / candidate_mass for _, prob in local_pairs],
                                dtype=torch.float32,
                                device=device,
                            )
                            if vd_kl_d2v_positive_smoothing > 0:
                                local_positive_cols = [
                                    candidate_local_col[int(p)]
                                    for p in positive_pool_pos
                                    if int(p) in candidate_local_col
                                ]
                                local_idx_t, local_prob_t = _blend_sparse_teacher_with_positives(
                                    teacher_idx=local_idx_t,
                                    teacher_prob=local_prob_t,
                                    positive_cols=local_positive_cols,
                                    alpha=float(vd_kl_d2v_positive_smoothing),
                                    device=device,
                                )
                            teacher_d2v_indices.append(local_idx_t)
                            teacher_d2v_probs.append(local_prob_t)
                            teacher_d2v_row_weights.append(
                                torch.tensor(
                                    max((positive_anchor_mass * candidate_mass) ** 0.5, 1e-6),
                                    dtype=torch.float32,
                                    device=device,
                                )
                            )
                            valid_disease_ids.append(int(disease_id))

                        n_valid_d = int(len(valid_disease_ids))
                        if n_valid_d > 0 and vd_pool_variant_idx_t is not None and vd_pool_gene_idx_t is not None:
                            epoch_vd_d2v_valid_diseases += n_valid_d
                            row_count_scale = min(
                                float(n_valid_d) / float(vd_kl_d2v_min_rows_per_step),
                                1.0,
                            )
                            if n_valid_d < int(vd_kl_d2v_min_rows_per_step):
                                epoch_vd_d2v_low_row_steps += 1
                            candidate_pool_t = torch.tensor(candidate_pool, dtype=torch.long, device=device)
                            candidate_variant_idx = vd_pool_variant_idx_t.index_select(0, candidate_pool_t)
                            candidate_gene_idx = vd_pool_gene_idx_t.index_select(0, candidate_pool_t)
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                z_v_aux_out = model.encode_variant(
                                    candidate_variant_idx,
                                    candidate_gene_idx,
                                    variant_x,
                                    protein_x,
                                    aux_gene_graph_emb,
                                    gate_temperature=gate_temperature,
                                    return_gate_weights=(gate_entropy_weight > 0),
                                )
                                if gate_entropy_weight > 0:
                                    z_v_aux_raw, gate_weights = z_v_aux_out
                                    gate_weight_batches.append(gate_weights)
                                else:
                                    z_v_aux_raw = z_v_aux_out
                                z_v_aux = F.normalize(model.clip_variant_proj(z_v_aux_raw), dim=-1)
                                if main_disease_emb is not None:
                                    valid_cols = torch.tensor(
                                        [disease_id_to_col[d] for d in valid_disease_ids],
                                        dtype=torch.long,
                                        device=device,
                                    )
                                    z_d_aux = main_disease_emb.index_select(0, valid_cols)
                                else:
                                    z_d_aux = model.encode_disease_batch(
                                        valid_disease_ids,
                                        disease_to_traits,
                                        trait_graph_emb,
                                    )
                            logits_d2v = (z_d_aux.float() @ z_v_aux.float().t()) / max(float(vd_kl_temperature), 1e-6)
                            log_probs_d2v = F.log_softmax(logits_d2v, dim=1)
                            d2v_kl, n_rows_d2v = sparse_teacher_kl_from_log_probs(
                                log_probs_d2v,
                                teacher_d2v_indices,
                                teacher_d2v_probs,
                                row_weights=teacher_d2v_row_weights,
                            )
                            if n_rows_d2v > 0:
                                effective_lambda_d2v = float(cur_vd_lambda_d2v)
                                if vd_kl_adaptive_weight:
                                    cur_scale = float(main_logit_scale.detach().item()) if main_logit_scale is not None else 1.0
                                    ref_scale = float(vd_kl_adaptive_reference_scale)
                                    if cur_scale > 0 and ref_scale > 0:
                                        adaptive_factor = max(cur_scale / ref_scale, 1.0)
                                        effective_lambda_d2v *= adaptive_factor
                                weighted_d2v = d2v_kl * effective_lambda_d2v * row_count_scale
                                losses["vd_kl_d2v"] = weighted_d2v
                                epoch_vd_kl_d2v += float(weighted_d2v.item())
                                epoch_vd_kl_d2v_rows += int(n_rows_d2v)
                        else:
                            epoch_vd_d2v_skipped_steps += 1

            # Domain
            if aux_tasks_active and "domain" in task_iters and (step_idx % aux_domain_interval == 0):
                try:
                    batch = next(task_iters["domain"])
                except StopIteration:
                    pass
                else:
                    epoch_task_steps["domain"] += 1
                    epoch_task_rows["domain"] += int(batch["variant_idx"].shape[0])
                    variant_idx = batch["variant_idx"].to(device, non_blocking=True)
                    gene_idx = batch["gene_idx"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        domain_out = model.forward_domain(
                            variant_idx,
                            gene_idx,
                            variant_x,
                            protein_x,
                            aux_gene_graph_emb,
                            domain_embeddings,
                            temperature=domain_temperature,
                            gate_temperature=gate_temperature,
                            return_gate_weights=(gate_entropy_weight > 0),
                        )
                    if gate_entropy_weight > 0:
                        logits, gate_weights = domain_out
                        gate_weight_batches.append(gate_weights)
                    else:
                        logits = domain_out
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        losses["domain"] = domain_sampled_infonce_loss(
                            logits,
                            labels,
                            num_negatives=domain_contrastive_negatives,
                        )

            # FuncImpact
            if aux_tasks_active and "func" in task_iters and (step_idx % aux_func_interval == 0):
                try:
                    batch = next(task_iters["func"])
                except StopIteration:
                    pass
                else:
                    epoch_task_steps["func"] += 1
                    epoch_task_rows["func"] += int(batch["variant_idx"].shape[0])
                    variant_idx = batch["variant_idx"].to(device, non_blocking=True)
                    gene_idx = batch["gene_idx"].to(device, non_blocking=True)
                    reg_target = batch["regression_target"].to(device, non_blocking=True)
                    reg_mask = batch["regression_mask"].to(device, non_blocking=True)
                    mech_target = batch["mechanism_target"].to(device, non_blocking=True)
                    mech_mask = batch["mechanism_mask"].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        func_out = model.forward_func(
                            variant_idx,
                            gene_idx,
                            variant_x,
                            protein_x,
                            aux_gene_graph_emb,
                            gate_temperature=gate_temperature,
                            return_gate_weights=(gate_entropy_weight > 0),
                        )
                    if gate_entropy_weight > 0:
                        func_preds, gate_weights = func_out
                        gate_weight_batches.append(gate_weights)
                    else:
                        func_preds = func_out
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        func_losses = func_multiaxis_loss(
                            func_preds,
                            reg_target,
                            reg_mask,
                            mech_target,
                            mech_mask,
                            axis_slices=FUNC_AXIS_SLICES,
                            column_scales=func_column_scales,
                            regression_loss_type=func_regression_loss_type,
                            regression_beta=func_regression_smooth_l1_beta,
                            mechanism_pos_weight=func_mechanism_pos_weight,
                        )
                    losses.update(func_losses)

            if not losses:
                continue
            for loss_name, loss_val in losses.items():
                scalar = float(loss_val.detach().item())
                w = float(loss_weights.get(loss_name, 1.0))
                epoch_loss_term_sum[loss_name] = epoch_loss_term_sum.get(loss_name, 0.0) + scalar
                epoch_loss_term_weighted_sum[loss_name] = (
                    epoch_loss_term_weighted_sum.get(loss_name, 0.0) + (w * scalar)
                )
                epoch_loss_term_count[loss_name] = epoch_loss_term_count.get(loss_name, 0) + 1
            if gate_entropy_weight > 0 and gate_weight_batches:
                all_gate = torch.cat(gate_weight_batches, dim=0)
                gate_entropy = -(all_gate.clamp_min(1e-8) * all_gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
                losses["gate_entropy_reg"] = -gate_entropy_weight * gate_entropy
                epoch_gate_entropy += float(gate_entropy.item())
                epoch_gate_entropy_steps += 1

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = total_loss(losses, loss_weights)
            if use_grad_scaler:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = float(clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = float(clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm))
                optimizer.step()
            if refresh_graph_this_step:
                epoch_graph_refreshes += 1
                if effective_graph_refresh > 1:
                    cached_train_gene_graph_emb, cached_train_trait_graph_emb = _compute_detached_graph_cache(
                        model=model,
                        graph=graph,
                        device=device,
                    )
                    gene_graph_emb = cached_train_gene_graph_emb
                    trait_graph_emb = cached_train_trait_graph_emb
                    steps_until_graph_refresh = effective_graph_refresh - 1
                else:
                    cached_train_gene_graph_emb = None
                    cached_train_trait_graph_emb = None
                    steps_until_graph_refresh = 0
            elif graph_encoder_trainable and steps_until_graph_refresh > 0:
                steps_until_graph_refresh -= 1
            if grad_norm >= 0:
                epoch_grad_norm_sum += grad_norm
                epoch_grad_norm_steps += 1

            epoch_loss += float(loss.item())
            step_count += 1

        if scheduler is not None:
            scheduler.step()

        should_eval = (epoch % eval_interval == 0) or (epoch == epochs - 1)
        if should_eval:
            val_metrics = evaluate_all_tasks(
                model=model,
                graph=eval_graph,
                loaders=val_loaders,
                variant_x=variant_x,
                protein_x=protein_x,
                domain_embeddings=domain_embeddings,
                disease_ids=eval_disease_ids,
                disease_to_traits=disease_to_traits,
                device=device,
                domain_temperature=domain_temperature,
                domain_seen_labels=domain_seen_labels,
                disease_freq_buckets=disease_freq_buckets,
                gene_size_buckets=gene_size_buckets,
                gate_temperature=gate_temperature,
                compute_heavy_metrics=False,
                minimal=True,
            )
            if (val_heldout_disease_ids is not None
                    and len(val_heldout_disease_ids) > 0
                    and "main_heldout" in early_stop_metric):
                main_loader_val = val_loaders.get("main")
                if main_loader_val is not None and len(main_loader_val) > 0:
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                        gene_graph_emb_h, trait_graph_emb_h = model.forward_graph(
                            eval_graph.x_dict, eval_graph.edge_index_dict
                        )
                    val_heldout_metrics = evaluate_main(
                        model,
                        main_loader_val,
                        variant_x,
                        protein_x,
                        gene_graph_emb_h,
                        trait_graph_emb_h,
                        eval_disease_ids,
                        disease_to_traits,
                        device,
                        restrict_to_disease_ids=val_heldout_disease_ids,
                        gate_temperature=gate_temperature,
                        minimal=True,
                    )
                    val_metrics["main_heldout"] = val_heldout_metrics
            metric_path = [p.strip() for p in early_stop_metric.split(".") if p.strip()]
            if len(metric_path) == 2:
                metric_value = float(val_metrics.get(metric_path[0], {}).get(metric_path[1], 0.0))
            else:
                metric_value = float(val_metrics.get("main", {}).get("mrr", 0.0))
        else:
            val_metrics = {}
            metric_value = None
        avg_epoch_loss = epoch_loss / max(step_count, 1)
        avg_gate_entropy = epoch_gate_entropy / max(epoch_gate_entropy_steps, 1)
        avg_vd_kl_v2d = epoch_vd_kl_v2d / max(step_count, 1)
        avg_vd_kl_d2v = epoch_vd_kl_d2v / max(step_count, 1)
        avg_vd_d2v_anchor_mass = epoch_vd_d2v_anchor_mass_sum / max(epoch_vd_d2v_anchor_mass_count, 1)
        avg_grad_norm = epoch_grad_norm_sum / max(epoch_grad_norm_steps, 1)
        main_logit_scale_val = float(
            model.get_main_logit_scale(
                min_scale=main_logit_scale_min,
                max_scale=main_logit_scale_max,
            ).detach().cpu().item()
        )
        loss_term_avg = {
            k: float(epoch_loss_term_sum[k] / max(epoch_loss_term_count.get(k, 1), 1))
            for k in epoch_loss_term_sum
        }
        loss_term_weighted_avg = {
            k: float(epoch_loss_term_weighted_sum[k] / max(epoch_loss_term_count.get(k, 1), 1))
            for k in epoch_loss_term_weighted_sum
        }
        param_delta = {}
        for mod_name, pack in tracked_params.items():
            p_ref = pack["param_ref"]
            init_t = pack["init"]
            p_cur = p_ref.detach().float()
            delta_l2 = float((p_cur - init_t).norm().item())
            cur_l2 = float(p_cur.norm().item())
            param_delta[f"{mod_name}_delta_l2"] = delta_l2
            param_delta[f"{mod_name}_norm_l2"] = cur_l2
        residual_alpha_val = None
        if hasattr(model.fusion, "residual_alpha"):
            residual_alpha_val = float(torch.sigmoid(model.fusion.residual_alpha).detach().cpu().item())
        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "grad_norm": avg_grad_norm,
            "gate_entropy": avg_gate_entropy,
            "gate_entropy_weight": gate_entropy_weight,
            "gate_temperature": float(gate_temperature),
            "main_logit_scale": main_logit_scale_val,
            "loss_terms_avg": loss_term_avg,
            "loss_terms_weighted_avg": loss_term_weighted_avg,
            "aux_tasks_active": bool(aux_tasks_active),
            "vd_kl_lambda_v2d": float(cur_vd_lambda_v2d),
            "vd_kl_lambda_d2v": float(cur_vd_lambda_d2v),
            "vd_kl_adaptive_factor": float(max(main_logit_scale_val / float(vd_kl_adaptive_reference_scale), 1.0)) if (vd_kl_adaptive_weight and vd_kl_adaptive_reference_scale > 0) else 1.0,
            "vd_kl_v2d": avg_vd_kl_v2d,
            "vd_kl_d2v": avg_vd_kl_d2v,
            "vd_kl_v2d_rows": int(epoch_vd_kl_v2d_rows),
            "vd_kl_d2v_rows": int(epoch_vd_kl_d2v_rows),
            "vd_kl_cache_refreshed": int(epoch_vd_cache_refreshed),
            "vd_d2v_candidate_diseases": int(epoch_vd_d2v_candidate_diseases),
            "vd_d2v_valid_diseases": int(epoch_vd_d2v_valid_diseases),
            "vd_d2v_anchor_mass_mean": float(avg_vd_d2v_anchor_mass),
            "vd_d2v_anchor_mass_count": int(epoch_vd_d2v_anchor_mass_count),
            "vd_d2v_skipped_steps": int(epoch_vd_d2v_skipped_steps),
            "vd_d2v_low_row_steps": int(epoch_vd_d2v_low_row_steps),
            "graph_refreshes": int(epoch_graph_refreshes),
            "graph_cache_refresh_steps": int(graph_cache_refresh_steps),
            "task_steps": {k: int(v) for k, v in epoch_task_steps.items()},
            "task_rows": {k: int(v) for k, v in epoch_task_rows.items()},
            "val_metrics": val_metrics,
        }
        epoch_record.update(param_delta)
        if residual_alpha_val is not None:
            epoch_record["residual_alpha"] = residual_alpha_val
        history.append(epoch_record)

        residual_str = f" res_alpha={residual_alpha_val:.4f}" if residual_alpha_val is not None else ""
        grad_str = f" grad_norm={avg_grad_norm:.4f}"
        graph_str = f" graph_refreshes={epoch_graph_refreshes}"
        gate_str = ""
        if gate_entropy_weight > 0:
            gate_str = f" gate_entropy={avg_gate_entropy:.4f} gate_lambda={gate_entropy_weight:.6f}"
        vd_kl_str = ""
        if enable_vd_kl:
            vd_kl_str = (
                f" vd_lambda={cur_vd_lambda_v2d:.4f}/{cur_vd_lambda_d2v:.4f}"
                f" vd_kl_v2d={avg_vd_kl_v2d:.4f}"
                f" vd_kl_d2v={avg_vd_kl_d2v:.4f}"
                f" vd_rows={epoch_vd_kl_v2d_rows}/{epoch_vd_kl_d2v_rows}"
                f" vd_d2v_diag={epoch_vd_d2v_candidate_diseases}/{epoch_vd_d2v_valid_diseases}"
                f" vd_anchor={avg_vd_d2v_anchor_mass:.3f}"
                f" vd_skip={epoch_vd_d2v_skipped_steps}"
                f" vd_low={epoch_vd_d2v_low_row_steps}"
                f" vd_cache={epoch_vd_cache_refreshed}"
                f" vd_d2v_on={int(vd_kl_d2v_active)}"
            )
        warmup_str = ""
        if main_warmup_active:
            warmup_str = f" aux_on={int(aux_tasks_active)}"
        task_usage_str = (
            " task_steps("
            f"m={epoch_task_steps['main']},d={epoch_task_steps['domain']},"
            f"f={epoch_task_steps['func']},vd={epoch_task_steps['vd_d2v']}"
            ")"
            " task_rows("
            f"m={epoch_task_rows['main']},d={epoch_task_rows['domain']},"
            f"f={epoch_task_rows['func']},vd={epoch_task_rows['vd_d2v']}"
            ")"
        )
        loss_term_str = ""
        if loss_term_avg:
            keys_order = [
                "main",
                "domain",
                "func",
                "vd_kl_v2d",
                "vd_kl_d2v",
                "gate_entropy_reg",
            ]
            present = [k for k in keys_order if k in loss_term_avg]
            present += [k for k in sorted(loss_term_avg.keys()) if k not in set(present)]
            part = []
            for k in present:
                part.append(f"{k}:{loss_term_avg[k]:.3f}/{loss_term_weighted_avg.get(k, loss_term_avg[k]):.3f}")
            loss_term_str = " loss_terms(" + ",".join(part) + ")"
        param_str = ""
        if tracked_params:
            tracked = []
            for k in ["graph_enc_delta_l2", "variant_enc_delta_l2", "clip_v_delta_l2", "clip_d_delta_l2", "fusion_delta_l2"]:
                if k in param_delta:
                    tracked.append(f"{k}={param_delta[k]:.4f}")
            if tracked:
                param_str = " param_delta(" + ",".join(tracked) + ")"
        main_val = val_metrics.get("main", {})
        bucket_log = ""
        if main_val:
            bucket_log = (
                " "
                f"rare(mrr/r10/ndcg10)="
                f"{main_val.get('bucket_rare_mrr', 0.0):.4f}/"
                f"{main_val.get('bucket_rare_recall@10', 0.0):.4f}/"
                f"{main_val.get('bucket_rare_ndcg@10', 0.0):.4f} "
                f"med(mrr/r10/ndcg10)="
                f"{main_val.get('bucket_medium_mrr', 0.0):.4f}/"
                f"{main_val.get('bucket_medium_recall@10', 0.0):.4f}/"
                f"{main_val.get('bucket_medium_ndcg@10', 0.0):.4f} "
                f"freq(mrr/r10/ndcg10)="
                f"{main_val.get('bucket_frequent_mrr', 0.0):.4f}/"
                f"{main_val.get('bucket_frequent_recall@10', 0.0):.4f}/"
                f"{main_val.get('bucket_frequent_ndcg@10', 0.0):.4f}"
            )
        if should_eval:
            val_str = (
                f"val_metric({early_stop_metric})={metric_value:.4f} "
                f"val_mrr={main_val.get('mrr', 0.0):.4f} "
                f"val_r1={main_val.get('recall@1', 0.0):.4f} "
                f"val_r5={main_val.get('recall@5', 0.0):.4f} "
                f"val_r10={main_val.get('recall@10', 0.0):.4f} "
                f"val_ndcg10={main_val.get('ndcg@10', 0.0):.4f} "
                f"val_gene_macro_ndcg10={main_val.get('gene_macro_ndcg@10', 0.0):.4f}"
                f"{bucket_log}"
            )
        else:
            val_str = "[eval_skipped]"
        print(
            f"epoch={epoch} train_loss={avg_epoch_loss:.4f} "
            f"logit_scale={main_logit_scale_val:.4f}"
            f"{grad_str}{graph_str}"
            f"{gate_str}{vd_kl_str}{warmup_str}{residual_str}{task_usage_str} "
            f"{loss_term_str}{param_str} "
            f"{val_str}"
        )

        if should_eval and metric_value is not None:
            is_better = (
                metric_value < best_metric_value
                if metric_lower_is_better
                else metric_value > best_metric_value
            )
            if is_better:
                best_metric_value = metric_value
                best_epoch = int(epoch)
                best_val_metrics = deepcopy(val_metrics)
                best_gate_temperature = gate_temperature
                wait = 0
                best_state = deepcopy(model.state_dict())
                best_gene_graph_emb, best_trait_graph_emb = _compute_detached_graph_cache(
                    model=model,
                    graph=graph,
                    device=device,
                )
                ckpt_payload = {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "best_metric_name": early_stop_metric,
                    "best_metric": best_metric_value,
                    "best_gate_temperature": float(best_gate_temperature),
                    "gene_graph_emb": best_gene_graph_emb.detach().cpu(),
                    "trait_graph_emb": best_trait_graph_emb.detach().cpu(),
                }
                torch.save(ckpt_payload, ckpt_path)
            else:
                wait += 1

        gate_temperature = max(gate_temperature * 0.9, 0.1)

        if should_eval and wait >= early_stopping_patience:
            print(
                f"early_stop_at_epoch={epoch} "
                f"best_{early_stop_metric}={best_metric_value:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_metric": best_metric_value,
        "best_metric_name": early_stop_metric,
        "best_gate_temperature": float(best_gate_temperature),
        "best_epoch": int(best_epoch),
        "completed_epochs": int(len(history)),
        "best_val_metrics": best_val_metrics,
        "history": history,
        "checkpoint_path": str(ckpt_path),
    }
