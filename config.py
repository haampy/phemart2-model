from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import torch


@dataclass
class PathsConfig:
    main_labels: str = "../data/main_task/output/expanded_labels_clean.csv"
    disease_table: str = "../data/main_task/output/new_disease_to_traits.csv"
    variant_x: str = "../data/variant_data/output/main_task_variants/variant_x.csv"
    protein_x: str = "../data/variant_data/output/main_task_variants/gene_local_x_mean.csv"
    legacy_variant_x: str = "../data/variant_data/output/variant_x.csv"
    legacy_protein_x: str = "../data/variant_data/output/gene_local_x_mean.csv"
    gene_x: str = "../data/gene_data/output/gene_global_x.csv"
    trait_x: str = "../data/trait_data/output/trait_x.csv"
    gene_to_gene: str = "../data/graph_data/output/gene_to_gene.csv"
    gene_to_trait: str = "../data/graph_data/output/gene_to_trait.csv"
    trait_to_trait: str = "../data/graph_data/output/trait_to_trait.csv"
    trait_to_trait_enriched: str = "../data/graph_data/output/trait_to_trait_enriched.csv"
    domain_labels: str = "../data/domain_data/processed/domain_labels.csv"
    domain_embeddings: str = "../data/domain_data/processed/domain_embeddings.csv"
    mvp_hgvs_embeddings: str = "../data/MVP_data/embeddings/variant_x.csv"
    mvp_rsid_embeddings: str = "../data/MVP_data/embeddings/variant_x_rsid.csv"
    mvp_protein_x: str = "../data/MVP_data/embeddings/gene_local_x_mean.csv"
    mvp_topk_indices: str = "../data/MVP_data/processed/mvp_topk_indices_EUR_k256.npy"
    mvp_topk_values: str = "../data/MVP_data/processed/mvp_topk_values_EUR_k256_float16.npy"
    mvp_topk_metadata: str = "../data/MVP_data/processed/mvp_topk_metadata_EUR_k256.json"
    mvp_disease_concept_map: str = "../data/mvp_mappings/disease_to_concepts_v1_cui_sigmoid.csv"
    mvp_hpo_semantic_map: str = "../data/mvp_mappings/hpo_to_mvp_semantic.json"
    domain_variant_x: str = "../data/domain_data/processed/domain_variant_embeddings.csv"
    func_labels: str = "../data/func_impact_data/processed/func_impact_labels_v2.csv"
    gene_concept_svd: str = "../data/MVP_data/processed/gene_concept_svd64.npy"
    gene_concept_svd_metadata: str = "../data/MVP_data/processed/gene_concept_svd64_metadata.json"
    split_artifact_path: str = "artifacts/splits/default_split.json"
    output_dir: str = "experiments/minimal_multitask"


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    protocol: str = "gene_holdout"  # {"gene_holdout", "within_gene", "disease_holdout"}
    mode: str = "auto"  # {"auto", "generate", "load"}


@dataclass
class ModelConfig:
    variant_in_dim: int = 1280
    protein_in_dim: int = 1280
    graph_gene_in_dim: int = 768
    graph_trait_in_dim: int = 768
    hidden_dim: int = 256
    out_dim: int = 128
    num_heads: int = 2
    num_graph_layers: int = 2
    dropout: float = 0.3
    num_domains: int = 769
    domain_embedding_dim: int = 768
    func_conservation_dim: int = 5
    func_protein_impact_dim: int = 5
    func_integrative_dim: int = 5
    func_mechanism_dim: int = 9
    modality_drop_variant: float = 0.05
    modality_drop_protein: float = 0.30
    modality_drop_gene: float = 0.30
    trait_dropout: float = 0.15
    disease_size_embed: bool = True
    fusion_type: str = "concat_residual"  # {"gated", "concat", "concat_residual"}
    graph_mode: str = "hgt"  # {"hgt", "none"} — "none" disables HGT message passing
    enrich_trait_graph: bool = True  # use enriched trait-to-trait edges (HPO is_a + cosine kNN)
    residual_alpha_max: float = 0.4  # cap for ConcatResidualFusion residual bypass; forces >=60% through fusion MLP


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size_main: int = 128
    batch_size_domain: int = 128
    batch_size_func: int = 128
    lr: float = 1e-3
    lr_graph: float = 1e-6  # micro LR: graph fine-tunes minimally; high LR causes random-walk degradation
    lr_disease_encoder: float = 1e-4  # DiseaseEncoder (attention+MLP) has no pretrained weights; needs higher LR than graph
    lr_graph_warmup: float = 0.0  # disabled by default; if enabled, warmup period auto-uses refresh=1
    graph_warmup_epochs: int = 0  # disabled by default
    graph_train_mode: str = "weak"  # {"frozen", "weak", "full"}
    graph_cache_refresh_steps: int = 8  # post-warmup refresh interval; during warmup forced to 1
    graph_visibility: str = "transductive"  # {"inductive", "transductive"}
    weight_decay: float = 1e-2
    grad_clip_norm: float = 5.0
    eval_interval: int = 1
    early_stopping_patience: int = 20
    main_temperature: float = 0.22
    main_early_stop_metric: str = "main.ndcg@10"
    main_logit_scale_learnable: bool = True
    main_logit_scale_min: float = 1.0
    main_logit_scale_max: float = 15.0
    main_logit_scale_init: float = 0.0  # 0 = auto from 1/temperature; for main_temperature=0.22 this is ≈ 4.5
    main_logit_scale_lr_mult: float = 1.0  # logit_scale LR = main LR × this multiplier
    domain_temperature: float = 0.15
    domain_loss_type: str = "sampled_infonce"
    domain_contrastive_negatives: int = 63
    domain_data_mode: str = "full_random"
    domain_train_per_label_cap: int = 512  # <=0 disables per-label cap
    main_loss_type: str = "softmax"  # {"softmax", "bce"}
    label_smoothing: float = 0.0  # >0 enables label smoothing on main softmax loss
    aux_update_hgt: bool = False
    aux_domain_interval: int = 2
    aux_func_interval: int = 1
    func_regression_loss_type: str = "smooth_l1"  # {"smooth_l1", "mse"}
    func_regression_smooth_l1_beta: float = 1.0
    func_mechanism_pos_weight: float = 3.0
    gate_entropy_weight_start: float = 0.0
    gate_entropy_weight_end: float = 0.0
    main_only_warmup_epochs: int = 2
    func_min_valid_axes: int = 1
    func_train_per_gene_cap: int = 50
    min_train_records_func: int = 1000
    disease_freq_reweight: str = "sqrt_inv"  # {"none", "sqrt_inv", "log_inv"}
    disease_freq_weight_agg: str = "max"  # {"max", "mean"}
    disease_freq_weight_clip: float = 0.0  # <=0 means no clip
    scheduler_t0: int = 20  # CosineAnnealingWarmRestarts T_0
    scheduler_t_mult: int = 2  # CosineAnnealingWarmRestarts T_mult
    scheduler_eta_min: float = 1e-6  # CosineAnnealingWarmRestarts eta_min
    enable_vd_kl: bool = False
    vd_kl_loss_type: str = "kl"  # {"kl", "set_infonce", "weighted_set_infonce"}
    vd_kl_teacher_mode: str = "hybrid"  # {"concept_map", "hpo_semantic", "hybrid"}
    vd_kl_lambda_v2d: float = 0.6
    vd_kl_lambda_d2v: float = 0.1
    vd_kl_lambda_v2d_start: float = 0.0
    vd_kl_lambda_d2v_start: float = 0.0
    vd_kl_lambda_ramp_epochs: int = 8
    vd_kl_temperature: float = 0.15
    vd_kl_warmup_epochs: int = 3
    vd_kl_d2v_start_epoch: int = 8
    vd_kl_cache_refresh_interval: int = 1
    vd_kl_d2v_batch_size: int = 32
    vd_kl_hpo_topk_per_hpo: int = 5
    vd_kl_hpo_min_similarity: float = 0.35
    vd_kl_concept_direct_weight: float = 1.0
    vd_kl_concept_semantic_weight: float = 0.20
    vd_kl_min_variant_mapped_mass: float = 0.03  # lowered from 0.05 to pair with min_concept_corr=0.005 filtering
    vd_kl_max_diseases_per_variant: int = 32
    vd_kl_disease_score_concentration: float = 0.0  # keep disease only if score > C × (total/n_diseases); 0=disabled; 2.0=recommended for set_infonce with concept_map.
    vd_kl_max_diseases_per_concept: int = 0  # 0=unlimited; >0 limits fan-out per concept before normalization
    vd_kl_concept_specificity_weighting: bool = False  # IDF-like: down-weight concepts mapping to many diseases
    vd_kl_bridge_length_penalty_b: float = 0.0
    vd_kl_hpo_min_terms_for_full_weight: int = 0
    vd_kl_teacher_max_concepts: int = 0  # 0=use all mapped concepts; >0=only top-K by MVP probability
    vd_kl_teacher_concept_temperature: float = 1.0  # <1.0 sharpens concept probs before aggregation; 1.0=no change
    vd_kl_max_variants_per_disease: int = 256
    vd_kl_positive_smoothing: float = 0.10
    vd_kl_d2v_positive_smoothing: float = 0.20
    vd_kl_d2v_min_anchor_mass: float = 0.01
    vd_kl_d2v_min_rows_per_step: int = 4
    vd_kl_min_teacher_top1_prob: float = 0.0
    vd_kl_d2v_teacher_topk: int = 32
    vd_kl_d2v_random_negatives: int = 128
    vd_kl_d2v_max_positive_variants: int = 8
    vd_kl_adaptive_weight: bool = False  # MUST be False; adaptive scaling causes runaway KL (see CLAUDE.md)
    vd_kl_adaptive_reference_scale: float = 15.0  # reference logit_scale at which KL lambda equals its nominal value; above this, lambda scales up linearly
    vd_kl_gene_propagation: bool = False  # propagate teacher signal to same-gene variants without direct teacher
    vd_kl_gene_propagation_alpha: float = 0.3  # KL weight decay for propagated variants (0-1)
    vd_kl_gene_propagation_distance_lambda: float = 0.05  # distance decay for position-weighted propagation; 0 = uniform
    vd_kl_gene_propagation_max_distance: int = 0  # max aa distance for propagation sources; 0 = unlimited
    vd_kl_gene_propagation_adaptive_alpha: bool = False  # alpha decays with distance to nearest source
    vd_kl_gene_propagation_exclude_d2v: bool = True  # exclude propagated variants from D2V reverse teacher
    vd_kl_gene_propagation_entropy_threshold: float = 0.95  # filter propagated variants with normalized entropy > threshold; 0 = disabled
    vd_kl_gene_propagation_position_unknown_penalty: float = 0.3  # weight multiplier when protein position cannot be parsed
    vd_kl_gene_propagation_confidence_scaling: bool = False  # scale alpha by source-count confidence: 1-exp(-0.5*n)
    vd_kl_quality_row_weight: bool = False  # use mapped_mass as row weight for direct-mapped variants (instead of uniform 1.0)
    vd_kl_prior_correction_alpha: float = 0.0  # prior-correct disease scores: q'_d ∝ q_d / π(d)^α; 0=disabled; >0 boosts rare diseases
    vd_kl_slack_tau: float = 0.5  # target mass-on-support for slack_constraint loss; only used when loss_type=slack_constraint
    vd_kl_concept_filter_level: int = 0  # teacher concept filter: 0=none, 1=remove PheCodes+exact disease names, 2=L1+fuzzy 70%
    vd_kl_shuffle_teacher: bool = False  # shuffle teacher concept indices (negative control)
    vd_kl_raw_corr_mode: bool = False  # raw correlation pipeline: skip concept temp/renorm, use |corr| threshold, auto quality_row_weight
    vd_kl_min_concept_corr: float = 0.005  # minimum correlation to include a concept (only in raw_corr_mode); 0.005 ≈ 4σ with 600K samples
    vd_kl_disease_score_power: float = 1.0  # power transform on disease_scores before normalization; >1 sharpens (e.g. 2.0)
    vd_kl_concept_quality_scaling: bool = True  # multiply per-concept normalized weights by max(original_scores); prevents single-disease concepts from being inflated to w_cd=1.0 regardless of original score
    main_hard_negative_k: int = 0  # 0 = disabled; >0 = keep only top-k hardest negatives per sample in main softmax loss
    enable_concept_regression: bool = False  # gene-level concept profile regression
    concept_svd_dim: int = 64  # SVD 降维维度


@dataclass
class LossWeights:
    main: float = 1.0
    domain: float = 0.2
    func: float = 0.05
    concept: float = 0.1


@dataclass
class RuntimeConfig:
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    num_workers: int = field(default_factory=lambda: 2 if torch.cuda.is_available() else 0)


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def default_config() -> Config:
    return Config()


def ensure_output_dir(cfg: Config) -> Path:
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
