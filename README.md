# PheMART2

Missense variant-disease association retrieval via CLIP-style contrastive learning with heterogeneous graph neural networks and clinical prior distillation.

## Overview

PheMART2 frames variant-disease association as a **retrieval problem**: given a missense variant, rank all 1,563 diseases by association likelihood. Diseases are defined compositionally as sets of HPO (Human Phenotype Ontology) terms, enabling zero-shot generalization to unseen diseases.

### Architecture

```
Variant (ESM-2 DeltaP 1280d) ──┐
Protein (ESM-2 seq mean 1280d) ─┼── ConcatResidualFusion ── CLIP projection ── contrastive loss
Gene (HGT graph emb 256d) ─────┘                                    ↕
                                                        Disease (HPO attention pooling)
```

**Encoders:**
- **VariantEncoder**: ESM-2 DeltaP embeddings (mutant - wildtype) → 128d
- **ProteinEncoder**: ESM-2 mean-pooled sequence embeddings → 128d
- **GraphEncoder**: HGT over heterogeneous graph (gene-gene PPI, gene-trait HPO, trait-trait similarity) → 128d per node
- **DiseaseEncoder**: Attention pooling over HPO trait graph embeddings

**Fusion:** `z = (1-α) * MLP([variant, protein, gene]) + α * variant_raw` where α is learned (capped by `residual_alpha_max`).

### Tasks

| Task | Description | Loss | Default Weight |
|------|-------------|------|---------------|
| `main` | Variant-disease retrieval | CLIP softmax contrastive | 1.0 |
| `domain` | Pfam domain prediction | Sampled InfoNCE | 0.2 |
| `func` | Functional characterization (4 axes: conservation, protein impact, integrative, mechanism) | Smooth L1 + BCE | 0.05 |
| `VD-KL` (optional) | Clinical prior distillation from MVP PheWAS | KL divergence | λ_v2d=0.6 |

**task_mode** options: `main_only`, `main_domain`, `main_func`, `main_domain_func` (alias: `full`)

## Quick Start

```bash
# Baseline (main task only)
python run.py --task-mode main_only --epochs 50 --seed 42 \
  --split-protocol gene_holdout --output-dir experiments/baseline/seed_42

# Full model with KL distillation (best known config)
python run.py --task-mode main_domain_func --epochs 50 --seed 42 \
  --split-protocol gene_holdout \
  --enable-vd-kl 1 --vd-kl-teacher-mode hybrid \
  --vd-kl-lambda-v2d 0.6 --vd-kl-lambda-d2v 0.1 \
  --vd-kl-temperature 0.15 --main-logit-scale-max 15 \
  --vd-kl-adaptive-weight 0 \
  --output-dir experiments/full_kl/seed_42
```

## Output Files

| File | Content |
|------|---------|
| `best_model.pt` | Best checkpoint (by early stop metric) |
| `train_result.json` | Full training history, per-epoch metrics, residual_alpha |
| `test_metrics.json` | Final test set evaluation |
| `run_config.json` | Exact config used for reproducibility |

## Data Layout

All paths are relative and configured in `PathsConfig` (see `config.py`). Default assumes:

```
../data/
├── main_task/output/         # expanded_labels_clean.csv, new_disease_to_traits.csv
├── variant_data/output/      # variant_x.csv (~20GB), gene_local_x_mean.csv
├── gene_data/output/         # gene_global_x.csv (SapBERT gene descriptions)
├── trait_data/output/        # trait_x.csv (SapBERT HPO term embeddings)
├── graph_data/output/        # gene_to_gene.csv, gene_to_trait.csv, trait_to_trait*.csv
├── domain_data/processed/    # domain_labels.csv, domain_embeddings.csv
├── func_impact_data/processed/ # func_impact_labels_v2.csv
├── MVP_data/                 # embeddings/ + processed/ (for KL teacher)
└── mvp_mappings/             # disease_to_concepts.csv, hpo_to_mvp_semantic.json
```

Override any path via CLI: `--main-labels /path/to/labels.csv`

## Hyperparameter Reference

### Training

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Learning rate | `--lr` | 1e-3 | CosineAnnealingWarmRestarts(T_0=20, T_mult=2, eta_min=1e-6) |
| Graph LR | `--lr-graph` | 1e-4 | Actual = min(lr_graph, lr*0.1) when graph_train_mode=weak |
| Graph warmup LR | `--lr-graph-warmup` | 1e-3 | First `graph_warmup_epochs` epochs |
| Graph warmup epochs | `--graph-warmup-epochs` | 10 | |
| Weight decay | `--weight-decay` | 1e-2 | |
| Gradient clip | `--grad-clip-norm` | 5.0 | |
| Batch size (main) | `--batch-size-main` | 128 | |
| Batch size (domain) | `--batch-size-domain` | 128 | |
| Batch size (func) | `--batch-size-func` | 128 | |
| Early stop patience | `--early-stopping-patience` | 20 | |
| Early stop metric | `--main-early-stop-metric` | main.gene_macro_ndcg@10 | |
| Epochs | `--epochs` | 50 | |

### Main Task (CLIP)

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Temperature | `--main-temperature` | 0.22 | Controls softmax sharpness |
| Logit scale learnable | `--main-logit-scale-learnable` | 1 | |
| Logit scale init | `--main-logit-scale-init` | 0 | 0 = auto (1/temperature ≈ 4.5) |
| Logit scale max | `--main-logit-scale-max` | 30 | **Use 15 when KL is enabled** |
| Logit scale min | `--main-logit-scale-min` | 1.0 | |
| Loss type | `--main-loss-type` | softmax | {softmax, bce} |
| Label smoothing | `--label-smoothing` | 0.0 | |
| Disease freq reweight | `--disease-freq-reweight` | sqrt_inv | {none, sqrt_inv, log_inv} |

### Model Architecture

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Hidden dim | (config only) | 256 | |
| Output dim | (config only) | 128 | CLIP embedding dimension |
| Graph layers | `--num-graph-layers` | 2 | HGT layers |
| Fusion type | `--fusion-type` | concat_residual | {gated, concat, concat_residual} |
| Residual alpha max | `--residual-alpha-max` | 0.4 | Caps variant bypass; 0.4 = at most 40% pure variant |
| Graph mode | `--graph-mode` | hgt | {hgt, none} |
| Modality drop (variant) | `--modality-drop-variant` | 0.05 | Training-time dropout per modality |
| Modality drop (protein) | `--modality-drop-protein` | 0.30 | |
| Modality drop (gene) | `--modality-drop-gene` | 0.30 | |
| Trait dropout | `--trait-dropout` | 0.15 | Drops HPO traits in disease encoding |
| Disease encoder | `--disease-encoder-type` | hpo_attention | {hpo_attention, disease_id} |

### Auxiliary Tasks

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Domain weight | `--loss-weight-domain` | 0.2 | |
| Func weight | `--loss-weight-func` | 0.05 | |
| Domain temperature | `--domain-temperature` | 0.15 | |
| Domain negatives | `--domain-contrastive-negatives` | 63 | |
| Aux warmup | `--main-only-warmup-epochs` | 2 | Epochs before aux tasks activate |
| Domain interval | `--aux-domain-interval` | 2 | Run domain every N epochs |
| Func interval | `--aux-func-interval` | 1 | |

### VD-KL Distillation

Enable with `--enable-vd-kl 1`. The teacher is built from MVP PheWAS co-occurrence data.

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Teacher mode | `--vd-kl-teacher-mode` | hybrid | {concept_map, hpo_semantic, hybrid} |
| Lambda v2d | `--vd-kl-lambda-v2d` | 0.6 | Variant→disease KL weight. **Don't exceed 0.8** |
| Lambda d2v | `--vd-kl-lambda-d2v` | 0.1 | Disease→variant KL weight |
| Temperature | `--vd-kl-temperature` | 0.15 | Lower = sharper teacher. **Interacts with logit_scale_max** |
| Warmup epochs | `--vd-kl-warmup-epochs` | 2 | |
| Lambda ramp epochs | `--vd-kl-lambda-ramp-epochs` | 6 | Linear ramp from 0 to target lambda |
| D2V start epoch | `--vd-kl-d2v-start-epoch` | 4 | |
| Adaptive weight | `--vd-kl-adaptive-weight` | 0 | **Keep off.** Causes KL signal decay loop |
| Cache refresh | `--vd-kl-cache-refresh-interval` | 1 | |
| Positive smoothing | `--vd-kl-positive-smoothing` | 0.10 | |

### Split Configuration

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Protocol | `--split-protocol` | gene_holdout | {gene_holdout, within_gene, disease_holdout} |
| Mode | `--split-mode` | auto | {auto, generate, load} |
| Seed | `--seed` | 42 | Used for both split and training |
| Split artifact | `--split-artifact-path` | artifacts/splits/default_split.json | For reproducible splits |

## Known Parameter Interactions

1. **LR ↔ grad_clip_norm**: At lr=1e-3, grad_clip=5.0 works. If increasing LR, may need to increase clip.
2. **vd_kl_temperature ↔ main_logit_scale_max**: Both control softmax sharpness. Don't lower both simultaneously (t=0.05 + s=15 over-suppresses main task).
3. **KL schedule ↔ cosine restart**: KL warmup/ramp should complete within the first cosine cycle (epoch 0-20 at default T_0=20). If changing T_0, adjust KL schedule.
4. **early_stopping_patience > warmup epochs**: Must be larger than both `main_only_warmup_epochs` and `vd_kl_warmup_epochs`, otherwise model gets killed before aux tasks activate.
5. **main_logit_scale_max**: Use 15 with KL enabled, 30 without. Higher scale with KL drowns the KL signal.

## Current Best Results (gene_holdout, 5 seeds)

| Config | MRR | nDCG@10 | AUPRC | R@10 |
|--------|-----|---------|-------|------|
| M1 (main_only) | 0.230 | 0.222 | 0.194 | 0.375 |
| M4 (main_domain_func + KL) | **0.272** | **0.239** | **0.207** | **0.398** |

## File Descriptions

| File | Lines | Description |
|------|-------|-------------|
| `config.py` | ~190 | All configuration dataclasses |
| `data.py` | ~3400 | Data loading, splitting, DataLoader construction, KL teacher building |
| `model.py` | ~700 | VariantEncoder, ProteinEncoder, GraphEncoder, DiseaseEncoder, Fusion, MultiTaskModel |
| `losses.py` | ~400 | CLIP loss, domain InfoNCE, func multi-axis loss, VD-KL loss |
| `train.py` | ~1500 | Training loop, optimizer setup, scheduling, logging |
| `eval.py` | ~900 | Evaluation: MRR, nDCG, AUROC, AUPRC, R@k, bucket-level metrics |
| `run.py` | ~2500 | CLI argument parsing, data pipeline orchestration, entry point |
