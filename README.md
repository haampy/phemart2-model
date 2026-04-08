# PheMART2

Missense variant-disease association retrieval via CLIP-style contrastive learning, heterogeneous graph neural networks, and clinical prior distillation.

## Overview

PheMART2 frames variant-disease association as a **retrieval problem**: given a missense variant, rank all 1,563 diseases by association likelihood. Diseases are defined compositionally as sets of HPO (Human Phenotype Ontology) terms, enabling zero-shot generalization to unseen diseases.

### Current Model Status and Known Findings

**Best results so far** (gene_holdout, 5 seeds):

| Config | MRR | nDCG@10 | AUPRC | R@10 |
|--------|-----|---------|-------|------|
| M1 (main_only) | 0.230 | 0.222 | 0.195 | 0.368 |
| M2 (main_only + KL) | 0.264 | 0.254 | 0.221 | 0.420 |
| M3 (main_domain_func) | 0.245 | 0.237 | 0.209 | 0.391 |
| M4 (main_domain_func + KL) | **0.272** | **0.262** | **0.231** | **0.426** |

**What the model has learned:**

The full model M4 improves MRR by +18.1% over the M1 baseline (0.272 vs 0.230). KL distillation is the largest single contributor (M2 vs M1: +14.4% MRR, p=0.039), sourced from population-level EHR co-occurrence data (MVP PheWAS). Auxiliary tasks alone provide +6.6% MRR (M3 vs M1). A separate 2x2 factorial on (HGT graph vs no-graph) x (KL vs no-KL) in the thesis found that HGT alone slightly hurts MRR (-4.0%, ns) but amplifies the KL signal (+4.4% interaction) -- note this is from graph-mode ablation experiments, not from the M1-M4 task-mode comparison above.

**Variant-level vs gene-level signal:**

The model learns "variant-disease associations in gene context" — neither purely variant-level nor collapsed to gene-level:
- Fusion formula: `z = (1-α) * MLP([variant, protein, gene]) + α * variant_raw`. After training, α ≈ 0.35 (cap = 0.4), meaning ~35% pure variant residual + 65% tri-modal fusion MLP.
- The model pushes α toward its cap across all seeds, indicating it "wants" more pure variant signal.
- MFN2 case study confirms variant-level discrimination: different variants in the same gene produce different disease rankings.
- However, 65% of the signal passes through the fusion MLP, where protein_x and gene_graph_emb are gene-level features.

**Known limitations:**
- KL teacher covers ~10.8% of training variants, biased toward well-studied genes.
- Gains concentrate on the frequent disease bucket (63-78% of test samples depending on seed); rare/medium buckets show no significant improvement.
- EXT1 case study: when population frequency conflicts with clinical importance (common osteochondroma vs rare chondrosarcoma), the KL teacher pulls the correct ranking from #3 to #832.
- External validation (Phenopackets, 0% variant overlap): seen-gene MRR = 0.418, novel-gene MRR = 0.089.

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

## Dependencies

- Python 3.10+
- PyTorch 2.x (CUDA)
- PyTorch Geometric (torch_geometric)
- pandas, numpy, scipy, scikit-learn, tqdm
- wandb (optional, for experiment tracking)

## Quick Start

**Important:** Data paths default to `../data/` relative to the code directory. Use `--data-root` to point to your data location:

```bash
# If your data is at /path/to/data/ instead of ../data/
python run.py --data-root /path/to/data/ ...

# Baseline (main task only), ~13 min on L40s
python run.py --task-mode main_only --epochs 50 --seed 42 \
  --split-protocol gene_holdout --output-dir experiments/baseline/seed_42

# Full model with KL distillation (best known config), ~24 min on L40s
python run.py --task-mode main_domain_func --epochs 50 --seed 42 \
  --split-protocol gene_holdout \
  --enable-vd-kl 1 --vd-kl-teacher-mode hybrid \
  --vd-kl-lambda-v2d 0.6 --vd-kl-lambda-d2v 0.1 \
  --vd-kl-temperature 0.15 --main-logit-scale-max 15 \
  --vd-kl-adaptive-weight 0 \
  --output-dir experiments/full_kl/seed_42

# Quick sanity check with 10% data
python run.py --task-mode main_only --epochs 10 --seed 42 \
  --subsample-fraction 0.1 --output-dir experiments/sanity
```

**Typical wall-clock per seed (50 epochs):**

| Config | L40s | V100s |
|--------|------|-------|
| main_only | ~13 min | ~23 min |
| main_only + KL | ~17 min | ~31 min |
| main_domain_func | ~20 min | ~36 min |
| main_domain_func + KL | ~24 min | ~43 min |

## Output Files

| File | Content |
|------|---------|
| `best_model.pt` | Best checkpoint (by early stop metric) |
| `train_result.json` | Full training history, per-epoch metrics, residual_alpha |
| `test_metrics.json` | Final test set evaluation |
| `run_config.json` | Exact config used for reproducibility |

Use `--export-predictions 1` to additionally export per-example predictions CSV.

## Data Layout

All paths are relative and configured in `PathsConfig` (see `config.py`). Default assumes:

```
../data/
├── main_task/output/           # expanded_labels_clean.csv, new_disease_to_traits.csv
├── variant_data/output/        # variant_x.csv (~20GB), gene_local_x_mean.csv
├── gene_data/output/           # gene_global_x.csv (SapBERT gene descriptions)
├── trait_data/output/          # trait_x.csv (SapBERT HPO term embeddings)
├── graph_data/output/          # gene_to_gene.csv, gene_to_trait.csv, trait_to_trait*.csv
├── domain_data/processed/      # domain_labels.csv, domain_embeddings.csv
├── func_impact_data/processed/ # func_impact_labels_v2.csv
├── MVP_data/                   # embeddings/ + processed/ (for KL teacher)
└── mvp_mappings/               # disease_to_concepts.csv, hpo_to_mvp_semantic.json
```

Override any path via CLI: `--main-labels /path/to/labels.csv`

## Hyperparameter Reference

### Training

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Data root | `--data-root` | ../data/ | Override base data directory for all default paths |
| Num workers | `--num-workers` | 2 (GPU) / 0 (CPU) | DataLoader worker processes |
| Learning rate | `--lr` | 1e-3 | |
| Graph LR | `--lr-graph` | 1e-4 | Actual = min(lr_graph, lr*0.1) when graph_train_mode=weak |
| Graph warmup LR | `--lr-graph-warmup` | 1e-3 | First `graph_warmup_epochs` epochs |
| Graph warmup epochs | `--graph-warmup-epochs` | 10 | |
| Graph train mode | `--graph-train-mode` | weak | {frozen, weak, full}. "weak" uses lr_graph; "full" uses main lr |
| Weight decay | `--weight-decay` | 1e-2 | |
| Gradient clip | `--grad-clip-norm` | 5.0 | |
| Batch size (main) | `--batch-size-main` | 128 | |
| Batch size (domain) | `--batch-size-domain` | 128 | |
| Batch size (func) | `--batch-size-func` | 128 | |
| Early stop patience | `--early-stopping-patience` | 20 | |
| Early stop metric | `--main-early-stop-metric` | main.gene_macro_ndcg@10 | |
| Epochs | `--epochs` | 50 | |
| Data subsample | `--subsample-fraction` | 1.0 | < 1.0 for quick experiments |
| Scheduler T_0 | `--scheduler-t0` | 20 | CosineAnnealingWarmRestarts first cycle length |
| Scheduler T_mult | `--scheduler-t-mult` | 2 | Cycle length multiplier |
| Scheduler eta_min | `--scheduler-eta-min` | 1e-6 | Minimum LR |

**LR schedule**: CosineAnnealingWarmRestarts. Default cycles: epoch 0-20, 20-60, 60-140. If changing T_0, adjust KL schedule accordingly (see Parameter Interactions).

### Main Task (CLIP)

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Temperature | `--main-temperature` | 0.22 | Controls softmax sharpness |
| Logit scale learnable | `--main-logit-scale-learnable` | 1 | |
| Logit scale init | `--main-logit-scale-init` | 0 | 0 = auto (1/temperature ≈ 4.5) |
| Logit scale max | `--main-logit-scale-max` | 30 | **Use 15 when KL is enabled** |
| Logit scale min | `--main-logit-scale-min` | 1.0 | |
| Logit scale LR multiplier | `--main-logit-scale-lr-mult` | 1.0 | Logit scale LR = main LR × this |
| Loss type | `--main-loss-type` | softmax | {softmax, bce} |
| Label smoothing | `--label-smoothing` | 0.0 | |
| Main loss weight | `--loss-weight-main` | 1.0 | |
| Disease freq reweight | `--disease-freq-reweight` | sqrt_inv | {none, sqrt_inv, log_inv} |
| Disease freq weight agg | `--disease-freq-weight-agg` | max | {max, mean} |
| Disease freq weight clip | `--disease-freq-weight-clip` | 0.0 | <=0 means no clip |

### Model Architecture

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Hidden dim | `--hidden-dim` | 256 | |
| Output dim | `--out-dim` | 128 | CLIP embedding dimension |
| Graph layers | `--num-graph-layers` | 2 | HGT layers |
| Fusion type | `--fusion-type` | concat_residual | {gated, concat, concat_residual} |
| Residual alpha max | `--residual-alpha-max` | 0.4 | Caps variant bypass; 0.4 = at most 40% pure variant |
| Graph mode | `--graph-mode` | hgt | {hgt, none} |
| Graph visibility | `--graph-visibility` | transductive | {inductive, transductive} |
| Modality drop (variant) | `--modality-drop-variant` | 0.05 | Training-time dropout per modality |
| Modality drop (protein) | `--modality-drop-protein` | 0.30 | |
| Modality drop (gene) | `--modality-drop-gene` | 0.30 | |
| Trait dropout | `--trait-dropout` | 0.15 | Drops HPO traits in disease encoding |
| Disease encoder | `--disease-encoder-type` | hpo_attention | {hpo_attention, disease_id} |
| Disease size embed | `--disease-size-embed` | 1 | Embed disease HPO set size as feature |
| Enriched trait graph | `--enrich-trait-graph` | 1 | Use HPO is_a + cosine kNN trait edges |

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
| Func mechanism pos weight | `--func-mechanism-pos-weight` | 3.0 | BCE positive weight for mechanism axis |

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
| Cache refresh | `--vd-kl-cache-refresh-interval` | 1 | Refresh teacher cache every N epochs |
| Positive smoothing | `--vd-kl-positive-smoothing` | 0.10 | |
| D2V positive smoothing | `--vd-kl-d2v-positive-smoothing` | 0.20 | |
| Concept direct weight | `--vd-kl-concept-direct-weight` | 1.0 | Weight for direct CUI mapping |
| Concept semantic weight | `--vd-kl-concept-semantic-weight` | 0.35 | Weight for SapBERT semantic mapping |
| HPO topk per HPO | `--vd-kl-hpo-topk-per-hpo` | 5 | Semantic mapping: top-k MVPs per HPO |
| HPO min similarity | `--vd-kl-hpo-min-similarity` | 0.35 | Semantic mapping: similarity threshold |
| Min variant mapped mass | `--vd-kl-min-variant-mapped-mass` | 0.05 | Skip variants with too little teacher signal |
| Max diseases per variant | `--vd-kl-max-diseases-per-variant` | 32 | Cap teacher distribution width |
| Max variants per disease | `--vd-kl-max-variants-per-disease` | 256 | Cap reverse teacher width |
| D2V batch size | `--vd-kl-d2v-batch-size` | 32 | |
| D2V teacher topk | `--vd-kl-d2v-teacher-topk` | 32 | |
| D2V random negatives | `--vd-kl-d2v-random-negatives` | 128 | |
| Shuffle teacher (control) | `--vd-kl-shuffle-teacher` | 0 | Set 1 for shuffled-teacher control experiment |

### Split Configuration

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Protocol | `--split-protocol` | gene_holdout | {gene_holdout, within_gene, disease_holdout} |
| Mode | `--split-mode` | auto | {auto, generate, load} |
| Seed | `--seed` | 42 | Used for both split and training |
| Split artifact | `--split-artifact-path` | artifacts/splits/default_split.json | For reproducible splits |
| Split only | `--split-only` | 0 | Set 1 to generate split and exit without training |

## Known Parameter Interactions

1. **LR ↔ grad_clip_norm**: At lr=1e-3, grad_clip=5.0 works. If increasing LR, may need to increase clip.
2. **vd_kl_temperature ↔ main_logit_scale_max**: Both control softmax sharpness. Don't lower both simultaneously (t=0.05 + s=15 over-suppresses main task).
3. **KL schedule ↔ cosine restart**: KL warmup/ramp should complete within the first cosine cycle (epoch 0-20 at default `--scheduler-t0=20`). If changing T_0, adjust KL schedule accordingly.
4. **early_stopping_patience > warmup epochs**: Must be larger than both `main_only_warmup_epochs` and `vd_kl_warmup_epochs`, otherwise model gets killed before aux tasks activate.
5. **main_logit_scale_max**: Use 15 with KL enabled, 30 without. Higher scale with KL drowns the KL signal.
6. **lr_graph effective value**: When `graph_train_mode=weak`, actual graph LR = `min(lr_graph, lr * 0.1)`. Raising `--lr` alone won't automatically raise graph LR if lr_graph is the bottleneck.
7. **vd_kl_lambda_v2d > 0.8**: Tested λ={0.6, 0.8, 1.0}. λ>0.6 inflates gm_nDCG but degrades nDCG/MRR/AUPRC — the KL over-flattens the distribution.

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
