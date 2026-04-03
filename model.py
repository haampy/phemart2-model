from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HGTConv
except Exception as exc:  # pragma: no cover
    HGTConv = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoGraphEncoder(nn.Module):
    """Bypass graph encoder: project input features directly without message passing.

    Matches GraphEncoder's MLP capacity (in→hidden→out with ReLU + dropout)
    so that graph_mode="none" ablation isolates message-passing contribution,
    not MLP depth.
    """

    def __init__(
        self,
        gene_in_dim: int,
        trait_in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gene_mlp = nn.Sequential(
            nn.Linear(gene_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.trait_mlp = nn.Sequential(
            nn.Linear(trait_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gene_emb = self.gene_mlp(x_dict["gene"])
        trait_emb = self.trait_mlp(x_dict["trait"])
        return gene_emb, trait_emb


class GraphEncoder(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        gene_in_dim: int,
        trait_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if _PYG_IMPORT_ERROR is not None:
            raise ImportError("torch_geometric is required for GraphEncoder") from _PYG_IMPORT_ERROR

        self.dropout = dropout
        self.input_proj = nn.ModuleDict(
            {
                "gene": nn.Linear(gene_in_dim, hidden_dim),
                "trait": nn.Linear(trait_in_dim, hidden_dim),
            }
        )
        self.input_norm = nn.ModuleDict(
            {
                "gene": nn.LayerNorm(hidden_dim),
                "trait": nn.LayerNorm(hidden_dim),
            }
        )

        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads) for _ in range(num_layers)]
        )
        self.out_proj = nn.ModuleDict(
            {
                "gene": nn.Linear(hidden_dim, out_dim),
                "trait": nn.Linear(hidden_dim, out_dim),
            }
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = {}
        for node_type, x in x_dict.items():
            h[node_type] = self.input_norm[node_type](self.input_proj[node_type](x))

        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            for node_type in h.keys():
                if node_type in h_new:
                    h[node_type] = F.dropout(
                        F.relu(h_new[node_type] + h[node_type]),
                        p=self.dropout,
                        training=self.training,
                    )

        gene_emb = self.out_proj["gene"](h["gene"])
        trait_emb = self.out_proj["trait"](h["trait"])
        return gene_emb, trait_emb


class TrilinearFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        modality_drop_variant: float = 0.0,
        modality_drop_protein: float = 0.0,
        modality_drop_gene: float = 0.0,
    ) -> None:
        super().__init__()
        self.variant_proj = nn.Linear(dim, dim)
        self.protein_proj = nn.Linear(dim, dim)
        self.gene_proj = nn.Linear(dim, dim)
        self.variant_norm = nn.LayerNorm(dim)
        self.protein_norm = nn.LayerNorm(dim)
        self.gene_norm = nn.LayerNorm(dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.final = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.modality_drop_variant = modality_drop_variant
        self.modality_drop_protein = modality_drop_protein
        self.modality_drop_gene = modality_drop_gene

        self.register_buffer("gate_bias", torch.tensor([0.1, 0.0, -0.1], dtype=torch.float32))

    def forward(
        self,
        variant_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        gene_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            bsz = variant_emb.shape[0]
            mv = torch.ones(bsz, 1, device=variant_emb.device)
            mp = torch.ones(bsz, 1, device=variant_emb.device)
            mg = torch.ones(bsz, 1, device=variant_emb.device)

            if self.modality_drop_variant > 0:
                mv = torch.bernoulli(mv * (1.0 - self.modality_drop_variant))
            if self.modality_drop_protein > 0:
                mp = torch.bernoulli(mp * (1.0 - self.modality_drop_protein))
            if self.modality_drop_gene > 0:
                mg = torch.bernoulli(mg * (1.0 - self.modality_drop_gene))

            all_zero = (mv + mp + mg) == 0
            if all_zero.any():
                mv[all_zero] = 1.0

            variant_emb = variant_emb * mv
            protein_emb = protein_emb * mp
            gene_emb = gene_emb * mg

        v = self.variant_norm(self.variant_proj(variant_emb))
        p = self.protein_norm(self.protein_proj(protein_emb))
        g = self.gene_norm(self.gene_proj(gene_emb))

        logits = self.gate(torch.cat([v, p, g], dim=-1)) + self.gate_bias
        weights = F.softmax(logits / max(gate_temperature, 1e-3), dim=-1)

        fused = weights[:, 0:1] * variant_emb + weights[:, 1:2] * protein_emb + weights[:, 2:3] * gene_emb
        fused = self.final(fused)
        if return_gate_weights:
            return fused, weights
        return fused


class ConcatFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        modality_drop_variant: float = 0.0,
        modality_drop_protein: float = 0.0,
        modality_drop_gene: float = 0.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.modality_drop_variant = modality_drop_variant
        self.modality_drop_protein = modality_drop_protein
        self.modality_drop_gene = modality_drop_gene

    def _apply_modality_dropout(
        self,
        variant_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        gene_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            return variant_emb, protein_emb, gene_emb
        bsz = variant_emb.shape[0]
        mv = torch.ones(bsz, 1, device=variant_emb.device)
        mp = torch.ones(bsz, 1, device=variant_emb.device)
        mg = torch.ones(bsz, 1, device=variant_emb.device)
        if self.modality_drop_variant > 0:
            mv = torch.bernoulli(mv * (1.0 - self.modality_drop_variant))
        if self.modality_drop_protein > 0:
            mp = torch.bernoulli(mp * (1.0 - self.modality_drop_protein))
        if self.modality_drop_gene > 0:
            mg = torch.bernoulli(mg * (1.0 - self.modality_drop_gene))
        all_zero = (mv + mp + mg) == 0
        if all_zero.any():
            mv[all_zero] = 1.0
        return variant_emb * mv, protein_emb * mp, gene_emb * mg

    def forward(
        self,
        variant_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        gene_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        v, p, g = self._apply_modality_dropout(variant_emb, protein_emb, gene_emb)
        fused = self.mlp(torch.cat([v, p, g], dim=-1))
        if return_gate_weights:
            uniform = torch.full((variant_emb.shape[0], 3), 1.0 / 3, device=variant_emb.device)
            return fused, uniform
        return fused


class ConcatResidualFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        modality_drop_variant: float = 0.0,
        modality_drop_protein: float = 0.0,
        modality_drop_gene: float = 0.0,
        residual_alpha_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        if residual_alpha_max <= 0.0 or residual_alpha_max > 1.0:
            raise ValueError("residual_alpha_max must be in (0, 1]")
        self.residual_alpha = nn.Parameter(torch.tensor(0.0))
        # Alpha now controls a convex combination between fusion output and
        # raw variant features, so alpha_max=0.4 truly guarantees that at
        # least 60% of the signal comes through the fusion MLP path.
        self.residual_alpha_max = residual_alpha_max
        self.modality_drop_variant = modality_drop_variant
        self.modality_drop_protein = modality_drop_protein
        self.modality_drop_gene = modality_drop_gene

    def _apply_modality_dropout(
        self,
        variant_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        gene_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            return variant_emb, protein_emb, gene_emb
        bsz = variant_emb.shape[0]
        mv = torch.ones(bsz, 1, device=variant_emb.device)
        mp = torch.ones(bsz, 1, device=variant_emb.device)
        mg = torch.ones(bsz, 1, device=variant_emb.device)
        if self.modality_drop_variant > 0:
            mv = torch.bernoulli(mv * (1.0 - self.modality_drop_variant))
        if self.modality_drop_protein > 0:
            mp = torch.bernoulli(mp * (1.0 - self.modality_drop_protein))
        if self.modality_drop_gene > 0:
            mg = torch.bernoulli(mg * (1.0 - self.modality_drop_gene))
        all_zero = (mv + mp + mg) == 0
        if all_zero.any():
            mv[all_zero] = 1.0
        return variant_emb * mv, protein_emb * mp, gene_emb * mg

    def forward(
        self,
        variant_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        gene_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Save variant_emb BEFORE modality dropout for residual path
        variant_raw = variant_emb
        v, p, g = self._apply_modality_dropout(variant_emb, protein_emb, gene_emb)
        fused = self.mlp(torch.cat([v, p, g], dim=-1))
        alpha = torch.sigmoid(self.residual_alpha)
        alpha = alpha * self.residual_alpha_max
        fused = (1.0 - alpha) * fused + alpha * variant_raw
        if return_gate_weights:
            uniform = torch.full((variant_emb.shape[0], 3), 1.0 / 3, device=variant_emb.device)
            return fused, uniform
        return fused


class DiseaseEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        trait_dropout: float = 0.0,
        use_size_embed: bool = False,
    ) -> None:
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )
        self.attn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.scale = math.sqrt(dim)
        self.trait_dropout = trait_dropout
        self.use_size_embed = use_size_embed
        if use_size_embed:
            self.size_proj = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, dim),
            )
        self._index_cache_cpu: Dict[Tuple[int, ...], Dict[str, torch.Tensor]] = {}
        self._index_cache_device: Dict[Tuple[int, ...], Dict[str, Dict[str, torch.Tensor]]] = {}

    def _build_index_cache(
        self,
        disease_ids: Sequence[int],
        disease_to_traits: Dict[int, List[int]],
    ) -> Dict[str, torch.Tensor]:
        disease_ids_int = [int(d) for d in disease_ids]
        if not disease_ids_int:
            return {
                "trait_idx": torch.empty((0, 0), dtype=torch.long),
                "mask": torch.empty((0, 0), dtype=torch.bool),
                "log_size": torch.empty((0, 1), dtype=torch.float32),
            }

        trait_lists: List[List[int]] = []
        max_traits = 0
        for disease_id in disease_ids_int:
            trait_ids = disease_to_traits.get(disease_id, [])
            if not trait_ids:
                raise ValueError(f"Disease {disease_id} has no mapped traits")
            trait_ids_int = [int(t) for t in trait_ids]
            trait_lists.append(trait_ids_int)
            max_traits = max(max_traits, len(trait_ids_int))

        n_diseases = len(trait_lists)
        trait_idx = torch.zeros((n_diseases, max_traits), dtype=torch.long)
        mask = torch.zeros((n_diseases, max_traits), dtype=torch.bool)
        log_size = torch.zeros((n_diseases, 1), dtype=torch.float32)
        for row_idx, trait_ids in enumerate(trait_lists):
            k = len(trait_ids)
            trait_idx[row_idx, :k] = torch.tensor(trait_ids, dtype=torch.long)
            mask[row_idx, :k] = True
            log_size[row_idx, 0] = math.log1p(k)

        return {"trait_idx": trait_idx, "mask": mask, "log_size": log_size}

    def _get_index_cache(
        self,
        disease_ids: Sequence[int],
        disease_to_traits: Dict[int, List[int]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        cache_key = tuple(int(d) for d in disease_ids)
        if cache_key not in self._index_cache_cpu:
            self._index_cache_cpu[cache_key] = self._build_index_cache(
                disease_ids=disease_ids,
                disease_to_traits=disease_to_traits,
            )

        device_key = str(device)
        per_device = self._index_cache_device.setdefault(cache_key, {})
        if device_key not in per_device:
            cpu_cache = self._index_cache_cpu[cache_key]
            per_device[device_key] = {
                "trait_idx": cpu_cache["trait_idx"].to(device=device, non_blocking=True),
                "mask": cpu_cache["mask"].to(device=device, non_blocking=True),
                "log_size": cpu_cache["log_size"].to(device=device, non_blocking=True),
            }
        return per_device[device_key]

    def forward(
        self,
        trait_graph_emb: torch.Tensor,
        disease_ids: Sequence[int],
        disease_to_traits: Dict[int, List[int]],
    ) -> torch.Tensor:
        index_cache = self._get_index_cache(
            disease_ids=disease_ids,
            disease_to_traits=disease_to_traits,
            device=trait_graph_emb.device,
        )
        trait_idx = index_cache["trait_idx"]
        base_mask = index_cache["mask"]
        log_size = index_cache["log_size"]
        n_diseases, max_traits = trait_idx.shape
        if n_diseases == 0 or max_traits == 0:
            return torch.empty((0, trait_graph_emb.shape[-1]), device=trait_graph_emb.device)

        x = trait_graph_emb.index_select(0, trait_idx.reshape(-1)).reshape(
            n_diseases, max_traits, trait_graph_emb.shape[-1]
        )

        effective_mask = base_mask
        if self.training and self.trait_dropout > 0 and max_traits > 1:
            keep_mask = (torch.rand(base_mask.shape, device=base_mask.device) > self.trait_dropout) & base_mask
            row_has_trait = keep_mask.any(dim=1)
            if not row_has_trait.all():
                first_valid = base_mask.to(torch.int64).argmax(dim=1)
                missing_rows = (~row_has_trait).nonzero(as_tuple=False).squeeze(-1)
                keep_mask[missing_rows, first_valid.index_select(0, missing_rows)] = True
            effective_mask = keep_mask

        h = self.shared_mlp(x)
        logits = self.attn(h) / self.scale
        logits = logits.masked_fill(
            ~effective_mask.unsqueeze(-1),
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * h).sum(dim=1)

        if self.use_size_embed:
            pooled = pooled + self.size_proj(log_size.to(dtype=pooled.dtype))

        return pooled


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        gene_in_dim: int,
        trait_in_dim: int,
        variant_in_dim: int,
        protein_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        num_graph_layers: int,
        dropout: float,
        num_domains: int,
        domain_embedding_dim: int,
        func_conservation_dim: int = 5,
        func_protein_impact_dim: int = 5,
        func_integrative_dim: int = 5,
        func_mechanism_dim: int = 9,
        modality_drop_variant: float = 0.0,
        modality_drop_protein: float = 0.0,
        modality_drop_gene: float = 0.0,
        main_temperature: float = 0.15,
        main_logit_scale_learnable: bool = True,
        main_logit_scale_init: float = 0.0,
        trait_dropout: float = 0.0,
        disease_size_embed: bool = False,
        fusion_type: str = "concat_residual",
        disease_encoder_type: str = "hpo_attention",
        num_diseases: int = 0,
        graph_mode: str = "hgt",
        residual_alpha_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.graph_mode = graph_mode
        self._residual_alpha_max = residual_alpha_max
        if graph_mode == "none":
            self.graph_encoder = NoGraphEncoder(
                gene_in_dim=gene_in_dim,
                trait_in_dim=trait_in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.graph_encoder = GraphEncoder(
                metadata=metadata,
                gene_in_dim=gene_in_dim,
                trait_in_dim=trait_in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_heads=num_heads,
                num_layers=num_graph_layers,
                dropout=dropout,
            )
        self.variant_encoder = MLPEncoder(variant_in_dim, hidden_dim, out_dim, dropout)
        self.protein_encoder = MLPEncoder(protein_in_dim, hidden_dim, out_dim, dropout)
        fusion_kwargs = dict(
            dim=out_dim,
            dropout=dropout,
            modality_drop_variant=modality_drop_variant,
            modality_drop_protein=modality_drop_protein,
            modality_drop_gene=modality_drop_gene,
        )
        if fusion_type == "gated":
            self.fusion = TrilinearFusion(**fusion_kwargs)
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(**fusion_kwargs)
        elif fusion_type == "concat_residual":
            self.fusion = ConcatResidualFusion(**fusion_kwargs, residual_alpha_max=residual_alpha_max)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.disease_encoder_type = disease_encoder_type
        if disease_encoder_type == "disease_id":
            assert num_diseases > 0, "num_diseases required for disease_id encoder"
            self.disease_id_emb = nn.Embedding(num_diseases, out_dim)
            nn.init.normal_(self.disease_id_emb.weight, std=0.02)
            self.disease_encoder = None
        else:
            self.disease_encoder = DiseaseEncoder(
                out_dim, hidden_dim, dropout,
                trait_dropout=trait_dropout,
                use_size_embed=disease_size_embed,
            )
            self.disease_id_emb = None

        self.clip_variant_proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.clip_disease_proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        if main_logit_scale_init > 0:
            init_main_scale = main_logit_scale_init
        else:
            init_main_scale = 1.0 / max(main_temperature, 1e-6)
        self.main_logit_scale_log = nn.Parameter(
            torch.tensor(math.log(init_main_scale), dtype=torch.float32),
            requires_grad=main_logit_scale_learnable,
        )

        self.domain_variant_proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.domain_transform = nn.Linear(domain_embedding_dim, out_dim)
        self.num_domains = num_domains

        def _func_axis_head(out_d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_d),
            )

        self.func_conservation_head = _func_axis_head(func_conservation_dim)
        self.func_protein_impact_head = _func_axis_head(func_protein_impact_dim)
        self.func_integrative_head = _func_axis_head(func_integrative_dim)
        self.func_mechanism_head = _func_axis_head(func_mechanism_dim) if func_mechanism_dim > 0 else None

    def forward_graph(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.graph_encoder(x_dict, edge_index_dict)

    def encode_variant(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        v = self.variant_encoder(variant_x.index_select(0, variant_ids))
        # build_feature_store keeps protein_x aligned to the same per-variant row index as variant_x.
        p = self.protein_encoder(protein_x.index_select(0, variant_ids))
        g = gene_graph_emb.index_select(0, gene_ids)
        return self.fusion(
            v,
            p,
            g,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )

    def get_main_logit_scale(
        self,
        min_scale: float = 1.0,
        max_scale: float = 30.0,
    ) -> torch.Tensor:
        return torch.exp(self.main_logit_scale_log).clamp(min=min_scale, max=max_scale)

    def encode_disease_batch(
        self,
        disease_ids: Sequence[int],
        disease_to_traits: Dict[int, List[int]],
        trait_graph_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.disease_encoder_type == "disease_id":
            dev = next(self.parameters()).device
            ids_t = torch.as_tensor(disease_ids, dtype=torch.long, device=dev)
            raw = self.disease_id_emb(ids_t)
        else:
            raw = self.disease_encoder(trait_graph_emb, disease_ids, disease_to_traits)
        return F.normalize(self.clip_disease_proj(raw), dim=-1)

    def forward_main(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        trait_graph_emb: torch.Tensor,
        disease_ids: Sequence[int],
        disease_to_traits: Dict[int, List[int]],
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_v_out = self.encode_variant(
            variant_ids,
            gene_ids,
            variant_x,
            protein_x,
            gene_graph_emb,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )
        if return_gate_weights:
            z_v, gate_weights = z_v_out
        else:
            z_v = z_v_out
        z_v = F.normalize(self.clip_variant_proj(z_v), dim=-1)
        z_d = self.encode_disease_batch(disease_ids, disease_to_traits, trait_graph_emb)
        if return_gate_weights:
            return z_v, z_d, gate_weights
        return z_v, z_d

    def forward_domain(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        domain_embeddings: torch.Tensor,
        temperature: float = 0.15,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        z_v_out = self.encode_variant(
            variant_ids,
            gene_ids,
            variant_x,
            protein_x,
            gene_graph_emb,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )
        if return_gate_weights:
            z_v, gate_weights = z_v_out
        else:
            z_v = z_v_out
        z_v = F.normalize(self.domain_variant_proj(z_v), dim=-1)
        z_p = F.normalize(self.domain_transform(domain_embeddings), dim=-1)
        logits = z_v @ z_p.t() / max(temperature, 1e-6)
        if return_gate_weights:
            return logits, gate_weights
        return logits

    def forward_func(
        self,
        variant_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        variant_x: torch.Tensor,
        protein_x: torch.Tensor,
        gene_graph_emb: torch.Tensor,
        gate_temperature: float = 1.0,
        return_gate_weights: bool = False,
    ) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        z_v_out = self.encode_variant(
            variant_ids,
            gene_ids,
            variant_x,
            protein_x,
            gene_graph_emb,
            gate_temperature=gate_temperature,
            return_gate_weights=return_gate_weights,
        )
        if return_gate_weights:
            z_v, gate_weights = z_v_out
        else:
            z_v = z_v_out
        preds: Dict[str, torch.Tensor] = {
            "conservation": self.func_conservation_head(z_v),
            "protein_impact": self.func_protein_impact_head(z_v),
            "integrative": self.func_integrative_head(z_v),
        }
        if self.func_mechanism_head is not None:
            preds["mechanism"] = self.func_mechanism_head(z_v)
        if return_gate_weights:
            return preds, gate_weights
        return preds
