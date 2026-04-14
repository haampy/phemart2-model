from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import hashlib
import json
import math
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import ToUndirected
except Exception as exc:  # pragma: no cover
    HeteroData = Any
    ToUndirected = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


FUNC_REGRESSION_COLS = [
    # Conservation (axis 0-4)
    "phyloP100way_rs", "phyloP17way_rs", "phastCons100way_rs", "phastCons17way_rs", "GERP_RS_rs",
    # Protein Impact (axis 5-9)
    "ESM1b_rs", "PROVEAN_rs", "SIFT_rs", "Polyphen2_HDIV_rs", "AlphaMissense_rs",
    # Integrative (axis 10-14)
    "CADD_rs", "MetaRNN_rs", "BayesDel_rs", "REVEL_rs", "ClinPred_rs",
]
FUNC_REGRESSION_MASK_COLS = [f"{c}_mask" for c in FUNC_REGRESSION_COLS]
FUNC_MECHANISM_COLS = [
    "mech_structure", "mech_ptm", "mech_binding", "mech_metal",
    "mech_membrane", "mech_dna", "mech_catalytic", "mech_disulfide", "mech_solvent",
]
FUNC_AXIS_SLICES = {
    "conservation": (0, 5),
    "protein_impact": (5, 10),
    "integrative": (10, 15),
}
# 保留旧名以兼容导入
FUNC_TARGET_COLS = FUNC_REGRESSION_COLS
_GENE_TOKEN_SPLIT_RE = re.compile(r"[;|,]")
_HGVS_GENE_HINT_RE = re.compile(r"\(([^)]+)\)")
_HGVS_PROTEIN_POS_RE = re.compile(r"\(p\.[a-z]+(\d+)")
_HGVS_CDNA_POS_RE = re.compile(r":c\.(\d+)")


def get_func_mask_cols(target_cols: Sequence[str]) -> List[str]:
    return [f"{c}_mask" for c in target_cols]



def normalize_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def parse_gene_tokens(raw: Any) -> List[str]:
    text = normalize_id(raw)
    if not text:
        return []
    tokens = [t.strip() for t in _GENE_TOKEN_SPLIT_RE.split(text) if t.strip()]
    dedup: List[str] = []
    for token in tokens:
        if token not in dedup:
            dedup.append(token)
    return dedup


def parse_hgvs_gene_hint(variant_id: Any) -> str:
    text = normalize_id(variant_id)
    if not text:
        return ""
    match = _HGVS_GENE_HINT_RE.search(text)
    if match is None:
        return ""
    return normalize_id(match.group(1))


def parse_hgvs_protein_position(variant_id: Any) -> Optional[int]:
    text = normalize_id(variant_id)
    if not text:
        return None
    m = _HGVS_PROTEIN_POS_RE.search(text)
    if m:
        return int(m.group(1))
    m = _HGVS_CDNA_POS_RE.search(text)
    if m:
        return int(m.group(1)) // 3
    return None


def resolve_gene_id(
    raw_gene: Any,
    variant_id: Any = "",
    allowed_gene_ids: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    text = normalize_id(raw_gene)
    if not text:
        return "", "empty"
    if allowed_gene_ids is not None and text in allowed_gene_ids:
        return text, "direct"

    tokens = parse_gene_tokens(text)
    if not tokens:
        tokens = [text]

    if allowed_gene_ids is None:
        if len(tokens) == 1:
            return tokens[0], "token"
        hint = parse_hgvs_gene_hint(variant_id)
        if hint and hint in tokens:
            return hint, "variant_hint"
        return text, "ambiguous_raw"

    hint = parse_hgvs_gene_hint(variant_id)
    if hint and hint in allowed_gene_ids:
        if hint in tokens or text.startswith("ensg") or not any(t in allowed_gene_ids for t in tokens):
            return hint, "variant_hint"

    covered_tokens = [token for token in tokens if token in allowed_gene_ids]
    if len(covered_tokens) == 1:
        return covered_tokens[0], "unique_token"
    if hint and hint in covered_tokens:
        return hint, "variant_hint"

    preferred_tokens = [
        token for token in covered_tokens if not token.startswith("loc") and "-as" not in token
    ]
    if len(preferred_tokens) == 1:
        return preferred_tokens[0], "preferred_symbol"

    return "", "unresolved"


def parse_hpo_ids(raw: Any) -> List[str]:
    if pd.isna(raw):
        return []
    if isinstance(raw, (list, tuple)):
        return [normalize_id(v) for v in raw if normalize_id(v)]
    text = str(raw)
    return [normalize_id(v) for v in text.split("|") if normalize_id(v)]


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _standardize_variant_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    var_col = _first_existing(df, ["variant_id", "variant", "snps", "snp", "variant_raw"])
    gene_col = _first_existing(df, ["gene_id", "gene_name", "genes", "gene"])
    if var_col is None:
        raise ValueError("Cannot find variant column")
    if gene_col is None:
        raise ValueError("Cannot find gene column")
    if var_col != "variant_id":
        df = df.rename(columns={var_col: "variant_id"})
    if gene_col != "gene_id":
        df = df.rename(columns={gene_col: "gene_id"})
    df["variant_id"] = df["variant_id"].map(normalize_id)
    df["gene_id"] = df["gene_id"].map(normalize_id)
    df = df[(df["variant_id"] != "") & (df["gene_id"] != "")]
    return df


def load_main_labels(path: str) -> pd.DataFrame:
    df = _standardize_variant_gene_columns(_read_csv(path))
    disease_col = _first_existing(df, ["disease_index", "disease_id"])
    hpo_col = _first_existing(df, ["hpo_ids", "hpo_id"])
    confidence_col = _first_existing(df, ["confidence"])
    if disease_col is None:
        raise ValueError("Cannot find disease_index column in main labels")
    if disease_col != "disease_index":
        df = df.rename(columns={disease_col: "disease_index"})
    if hpo_col and hpo_col != "hpo_ids":
        df = df.rename(columns={hpo_col: "hpo_ids"})
    if "hpo_ids" not in df.columns:
        df["hpo_ids"] = ""
    if confidence_col and confidence_col != "confidence":
        df = df.rename(columns={confidence_col: "confidence"})
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = (
        pd.to_numeric(df["confidence"], errors="coerce")
        .fillna(1.0)
        .astype(np.float32)
    )
    df = df.dropna(subset=["disease_index"])
    df["disease_index"] = df["disease_index"].astype(int)
    return df[["variant_id", "gene_id", "disease_index", "hpo_ids", "confidence"]]


def load_disease_table(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    disease_col = _first_existing(df, ["disease_index", "disease_id"])
    hpo_col = _first_existing(df, ["hpo_ids", "hpo_id"])
    if disease_col is None or hpo_col is None:
        raise ValueError("Disease table must contain disease_index and hpo_ids")
    df = df.rename(columns={disease_col: "disease_index", hpo_col: "hpo_ids"})
    df = df.dropna(subset=["disease_index"])
    df["disease_index"] = df["disease_index"].astype(int)
    return df[["disease_index", "hpo_ids"]].drop_duplicates("disease_index")


def load_domain_labels(path: str) -> pd.DataFrame:
    df = _standardize_variant_gene_columns(_read_csv(path))
    domain_col = _first_existing(df, ["domain_map", "domain", "label"])
    if domain_col is None:
        raise ValueError("Cannot find domain label column")
    if domain_col != "domain_map":
        df = df.rename(columns={domain_col: "domain_map"})
    df = df.dropna(subset=["domain_map"])
    df["domain_map"] = df["domain_map"].astype(int)
    return df[["variant_id", "gene_id", "domain_map"]]


def load_func_labels(
    path: str,
    target_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """加载 v2 多轴 func 数据。支持 v1 (旧8列) 和 v2 (15回归+9机制) 格式。"""
    df = _standardize_variant_gene_columns(_read_csv(path))

    # 检测是否为 v2 格式
    is_v2 = "phyloP100way_rs" in df.columns
    if is_v2:
        reg_cols = list(FUNC_REGRESSION_COLS)
        reg_mask_cols = list(FUNC_REGRESSION_MASK_COLS)
        mech_cols = [c for c in FUNC_MECHANISM_COLS if c in df.columns]
        has_mechanism = "mechanism_mask" in df.columns and len(mech_cols) > 0

        missing = [c for c in reg_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing FUNC regression columns: {missing}")
        missing_masks = [c for c in reg_mask_cols if c not in df.columns]
        if missing_masks:
            raise ValueError(f"Missing FUNC regression mask columns: {missing_masks}")

        all_cols = ["variant_id", "gene_id"] + reg_cols + reg_mask_cols
        if has_mechanism:
            all_cols += mech_cols + ["mechanism_mask"]
        df = df[all_cols].dropna(subset=["variant_id", "gene_id"])

        for c in reg_mask_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)
        for c in reg_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for tc, mc in zip(reg_cols, reg_mask_cols):
            df.loc[df[mc] == 0, tc] = 0.0
        if has_mechanism:
            df["mechanism_mask"] = pd.to_numeric(df["mechanism_mask"], errors="coerce").fillna(0).astype(np.float32)
            for c in mech_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)

        n_nan = int(df[reg_cols].isna().any(axis=1).sum())
        if n_nan > 0:
            print(f"func_warn: {n_nan} rows have NaN in mask=1 targets, dropping")
            df = df.dropna(subset=reg_cols)
    else:
        # 旧 v1 格式 fallback
        selected_targets = list(target_cols) if target_cols is not None else list(FUNC_TARGET_COLS)
        selected_masks = get_func_mask_cols(selected_targets)
        missing_targets = [c for c in selected_targets if c not in df.columns]
        missing_masks = [c for c in selected_masks if c not in df.columns]
        if missing_targets:
            raise ValueError(f"Missing FUNC target columns: {missing_targets}")
        if missing_masks:
            raise ValueError(f"Missing FUNC mask columns: {missing_masks}")
        cols = ["variant_id", "gene_id"] + selected_targets + selected_masks
        df = df[cols].dropna(subset=["variant_id", "gene_id"])
        for c in selected_masks:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)
        for c in selected_targets:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for tc, mc in zip(selected_targets, selected_masks):
            df.loc[df[mc] == 0, tc] = 0.0
        n_nan = int(df[selected_targets].isna().any(axis=1).sum())
        if n_nan > 0:
            print(f"func_warn: {n_nan} rows have NaN in mask=1 targets, dropping")
            df = df.dropna(subset=selected_targets)
    return df


def load_gene_concept_targets(
    svd_path: str,
    metadata_path: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """加载 gene-level concept SVD embeddings。

    Returns:
        embeddings: float32 array [n_genes, dim]
        gene_to_row: gene name (大写) -> row index 映射
    """
    embeddings = np.load(svd_path).astype(np.float32)
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    gene_to_row: Dict[str, int] = {
        str(k).upper(): int(v) for k, v in meta["gene_to_row"].items()
    }
    print(f"gene_concept_targets: {embeddings.shape[0]} genes, dim={embeddings.shape[1]}, "
          f"gene_to_row entries={len(gene_to_row)}")
    return embeddings, gene_to_row


def _load_embeddings_npy(
    index_path: str, data_path: str, required_ids: Optional[Set[str]] = None
) -> pd.DataFrame:
    """Fast path: load from pre-converted npy files."""
    ids = np.load(index_path, allow_pickle=True)
    # Vectorized normalization (avoids slow Python loop over 800K+ items).
    ids_normalized = np.char.lower(np.char.strip(ids.astype(str)))

    if required_ids is None:
        data = np.load(data_path)
        df = pd.DataFrame(data, index=ids_normalized).astype(np.float32)
        df = df[~df.index.duplicated(keep="first")]
        df.columns = df.columns.map(str)
        return df

    # Set-based lookup (faster than np.isin for object/string arrays).
    required_set = set(required_ids)
    mask = np.array([x in required_set for x in ids_normalized])
    row_indices = np.where(mask)[0]
    # Load full array into RAM instead of mmap — network filesystems (NFS/Lustre)
    # have catastrophic random-access latency with mmap.
    data = np.load(data_path)
    selected = data[row_indices]
    del data
    df = pd.DataFrame(selected, index=ids_normalized[row_indices]).astype(np.float32)
    df = df[~df.index.duplicated(keep="first")]
    # Keep feature columns consistent with CSV-loaded embeddings.
    df.columns = df.columns.map(str)
    return df


def load_embeddings(
    path: str,
    required_ids: Optional[Set[str]] = None,
    index_col: Optional[str] = None,
    chunksize: int = 50000,
) -> pd.DataFrame:
    # Fast path: use pre-converted npy files if available
    p = Path(path)
    index_npy = p.parent / f"{p.stem}_index.npy"
    data_npy = p.parent / f"{p.stem}_data.npy"
    if index_npy.exists() and data_npy.exists():
        print(f"  [npy fast-load] {p.name} -> {data_npy.name}")
        return _load_embeddings_npy(str(index_npy), str(data_npy), required_ids)

    # Slow path: CSV chunked reading
    if required_ids is None:
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.map(normalize_id)
        df = df[~df.index.duplicated(keep="first")]
        df = df.astype(np.float32)
        df.columns = df.columns.map(str)
        return df

    def _collect_chunks(reader) -> List[pd.DataFrame]:
        out: List[pd.DataFrame] = []
        for chunk in reader:
            idx_col = index_col or chunk.columns[0]
            chunk = chunk.rename(columns={idx_col: "__id__"})
            chunk["__id__"] = chunk["__id__"].map(normalize_id)
            sub = chunk[chunk["__id__"].isin(required_ids)]
            if not sub.empty:
                out.append(sub)
        return out

    try:
        collected = _collect_chunks(pd.read_csv(path, chunksize=chunksize))
    except pd.errors.ParserError:
        collected = _collect_chunks(pd.read_csv(path, chunksize=chunksize, engine="python"))

    if not collected:
        return pd.DataFrame(dtype=np.float32)

    df = pd.concat(collected, axis=0)
    df = df.drop_duplicates(subset=["__id__"], keep="first").set_index("__id__")
    df = df.astype(np.float32)
    df.columns = df.columns.map(str)
    return df


def build_rsid_to_hgvs_map(
    hgvs_embed_csv: str,
    rsid_embed_csv: str,
    chunksize: int = 200000,
) -> Dict[str, str]:
    out: Dict[str, str] = {}

    # Fast path: use pre-converted npy index files if available.
    hgvs_idx_npy = Path(hgvs_embed_csv).parent / f"{Path(hgvs_embed_csv).stem}_index.npy"
    rsid_idx_npy = Path(rsid_embed_csv).parent / f"{Path(rsid_embed_csv).stem}_index.npy"
    if hgvs_idx_npy.exists() and rsid_idx_npy.exists():
        hgvs_ids = np.load(str(hgvs_idx_npy), allow_pickle=True)
        rsid_ids = np.load(str(rsid_idx_npy), allow_pickle=True)
        if len(hgvs_ids) != len(rsid_ids):
            raise ValueError("HGVS and RSID index files have mismatched lengths")
        for rsid_raw, hgvs_raw in zip(rsid_ids, hgvs_ids):
            rsid = normalize_id(str(rsid_raw))
            hgvs = normalize_id(str(hgvs_raw))
            if rsid and hgvs and rsid not in out:
                out[rsid] = hgvs
        print(f"rsid_to_hgvs_map: npy fast path, {len(out)} entries")
        return out

    # Slow path: chunked CSV reading.
    hgvs_iter = pd.read_csv(hgvs_embed_csv, usecols=["variant_id"], chunksize=chunksize)
    rsid_iter = pd.read_csv(rsid_embed_csv, usecols=["variant_id"], chunksize=chunksize)

    for hgvs_chunk, rsid_chunk in zip(hgvs_iter, rsid_iter):
        if len(hgvs_chunk) != len(rsid_chunk):
            raise ValueError("HGVS and RSID embedding files have mismatched chunk sizes")

        hgvs_ids = hgvs_chunk["variant_id"].map(normalize_id).tolist()
        rsid_ids = rsid_chunk["variant_id"].map(normalize_id).tolist()

        for rsid, hgvs in zip(rsid_ids, hgvs_ids):
            if not rsid or not hgvs:
                continue
            if rsid not in out:
                out[rsid] = hgvs

    return out


def remap_variant_ids_to_hgvs(
    df: pd.DataFrame,
    rsid_to_hgvs: Dict[str, str],
    task_name: str,
    preserve_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    before = len(out)
    raw_variant_id = out["variant_id"].map(normalize_id)

    resolved_variant_id: List[str] = []
    mapped_ok = 0
    preserved_ok = 0
    for variant_id in raw_variant_id.tolist():
        mapped = rsid_to_hgvs.get(variant_id)
        if mapped:
            resolved_variant_id.append(mapped)
            mapped_ok += 1
            continue
        if preserve_ids is not None and variant_id in preserve_ids:
            resolved_variant_id.append(variant_id)
            preserved_ok += 1
            continue
        resolved_variant_id.append("")

    out["variant_id"] = resolved_variant_id
    unresolved = int((out["variant_id"] == "").sum())
    out = out[out["variant_id"] != ""].copy()
    print(
        f"{task_name}_id_remap: before={before} mapped={mapped_ok} "
        f"preserved_direct={preserved_ok} "
        f"unresolved_drop={unresolved} after={len(out)}"
    )
    return out


def prepare_task_dataframe_for_training(
    df: pd.DataFrame,
    task_name: str,
    covered_variant_ids: Optional[Set[str]] = None,
    covered_gene_ids: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if "variant_id" not in df.columns or "gene_id" not in df.columns:
        raise ValueError(f"{task_name} dataframe must contain variant_id and gene_id")

    work = df.copy()
    if work.empty:
        return work, {
            "input_rows": 0,
            "rows_after_prepare": 0,
            "dropped_missing_variant": 0,
            "dropped_missing_gene": 0,
            "gene_resolution_direct": 0,
            "gene_resolution_variant_hint": 0,
            "gene_resolution_unique_token": 0,
            "gene_resolution_preferred_symbol": 0,
            "gene_resolution_unresolved": 0,
        }

    work["variant_id"] = work["variant_id"].map(normalize_id)
    raw_gene = work["gene_id"].map(normalize_id)
    resolved_gene: List[str] = []
    resolution_counts = {
        "direct": 0,
        "variant_hint": 0,
        "unique_token": 0,
        "preferred_symbol": 0,
        "unresolved": 0,
    }
    for variant_id, gene_id in zip(work["variant_id"].tolist(), raw_gene.tolist()):
        resolved, reason = resolve_gene_id(
            gene_id,
            variant_id=variant_id,
            allowed_gene_ids=covered_gene_ids,
        )
        resolved_gene.append(resolved)
        if reason in resolution_counts:
            resolution_counts[reason] += 1
        elif reason == "direct":
            resolution_counts["direct"] += 1
        else:
            resolution_counts["unresolved"] += 1
    work["gene_id"] = resolved_gene

    keep = work["variant_id"] != ""
    dropped_missing_variant = 0
    if covered_variant_ids is not None:
        variant_ok = work["variant_id"].isin(covered_variant_ids)
        dropped_missing_variant = int((~variant_ok).sum())
        keep &= variant_ok

    gene_ok = work["gene_id"] != ""
    if covered_gene_ids is not None:
        gene_ok &= work["gene_id"].isin(covered_gene_ids)
    dropped_missing_gene = int((~gene_ok).sum())
    keep &= gene_ok

    filtered = work.loc[keep].copy()
    stats = {
        "input_rows": int(len(df)),
        "rows_after_prepare": int(len(filtered)),
        "dropped_missing_variant": dropped_missing_variant,
        "dropped_missing_gene": dropped_missing_gene,
        "gene_resolution_direct": int(resolution_counts["direct"]),
        "gene_resolution_variant_hint": int(resolution_counts["variant_hint"]),
        "gene_resolution_unique_token": int(resolution_counts["unique_token"]),
        "gene_resolution_preferred_symbol": int(resolution_counts["preferred_symbol"]),
        "gene_resolution_unresolved": int(resolution_counts["unresolved"]),
    }
    return filtered, stats


def select_domain_train_subset(
    domain_train_df: pd.DataFrame,
    per_label_cap: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if domain_train_df.empty:
        return domain_train_df.copy(), {
            "input_rows": 0,
            "output_rows": 0,
            "input_labels": 0,
            "output_labels": 0,
            "per_label_cap": int(per_label_cap),
            "dropped_rows": 0,
            "max_label_count_before": 0,
            "max_label_count_after": 0,
        }
    if "domain_map" not in domain_train_df.columns:
        raise ValueError("Domain subset selection requires domain_map column")

    out = domain_train_df.copy()
    label_counts_before = out["domain_map"].astype(int).value_counts().sort_index()
    cap = int(per_label_cap)
    if cap > 0:
        sampled_parts: List[pd.DataFrame] = []
        for _, group in out.groupby("domain_map", sort=False):
            if len(group) <= cap:
                sampled_parts.append(group)
            else:
                sampled_parts.append(group.sample(n=cap, random_state=seed))
        out = (
            pd.concat(sampled_parts, axis=0, ignore_index=True)
            if sampled_parts
            else out.iloc[0:0].copy()
        )
    label_counts_after = out["domain_map"].astype(int).value_counts().sort_index()
    stats = {
        "input_rows": int(len(domain_train_df)),
        "output_rows": int(len(out)),
        "input_labels": int(label_counts_before.shape[0]),
        "output_labels": int(label_counts_after.shape[0]),
        "per_label_cap": cap,
        "dropped_rows": int(len(domain_train_df) - len(out)),
        "max_label_count_before": int(label_counts_before.max()) if not label_counts_before.empty else 0,
        "max_label_count_after": int(label_counts_after.max()) if not label_counts_after.empty else 0,
    }
    print("domain_train_subset=" + json.dumps(stats, ensure_ascii=False))
    return out, stats


def select_func_train_subset(
    func_train_df: pd.DataFrame,
    min_valid_axes: int,
    per_gene_cap: int,
    seed: int,
) -> pd.DataFrame:
    if func_train_df.empty:
        return func_train_df.copy()

    out = func_train_df.copy()

    # 按轴统计有效分数数: 每个轴至少有 1 个有效分数则该轴 mask=1
    axis_valid_count = 0
    for axis_name, (start, end) in FUNC_AXIS_SLICES.items():
        axis_mask_cols = [f"{c}_mask" for c in FUNC_REGRESSION_COLS[start:end]]
        present = [c for c in axis_mask_cols if c in out.columns]
        if present:
            axis_valid_count += (out[present].sum(axis=1) > 0).astype(int)
    # mechanism 也算一个轴
    if "mechanism_mask" in out.columns:
        axis_valid_count += (out["mechanism_mask"] > 0).astype(int)

    out["_valid_axes"] = axis_valid_count
    out = out[out["_valid_axes"] >= float(min_valid_axes)].copy()
    after_quality = len(out)

    if per_gene_cap > 0:
        out = (
            out.groupby("gene_id", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), per_gene_cap), random_state=seed))
            .reset_index(drop=True)
        )

    out = out.drop(columns=["_valid_axes"])
    print(
        f"func_train_subset: min_valid_axes={min_valid_axes} "
        f"after_quality={after_quality} after_gene_cap={len(out)} cap={per_gene_cap}"
    )
    return out


def _split_items(
    items: Sequence[str],
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[str, str]:
    train_r, val_r, test_r = ratios
    total = train_r + val_r + test_r
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Keep deterministic ordering before RNG shuffle; avoid PYTHONHASHSEED effects.
    items = sorted(set(items))
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    mapping: Dict[str, str] = {}
    for item in items[:n_train]:
        mapping[item] = "train"
    for item in items[n_train : n_train + n_val]:
        mapping[item] = "val"
    for item in items[n_train + n_val :]:
        mapping[item] = "test"
    return mapping


def _quantile_bin(values: pd.Series, n_bins: int) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=int, index=values.index)
    n_eff = max(1, min(int(n_bins), int(values.nunique())))
    if n_eff <= 1:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    ranked = values.rank(method="first")
    binned = pd.qcut(ranked, q=n_eff, labels=False, duplicates="drop")
    out = pd.Series(np.asarray(binned), index=values.index).fillna(0).astype(int)
    return out



SPLIT_NAMES = ("train", "val", "test")
SIZE_BUCKET_LABELS = (
    "1",
    "2-5",
    "6-20",
    "21-100",
    "101-200",
    "201-400",
    "401-800",
    "801+",
)
SIZE_BUCKET_FIXED_SPLITS = {
    "801+": "train",
}
SIZE_BUCKET_SPLIT_CAPS = {
    "401-800": {"val": 1, "test": 1},
    "801+": {"val": 0, "test": 0},
}
GENE_HOLDOUT_V4_ATTEMPTS = 64
GENE_HOLDOUT_V4_MAX_HOLDOUT_PAIR_DEV = 0.18
GENE_HOLDOUT_V4_MAX_TOP1_SHARE = {
    "val": 0.12,
    "test": 0.10,
}
GENE_HOLDOUT_V4_MAX_TOP10_SHARE = {
    "val": 0.40,
    "test": 0.40,
}

DISEASE_BUCKET_LABELS = (
    "1",
    "2-5",
    "6-20",
    "21-50",
    "51-100",
    "101-200",
    "201-500",
    "501-1000",
    "1001-2000",
    "2001+",
)
DISEASE_BUCKET_FIXED_SPLITS = {
    "2001+": "train",
}
DISEASE_BUCKET_SPLIT_CAPS = {
    "1001-2000": {"val": 1, "test": 1},
    "2001+": {"val": 0, "test": 0},
}
DISEASE_HOLDOUT_V2_ATTEMPTS = 64
DISEASE_HOLDOUT_V2_MAX_HOLDOUT_PAIR_DEV = 0.25
DISEASE_HOLDOUT_V2_MAX_TOP1_SHARE = {
    "val": 0.22,
    "test": 0.22,
}
DISEASE_HOLDOUT_V2_MAX_TOP10_SHARE = {
    "val": 0.55,
    "test": 0.55,
}


def _variant_count_bucket_label(variant_count: int) -> str:
    n = int(variant_count)
    if n <= 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 20:
        return "6-20"
    if n <= 100:
        return "21-100"
    if n <= 200:
        return "101-200"
    if n <= 400:
        return "201-400"
    if n <= 800:
        return "401-800"
    return "801+"


def _disease_variant_count_bucket_label(variant_count: int) -> str:
    n = int(variant_count)
    if n <= 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 20:
        return "6-20"
    if n <= 50:
        return "21-50"
    if n <= 100:
        return "51-100"
    if n <= 200:
        return "101-200"
    if n <= 500:
        return "201-500"
    if n <= 1000:
        return "501-1000"
    if n <= 2000:
        return "1001-2000"
    return "2001+"


def _build_main_gene_balance_frame(
    main_df: pd.DataFrame,
    covered_variant_ids: Optional[Set[str]] = None,
    covered_gene_ids: Optional[Set[str]] = None,
    n_bins: int = 3,
) -> pd.DataFrame:
    del covered_variant_ids
    del covered_gene_ids
    del n_bins
    required_cols = {"gene_id", "variant_id"}
    if not required_cols.issubset(set(main_df.columns)):
        return pd.DataFrame()

    work = main_df[["gene_id", "variant_id"]].copy()
    work["gene_id"] = work["gene_id"].map(normalize_id)
    work["variant_id"] = work["variant_id"].map(normalize_id)
    work = work[(work["gene_id"] != "") & (work["variant_id"] != "")]
    if work.empty:
        return pd.DataFrame()

    gene_stats = work.groupby("gene_id", as_index=False).agg(
        variant_count=("variant_id", "nunique"),
        pair_count=("variant_id", "size"),
    )
    gene_stats["variant_count"] = gene_stats["variant_count"].astype(int)
    gene_stats["pair_count"] = gene_stats["pair_count"].astype(int)
    gene_stats["bucket"] = gene_stats["variant_count"].map(_variant_count_bucket_label)
    return gene_stats


def build_gene_size_buckets(
    main_df: pd.DataFrame,
    gene_to_idx: Dict[str, int],
) -> Dict[str, Set[int]]:
    required_cols = {"gene_id", "variant_id"}
    out: Dict[str, Set[int]] = {label: set() for label in SIZE_BUCKET_LABELS}
    if not required_cols.issubset(set(main_df.columns)):
        return out

    work = main_df[["gene_id", "variant_id"]].copy()
    work["gene_id"] = work["gene_id"].map(normalize_id)
    work["variant_id"] = work["variant_id"].map(normalize_id)
    work = work[(work["gene_id"] != "") & (work["variant_id"] != "")]
    if work.empty:
        return out

    gene_variant_counts = work.groupby("gene_id")["variant_id"].nunique().astype(int)
    for gene_id, variant_count in gene_variant_counts.items():
        gene_idx = gene_to_idx.get(str(gene_id))
        if gene_idx is None:
            continue
        bucket = _variant_count_bucket_label(int(variant_count))
        out.setdefault(bucket, set()).add(int(gene_idx))
    return out


def _ratio_targets(
    total: float,
    ratios: Tuple[float, float, float],
) -> Dict[str, float]:
    return {
        "train": float(total) * float(ratios[0]),
        "val": float(total) * float(ratios[1]),
        "test": float(total) * float(ratios[2]),
    }


def _summarize_gene_split_assignment(
    gene_stats_df: pd.DataFrame,
    gene_to_split: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    if gene_stats_df.empty:
        return {
            split: {
                "gene_count": 0,
                "pair_count": 0,
                "top1_gene_pair_share": 0.0,
                "top10_gene_pair_share": 0.0,
                "bucket_gene_counts": {},
                "bucket_pair_counts": {},
            }
            for split in SPLIT_NAMES
        }

    work = gene_stats_df[["gene_id", "pair_count", "bucket"]].copy()
    work["_split"] = work["gene_id"].map(gene_to_split)
    work = work.dropna(subset=["_split"]).copy()

    summary: Dict[str, Dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        split_df = work[work["_split"] == split].copy()
        pair_values = (
            split_df["pair_count"].astype(int).sort_values(ascending=False).tolist()
            if not split_df.empty
            else []
        )
        total_pairs = int(sum(pair_values))
        top1_share = float(pair_values[0] / total_pairs) if total_pairs > 0 else 0.0
        top10_share = (
            float(sum(pair_values[:10]) / total_pairs) if total_pairs > 0 else 0.0
        )
        bucket_gene_counts = (
            split_df.groupby("bucket")["gene_id"].nunique().astype(int).to_dict()
            if not split_df.empty
            else {}
        )
        bucket_pair_counts = (
            split_df.groupby("bucket")["pair_count"].sum().astype(int).to_dict()
            if not split_df.empty
            else {}
        )
        summary[split] = {
            "gene_count": int(len(split_df)),
            "pair_count": total_pairs,
            "top1_gene_pair_share": top1_share,
            "top10_gene_pair_share": top10_share,
            "bucket_gene_counts": {str(k): int(v) for k, v in bucket_gene_counts.items()},
            "bucket_pair_counts": {str(k): int(v) for k, v in bucket_pair_counts.items()},
        }
    return summary


def summarize_gene_holdout_split(
    main_df: pd.DataFrame,
    gene_split_map: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    gene_stats = _build_main_gene_balance_frame(main_df)
    if gene_stats.empty:
        return {}
    return _summarize_gene_split_assignment(gene_stats, gene_split_map)


def _build_main_disease_balance_frame(main_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"variant_id", "gene_id", "disease_index"}
    if not required_cols.issubset(set(main_df.columns)):
        return pd.DataFrame(
            columns=["disease_index", "variant_count", "pair_count", "gene_count", "bucket"]
        )

    work = main_df[["variant_id", "gene_id", "disease_index"]].copy()
    work["variant_id"] = work["variant_id"].map(normalize_id)
    work["gene_id"] = work["gene_id"].map(normalize_id)
    work["disease_index"] = pd.to_numeric(work["disease_index"], errors="coerce")
    work = work[
        (work["variant_id"] != "") & (work["gene_id"] != "") & work["disease_index"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=["disease_index", "variant_count", "pair_count", "gene_count", "bucket"]
        )
    work["disease_index"] = work["disease_index"].astype(int)

    stats = work.groupby("disease_index", as_index=False).agg(
        variant_count=("variant_id", "nunique"),
        pair_count=("variant_id", "size"),
        gene_count=("gene_id", "nunique"),
    )
    if stats.empty:
        return pd.DataFrame(
            columns=["disease_index", "variant_count", "pair_count", "gene_count", "bucket"]
        )
    stats["variant_count"] = stats["variant_count"].astype(int)
    stats["pair_count"] = stats["pair_count"].astype(int)
    stats["gene_count"] = stats["gene_count"].astype(int)
    stats["bucket"] = stats["variant_count"].map(_disease_variant_count_bucket_label)
    return stats


def _summarize_disease_split_assignment(
    disease_stats_df: pd.DataFrame,
    disease_to_split: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    if disease_stats_df.empty:
        return {
            split: {
                "disease_count": 0,
                "pair_count": 0,
                "top1_disease_pair_share": 0.0,
                "top10_disease_pair_share": 0.0,
                "bucket_disease_counts": {},
                "bucket_pair_counts": {},
            }
            for split in SPLIT_NAMES
        }

    work = disease_stats_df[["disease_index", "pair_count", "bucket"]].copy()
    work["_split"] = work["disease_index"].map(disease_to_split)
    work = work.dropna(subset=["_split"]).copy()

    summary: Dict[str, Dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        split_df = work[work["_split"] == split].copy()
        pair_values = (
            split_df["pair_count"].astype(int).sort_values(ascending=False).tolist()
            if not split_df.empty
            else []
        )
        total_pairs = int(sum(pair_values))
        top1_share = float(pair_values[0] / total_pairs) if total_pairs > 0 else 0.0
        top10_share = float(sum(pair_values[:10]) / total_pairs) if total_pairs > 0 else 0.0
        bucket_disease_counts = (
            split_df.groupby("bucket")["disease_index"].nunique().astype(int).to_dict()
            if not split_df.empty
            else {}
        )
        bucket_pair_counts = (
            split_df.groupby("bucket")["pair_count"].sum().astype(int).to_dict()
            if not split_df.empty
            else {}
        )
        summary[split] = {
            "disease_count": int(len(split_df)),
            "pair_count": total_pairs,
            "top1_disease_pair_share": top1_share,
            "top10_disease_pair_share": top10_share,
            "bucket_disease_counts": {
                str(k): int(v) for k, v in bucket_disease_counts.items()
            },
            "bucket_pair_counts": {str(k): int(v) for k, v in bucket_pair_counts.items()},
        }
    return summary


def summarize_disease_holdout_split(
    main_df: pd.DataFrame,
    disease_split_map: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    disease_stats = _build_main_disease_balance_frame(main_df)
    if disease_stats.empty:
        return {}
    return _summarize_disease_split_assignment(disease_stats, disease_split_map)


def _evaluate_disease_holdout_v2_candidate(
    disease_stats_df: pd.DataFrame,
    disease_to_split: Dict[int, str],
    ratios: Tuple[float, float, float],
) -> Tuple[float, bool, Dict[str, Dict[str, Any]]]:
    summary = _summarize_disease_split_assignment(disease_stats_df, disease_to_split)
    total_pairs = float(disease_stats_df["pair_count"].sum())
    total_diseases = float(len(disease_stats_df))
    target_pairs = _ratio_targets(total_pairs, ratios)
    target_diseases = _ratio_targets(total_diseases, ratios)

    penalty = 0.0
    valid = True
    for split in SPLIT_NAMES:
        pair_count = float(summary[split]["pair_count"])
        disease_count = float(summary[split]["disease_count"])
        pair_dev = abs(pair_count - target_pairs[split]) / max(1.0, target_pairs[split])
        disease_dev = abs(disease_count - target_diseases[split]) / max(1.0, target_diseases[split])
        penalty += pair_dev + 0.10 * disease_dev
        if split in {"val", "test"} and pair_dev > DISEASE_HOLDOUT_V2_MAX_HOLDOUT_PAIR_DEV:
            penalty += 20.0 * (pair_dev - DISEASE_HOLDOUT_V2_MAX_HOLDOUT_PAIR_DEV)
            valid = False

    for split, limit in DISEASE_HOLDOUT_V2_MAX_TOP1_SHARE.items():
        excess = float(summary[split]["top1_disease_pair_share"]) - float(limit)
        if excess > 0:
            penalty += 35.0 * excess
            valid = False
    for split, limit in DISEASE_HOLDOUT_V2_MAX_TOP10_SHARE.items():
        excess = float(summary[split]["top10_disease_pair_share"]) - float(limit)
        if excess > 0:
            penalty += 20.0 * excess
            valid = False

    for bucket, forced_split in DISEASE_BUCKET_FIXED_SPLITS.items():
        bucket_df = disease_stats_df[disease_stats_df["bucket"] == bucket]
        for disease_id in bucket_df["disease_index"].tolist():
            if disease_to_split.get(int(disease_id)) != forced_split:
                penalty += 100.0
                valid = False

    for bucket, caps in DISEASE_BUCKET_SPLIT_CAPS.items():
        bucket_df = disease_stats_df[disease_stats_df["bucket"] == bucket]
        if bucket_df.empty:
            continue
        split_counts = (
            bucket_df["disease_index"].map(disease_to_split).value_counts().astype(int).to_dict()
        )
        for split, cap in caps.items():
            count = int(split_counts.get(split, 0))
            if count > cap:
                penalty += 50.0 * float(count - cap)
                valid = False

    return penalty, valid, summary


def _balanced_split_disease_stats(
    disease_stats_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[int, str]:
    train_r, val_r, test_r = ratios
    total = train_r + val_r + test_r
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    if disease_stats_df.empty:
        return {}

    work = disease_stats_df.copy()
    work["disease_index"] = pd.to_numeric(work["disease_index"], errors="coerce")
    work = work[work["disease_index"].notna()].copy()
    if work.empty:
        return {}
    work["disease_index"] = work["disease_index"].astype(int)
    work["pair_count"] = pd.to_numeric(work["pair_count"], errors="coerce").fillna(1).astype(int)
    work["variant_count"] = pd.to_numeric(
        work["variant_count"], errors="coerce"
    ).fillna(work["pair_count"]).astype(int)

    bucket_names = sorted(set(work["bucket"].astype(str).tolist())) if "bucket" in work.columns else []
    bucket_label = bucket_names[0] if len(bucket_names) == 1 else ""
    fixed_split = DISEASE_BUCKET_FIXED_SPLITS.get(bucket_label)
    if fixed_split:
        return {int(disease_id): fixed_split for disease_id in sorted(work["disease_index"].tolist())}

    caps = DISEASE_BUCKET_SPLIT_CAPS.get(bucket_label, {})
    target_pairs = _ratio_targets(float(work["pair_count"].sum()), ratios)
    target_diseases = _ratio_targets(float(len(work)), ratios)
    split_pairs = {split: 0.0 for split in SPLIT_NAMES}
    split_diseases = {split: 0 for split in SPLIT_NAMES}
    mapping: Dict[int, str] = {}

    rng = random.Random(seed)
    rows: List[Tuple[int, int, int, float]] = []
    for row in work[["disease_index", "variant_count", "pair_count"]].itertuples(index=False):
        disease_id, variant_count, pair_count = row
        rows.append((int(disease_id), int(variant_count), int(pair_count), rng.random()))
    rows.sort(key=lambda x: (-x[2], -x[1], x[3], x[0]))

    for disease_id, variant_count, pair_count, _ in rows:
        best_split = "train"
        best_score: Optional[float] = None
        candidate_order = list(SPLIT_NAMES)
        rng.shuffle(candidate_order)
        for split in candidate_order:
            cap = caps.get(split)
            if cap is not None and split_diseases[split] >= int(cap):
                continue

            trial_pairs = dict(split_pairs)
            trial_diseases = dict(split_diseases)
            trial_pairs[split] += float(pair_count)
            trial_diseases[split] += 1

            score = 0.0
            for split_name in SPLIT_NAMES:
                pair_gap = (
                    (trial_pairs[split_name] - target_pairs[split_name])
                    / max(1.0, target_pairs[split_name])
                )
                disease_gap = (
                    (trial_diseases[split_name] - target_diseases[split_name])
                    / max(1.0, target_diseases[split_name])
                )
                score += pair_gap * pair_gap + 0.10 * disease_gap * disease_gap

            if split != "train":
                max_holdout_target = max(target_pairs["val"], target_pairs["test"])
                if float(pair_count) > max_holdout_target:
                    score += 0.5
                if variant_count >= 2000:
                    score += 1.0
            score += 0.01 * rng.random()

            if best_score is None or score < best_score - 1e-12:
                best_split = split
                best_score = score
            elif best_score is not None and abs(score - best_score) <= 1e-12:
                if SPLIT_NAMES.index(split) < SPLIT_NAMES.index(best_split):
                    best_split = split

        mapping[int(disease_id)] = best_split
        split_pairs[best_split] += float(pair_count)
        split_diseases[best_split] += 1

    return mapping


def _build_disease_holdout_v2_split(
    disease_stats_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[int, str]:
    if disease_stats_df.empty:
        return {}

    best_mapping: Dict[int, str] = {}
    best_penalty: Optional[float] = None

    for attempt in range(DISEASE_HOLDOUT_V2_ATTEMPTS):
        attempt_seed = seed + attempt * 1009
        mapping: Dict[int, str] = {}
        for i, bucket in enumerate(DISEASE_BUCKET_LABELS):
            bucket_df = disease_stats_df[disease_stats_df["bucket"] == bucket].copy()
            if bucket_df.empty:
                continue
            bucket_split = _balanced_split_disease_stats(
                bucket_df,
                seed=attempt_seed + 31 * (i + 1),
                ratios=ratios,
            )
            mapping.update(bucket_split)

        penalty, valid, _ = _evaluate_disease_holdout_v2_candidate(
            disease_stats_df=disease_stats_df,
            disease_to_split=mapping,
            ratios=ratios,
        )
        if valid:
            return mapping
        if best_penalty is None or penalty < best_penalty:
            best_mapping = mapping
            best_penalty = penalty

    return best_mapping


def _evaluate_gene_holdout_v4_candidate(
    gene_stats_df: pd.DataFrame,
    gene_to_split: Dict[str, str],
    ratios: Tuple[float, float, float],
) -> Tuple[float, bool, Dict[str, Dict[str, Any]]]:
    summary = _summarize_gene_split_assignment(gene_stats_df, gene_to_split)
    total_pairs = float(gene_stats_df["pair_count"].sum())
    total_genes = float(len(gene_stats_df))
    target_pairs = _ratio_targets(total_pairs, ratios)
    target_genes = _ratio_targets(total_genes, ratios)

    penalty = 0.0
    valid = True
    for split in SPLIT_NAMES:
        pair_count = float(summary[split]["pair_count"])
        gene_count = float(summary[split]["gene_count"])
        pair_dev = abs(pair_count - target_pairs[split]) / max(1.0, target_pairs[split])
        gene_dev = abs(gene_count - target_genes[split]) / max(1.0, target_genes[split])
        penalty += pair_dev + 0.15 * gene_dev
        if split in {"val", "test"} and pair_dev > GENE_HOLDOUT_V4_MAX_HOLDOUT_PAIR_DEV:
            penalty += 20.0 * (pair_dev - GENE_HOLDOUT_V4_MAX_HOLDOUT_PAIR_DEV)
            valid = False

    for split, limit in GENE_HOLDOUT_V4_MAX_TOP1_SHARE.items():
        excess = float(summary[split]["top1_gene_pair_share"]) - float(limit)
        if excess > 0:
            penalty += 40.0 * excess
            valid = False
    for split, limit in GENE_HOLDOUT_V4_MAX_TOP10_SHARE.items():
        excess = float(summary[split]["top10_gene_pair_share"]) - float(limit)
        if excess > 0:
            penalty += 25.0 * excess
            valid = False

    for bucket, forced_split in SIZE_BUCKET_FIXED_SPLITS.items():
        bucket_df = gene_stats_df[gene_stats_df["bucket"] == bucket]
        for gene_id in bucket_df["gene_id"].tolist():
            if gene_to_split.get(gene_id) != forced_split:
                penalty += 100.0
                valid = False

    for bucket, caps in SIZE_BUCKET_SPLIT_CAPS.items():
        bucket_df = gene_stats_df[gene_stats_df["bucket"] == bucket]
        if bucket_df.empty:
            continue
        split_counts = (
            bucket_df["gene_id"].map(gene_to_split).value_counts().astype(int).to_dict()
        )
        for split, cap in caps.items():
            count = int(split_counts.get(split, 0))
            if count > cap:
                penalty += 50.0 * float(count - cap)
                valid = False

    return penalty, valid, summary


def _balanced_split_gene_stats(
    gene_stats_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[str, str]:
    train_r, val_r, test_r = ratios
    total = train_r + val_r + test_r
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    if gene_stats_df.empty:
        return {}

    if "gene_id" not in gene_stats_df.columns:
        return {}

    work = gene_stats_df.copy()
    work["gene_id"] = work["gene_id"].map(normalize_id)
    work = work[work["gene_id"] != ""].copy()
    if work.empty:
        return {}
    if "pair_count" not in work.columns:
        work["pair_count"] = 1
    if "variant_count" not in work.columns:
        work["variant_count"] = 1
    work["pair_count"] = pd.to_numeric(work["pair_count"], errors="coerce").fillna(1).astype(int)
    work["variant_count"] = (
        pd.to_numeric(work["variant_count"], errors="coerce").fillna(1).astype(int)
    )

    bucket_names = sorted(set(work["bucket"].astype(str).tolist())) if "bucket" in work.columns else []
    bucket_label = bucket_names[0] if len(bucket_names) == 1 else ""
    fixed_split = SIZE_BUCKET_FIXED_SPLITS.get(bucket_label)
    if fixed_split:
        return {gene_id: fixed_split for gene_id in sorted(work["gene_id"].tolist())}

    caps = SIZE_BUCKET_SPLIT_CAPS.get(bucket_label, {})
    target_pairs = _ratio_targets(float(work["pair_count"].sum()), ratios)
    target_genes = _ratio_targets(float(len(work)), ratios)
    split_pairs = {split: 0.0 for split in SPLIT_NAMES}
    split_genes = {split: 0 for split in SPLIT_NAMES}
    mapping: Dict[str, str] = {}

    rng = random.Random(seed)
    rows: List[Tuple[str, int, int, float]] = []
    for row in work[["gene_id", "variant_count", "pair_count"]].itertuples(index=False):
        gene_id, variant_count, pair_count = row
        rows.append((str(gene_id), int(variant_count), int(pair_count), rng.random()))
    rows.sort(key=lambda x: (-x[2], -x[1], x[3], x[0]))

    for gene_id, variant_count, pair_count, _ in rows:
        best_split = "train"
        best_score: Optional[float] = None
        candidate_order = list(SPLIT_NAMES)
        rng.shuffle(candidate_order)
        for split in candidate_order:
            cap = caps.get(split)
            if cap is not None and split_genes[split] >= int(cap):
                continue

            trial_pairs = dict(split_pairs)
            trial_genes = dict(split_genes)
            trial_pairs[split] += float(pair_count)
            trial_genes[split] += 1

            score = 0.0
            for split_name in SPLIT_NAMES:
                pair_gap = (
                    (trial_pairs[split_name] - target_pairs[split_name])
                    / max(1.0, target_pairs[split_name])
                )
                gene_gap = (
                    (trial_genes[split_name] - target_genes[split_name])
                    / max(1.0, target_genes[split_name])
                )
                score += pair_gap * pair_gap + 0.15 * gene_gap * gene_gap

            if split != "train":
                max_holdout_target = max(target_pairs["val"], target_pairs["test"])
                if float(pair_count) > max_holdout_target:
                    score += 0.5
                if variant_count >= 800:
                    score += 1.0
            score += 0.01 * rng.random()

            if best_score is None or score < best_score - 1e-12:
                best_split = split
                best_score = score
            elif best_score is not None and abs(score - best_score) <= 1e-12:
                if SPLIT_NAMES.index(split) < SPLIT_NAMES.index(best_split):
                    best_split = split

        mapping[gene_id] = best_split
        split_pairs[best_split] += float(pair_count)
        split_genes[best_split] += 1

    return mapping


def _build_gene_holdout_size_v4_split(
    gene_stats_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[str, str]:
    if gene_stats_df.empty:
        return {}

    best_mapping: Dict[str, str] = {}
    best_penalty: Optional[float] = None

    for attempt in range(GENE_HOLDOUT_V4_ATTEMPTS):
        attempt_seed = seed + attempt * 1009
        mapping: Dict[str, str] = {}
        for i, bucket in enumerate(SIZE_BUCKET_LABELS):
            bucket_df = gene_stats_df[gene_stats_df["bucket"] == bucket].copy()
            if bucket_df.empty:
                continue
            bucket_split = _balanced_split_gene_stats(
                bucket_df,
                seed=attempt_seed + 31 * (i + 1),
                ratios=ratios,
            )
            mapping.update(bucket_split)

        penalty, valid, _ = _evaluate_gene_holdout_v4_candidate(
            gene_stats_df=gene_stats_df,
            gene_to_split=mapping,
            ratios=ratios,
        )
        if valid:
            return mapping
        if best_penalty is None or penalty < best_penalty:
            best_mapping = mapping
            best_penalty = penalty

    return best_mapping


def load_embedding_id_set(path: str) -> Set[str]:
    p = Path(path)
    index_npy = p.parent / f"{p.stem}_index.npy"
    if index_npy.exists():
        ids = np.load(index_npy, allow_pickle=True)
        ids_norm = np.char.lower(np.char.strip(ids.astype(str)))
        return {str(v) for v in ids_norm.tolist() if str(v)}

    first_col = pd.read_csv(path, usecols=[0]).iloc[:, 0].map(normalize_id)
    return {v for v in first_col.tolist() if v}


def build_main_split_coverage_sets(
    variant_paths: Sequence[str],
    protein_paths: Sequence[str],
    gene_path: str,
) -> Tuple[Set[str], Set[str]]:
    variant_ids: Set[str] = set()
    protein_ids: Set[str] = set()
    for path in variant_paths:
        variant_ids.update(load_embedding_id_set(path))
    for path in protein_paths:
        protein_ids.update(load_embedding_id_set(path))
    gene_ids = load_embedding_id_set(gene_path)
    return variant_ids & protein_ids, gene_ids


def build_within_gene_variant_split(
    main_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    func_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Variant-level split constrained within each gene bucket.

    For each gene, variants are split into train/val/test by ratios.
    Genes with only one variant are assigned by gene-level random split to keep determinism.
    """
    dfs = [main_df, domain_df, aux_df, func_df]
    all_rows = []
    for df in dfs:
        if "variant_id" not in df.columns or "gene_id" not in df.columns:
            continue
        sub = df[["variant_id", "gene_id"]].copy()
        sub["variant_id"] = sub["variant_id"].map(normalize_id)
        sub["gene_id"] = sub["gene_id"].map(normalize_id)
        sub = sub[(sub["variant_id"] != "") & (sub["gene_id"] != "")]
        if not sub.empty:
            all_rows.append(sub)

    if not all_rows:
        return {}, {}

    merged = pd.concat(all_rows, axis=0).drop_duplicates()
    gene_to_variants: Dict[str, List[str]] = {}
    for row in merged.itertuples(index=False):
        v, g = row
        gene_to_variants.setdefault(g, []).append(v)
    gene_to_variants = {g: sorted(set(vs)) for g, vs in gene_to_variants.items()}

    variant_to_split: Dict[str, str] = {}
    gene_to_split: Dict[str, str] = {}
    rng = random.Random(seed)
    all_genes = sorted(gene_to_variants.keys())
    gene_fallback = _split_items(all_genes, seed + 17, ratios)
    for g in all_genes:
        variants = gene_to_variants[g]
        if len(variants) <= 1:
            split = gene_fallback[g]
            for v in variants:
                variant_to_split[v] = split
            gene_to_split[g] = split
            continue

        tmp = variants.copy()
        rng.shuffle(tmp)
        n = len(tmp)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1])) if n >= 3 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        for v in tmp[:n_train]:
            variant_to_split[v] = "train"
        for v in tmp[n_train : n_train + n_val]:
            variant_to_split[v] = "val"
        for v in tmp[n_train + n_val :]:
            variant_to_split[v] = "test"
        # Majority split is used as gene tag for reporting only.
        c_train = sum(1 for v in variants if variant_to_split[v] == "train")
        c_val = sum(1 for v in variants if variant_to_split[v] == "val")
        c_test = sum(1 for v in variants if variant_to_split[v] == "test")
        gene_to_split[g] = max(
            [("train", c_train), ("val", c_val), ("test", c_test)],
            key=lambda x: x[1],
        )[0]

    return variant_to_split, gene_to_split


def _stable_dict_hash(obj: Dict[str, Any]) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _split_artifact_identity_payload(
    protocol: Any,
    seed: Any,
    ratios: Sequence[Any],
    split_map: Dict[str, str],
    gene_split_map: Optional[Dict[str, str]],
    disease_split_map: Optional[Dict[int, str]],
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "protocol": str(protocol),
        "seed": int(seed) if seed is not None else -1,
        "ratios": [float(r) for r in ratios],
        "split_map": {normalize_id(k): str(v) for k, v in split_map.items() if normalize_id(k)},
        "gene_split_map": {
            normalize_id(k): str(v)
            for k, v in (gene_split_map or {}).items()
            if normalize_id(k)
        },
        "disease_split_map": {
            str(int(k)): str(v) for k, v in (disease_split_map or {}).items()
        },
        "metadata": metadata or {},
    }


def build_split_artifact(
    split_map: Dict[str, str],
    gene_split_map: Optional[Dict[str, str]],
    disease_split_map: Optional[Dict[int, str]],
    protocol: str,
    seed: int,
    ratios: Tuple[float, float, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    clean_split = {normalize_id(k): str(v) for k, v in split_map.items() if normalize_id(k)}
    clean_gene = {
        normalize_id(k): str(v) for k, v in (gene_split_map or {}).items() if normalize_id(k)
    }
    payload = {
        "protocol": protocol,
        "seed": int(seed),
        "ratios": list(ratios),
        "split_map": clean_split,
        "gene_split_map": clean_gene,
        "disease_split_map": {str(int(k)): str(v) for k, v in (disease_split_map or {}).items()},
        "metadata": metadata or {},
    }
    payload["artifact_id"] = _stable_dict_hash(
        _split_artifact_identity_payload(
            protocol=payload["protocol"],
            seed=payload["seed"],
            ratios=payload["ratios"],
            split_map=clean_split,
            gene_split_map=clean_gene,
            disease_split_map=disease_split_map,
            metadata=payload["metadata"],
        )
    )
    return payload


def save_split_artifact(artifact: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def load_split_artifact(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split artifact not found: {path}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if "split_map" not in raw:
        raise ValueError(f"Invalid split artifact (missing split_map): {path}")
    raw["split_map"] = {normalize_id(k): v for k, v in raw["split_map"].items()}
    raw["gene_split_map"] = {
        normalize_id(k): v for k, v in raw.get("gene_split_map", {}).items()
    }
    raw["disease_split_map"] = {
        int(k): v for k, v in raw.get("disease_split_map", {}).items()
    }
    raw["artifact_id"] = _stable_dict_hash(
        _split_artifact_identity_payload(
            protocol=raw.get("protocol", ""),
            seed=raw.get("seed"),
            ratios=raw.get("ratios", []),
            split_map=raw["split_map"],
            gene_split_map=raw["gene_split_map"],
            disease_split_map=raw.get("disease_split_map", {}),
            metadata=raw.get("metadata", {}),
        )
    )
    return raw


def validate_split_consistency(
    split_map: Dict[str, str],
    task_splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    strict_main_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    overlap = compute_train_test_overlap(task_splits)
    if overlap["train_test_overlap"] != 0:
        raise ValueError(
            f"Split leakage detected: train_test_overlap={overlap['train_test_overlap']}"
        )

    if "main" in task_splits and strict_main_counts is not None:
        train_df, val_df, test_df = task_splits["main"]
        curr = {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        }
        if curr != strict_main_counts:
            raise ValueError(
                f"Main split counts mismatch: expected={strict_main_counts}, got={curr}"
            )

    return {
        "split_size": len(split_map),
        **overlap,
    }


def build_protocol_id(
    protocol: str,
    seed: int,
    ratios: Tuple[float, float, float],
    artifact_id: str,
    graph_visibility: str,
) -> str:
    return (
        f"{protocol}_s{seed}_r{ratios[0]:.2f}-{ratios[1]:.2f}-{ratios[2]:.2f}_"
        f"{graph_visibility}_{artifact_id}"
    )


def _build_disease_split_map(
    main_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[int, str]:
    if "disease_index" not in main_df.columns:
        raise ValueError("disease_holdout requires disease_index in main dataframe")
    disease_stats = _build_main_disease_balance_frame(main_df)
    if disease_stats.empty:
        return {}
    return _build_disease_holdout_v2_split(disease_stats, seed=seed, ratios=ratios)


def _validate_disease_holdout_disjoint(
    main_split: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    disease_to_split: Dict[int, str],
) -> None:
    train_df, val_df, test_df = main_split
    assigned = {int(d): str(s) for d, s in disease_to_split.items()}
    for split_name, df in {"train": train_df, "val": val_df, "test": test_df}.items():
        diseases = pd.to_numeric(df["disease_index"], errors="coerce").dropna().astype(int).unique().tolist()
        bad = sorted([d for d in diseases if assigned.get(int(d)) != split_name])
        if bad:
            sample = bad[:10]
            raise ValueError(
                "Disease split mismatch detected in disease_holdout main split: "
                f"split={split_name} mismatched={len(bad)} sample={sample}"
            )


def apply_disease_split(
    df: pd.DataFrame,
    disease_col: str,
    disease_split_map: Dict[int, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    diseases = pd.to_numeric(work[disease_col], errors="coerce")
    work["_split"] = diseases.map(disease_split_map)
    work = work.dropna(subset=["_split"])

    train_df = work[work["_split"] == "train"].drop(columns=["_split"])
    val_df = work[work["_split"] == "val"].drop(columns=["_split"])
    test_df = work[work["_split"] == "test"].drop(columns=["_split"])
    return train_df, val_df, test_df


def build_disease_holdout_split(
    main_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    func_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[int, str]]:
    variant_to_split: Dict[str, str] = {}
    gene_to_split: Dict[str, str] = {}

    disease_to_split = _build_disease_split_map(main_df, seed=seed, ratios=ratios)
    if not disease_to_split:
        raise ValueError("No diseases found for disease_holdout split construction")

    main_work = main_df[["variant_id", "gene_id", "disease_index"]].copy()
    main_work["variant_id"] = main_work["variant_id"].map(normalize_id)
    main_work["gene_id"] = main_work["gene_id"].map(normalize_id)
    main_work["disease_index"] = pd.to_numeric(main_work["disease_index"], errors="coerce")
    main_work = main_work[
        (main_work["variant_id"] != "") & (main_work["gene_id"] != "") & main_work["disease_index"].notna()
    ].copy()
    main_work["disease_index"] = main_work["disease_index"].astype(int)

    variant_disease_splits: Dict[str, Set[str]] = {}
    for v, _, d in main_work.itertuples(index=False):
        split = disease_to_split.get(int(d))
        if split is None:
            continue
        variant_disease_splits.setdefault(v, set()).add(split)

    for v, splits in variant_disease_splits.items():
        if "test" in splits:
            variant_to_split[v] = "test"
        elif "val" in splits:
            variant_to_split[v] = "val"
        else:
            variant_to_split[v] = "train"

    split_priority = {"train": 0, "val": 1, "test": 2}
    gene_vote: Dict[str, Dict[str, int]] = {}
    for v, g in main_work[["variant_id", "gene_id"]].itertuples(index=False):
        split = variant_to_split.get(v)
        if not g or split is None:
            continue
        vote = gene_vote.setdefault(g, {"train": 0, "val": 0, "test": 0})
        vote[split] += 1
    for g, vote in gene_vote.items():
        gene_to_split[g] = max(
            vote.items(),
            key=lambda kv: (kv[1], split_priority[kv[0]]),
        )[0]

    for i, extra_df in enumerate([domain_df, aux_df, func_df], start=1):
        aux_genes = [g for g in extra_df["gene_id"].dropna().unique().tolist() if g]
        new_genes = [g for g in aux_genes if g not in gene_to_split]
        if new_genes:
            aux_split = _split_items(new_genes, seed + i, ratios)
            gene_to_split.update(aux_split)

    unresolved_variants: Set[str] = set()
    blocked_variants: Set[str] = set()
    for df in [main_df, domain_df, aux_df, func_df]:
        for v, g in df[["variant_id", "gene_id"]].itertuples(index=False):
            if not v:
                continue
            prev = variant_to_split.get(v)
            if prev is not None:
                continue
            split = gene_to_split.get(g) if g else None
            if split is None:
                unresolved_variants.add(v)
                continue
            variant_to_split[v] = split

    unresolved_variants = unresolved_variants - set(variant_to_split.keys()) - blocked_variants
    if unresolved_variants:
        variant_split = _split_items(sorted(unresolved_variants), seed + 99, ratios)
        variant_to_split.update(variant_split)

    main_split = apply_disease_split(main_df, "disease_index", disease_to_split)
    _validate_disease_holdout_disjoint(main_split, disease_to_split)
    return variant_to_split, gene_to_split, disease_to_split


def build_global_variant_split(
    main_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    func_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float],
    protocol: str = "gene_holdout",
    return_gene_split: bool = False,
    covered_variant_ids: Optional[Set[str]] = None,
    covered_gene_ids: Optional[Set[str]] = None,
) -> Dict[str, str] | Tuple[Dict[str, str], Dict[str, str]]:
    if protocol not in {"gene_holdout", "disease_holdout"}:
        raise ValueError(f"Unsupported protocol for build_global_variant_split: {protocol}")

    dfs = [main_df, domain_df, aux_df, func_df]
    for df in dfs:
        if "variant_id" not in df.columns or "gene_id" not in df.columns:
            raise ValueError("All task dataframes must contain variant_id and gene_id")

    variant_to_split: Dict[str, str] = {}
    gene_to_split: Dict[str, str] = {}
    unresolved_variants: Set[str] = set()
    blocked_variants: Set[str] = set()

    if protocol == "disease_holdout":
        variant_to_split, gene_to_split, _ = build_disease_holdout_split(
            main_df=main_df,
            domain_df=domain_df,
            aux_df=aux_df,
            func_df=func_df,
            seed=seed,
            ratios=ratios,
        )
        if return_gene_split:
            return variant_to_split, gene_to_split
        return variant_to_split
    else:
        main_genes = sorted([g for g in main_df["gene_id"].dropna().unique().tolist() if g])
        main_gene_set = set(main_genes)
        gene_stats = _build_main_gene_balance_frame(
            main_df,
            covered_variant_ids=covered_variant_ids,
            covered_gene_ids=covered_gene_ids,
            n_bins=3,
        )
        if not gene_stats.empty:
            gene_stats = gene_stats[gene_stats["gene_id"].isin(main_gene_set)].copy()
            main_gene_split = _build_gene_holdout_size_v4_split(
                gene_stats_df=gene_stats,
                seed=seed,
                ratios=ratios,
            )
            gene_to_split.update(main_gene_split)
        remaining_main_genes = [g for g in main_genes if g not in gene_to_split]
        if remaining_main_genes:
            fallback_split = _split_items(remaining_main_genes, seed + 997, ratios)
            gene_to_split.update(fallback_split)

        conflict_variants: Set[str] = set()
        for df in dfs:
            for v, g in df[["variant_id", "gene_id"]].itertuples(index=False):
                if not v:
                    continue
                split = gene_to_split.get(g) if g else None
                if split is None:
                    unresolved_variants.add(v)
                    continue
                prev = variant_to_split.get(v)
                if prev is not None and prev != split:
                    conflict_variants.add(v)
                    continue
                variant_to_split[v] = split

        for v in conflict_variants:
            variant_to_split.pop(v, None)
        blocked_variants.update(conflict_variants)
        unresolved_variants = unresolved_variants - set(variant_to_split.keys())
        unresolved_variants = unresolved_variants - conflict_variants

    for i, extra_df in enumerate([domain_df, aux_df, func_df], start=1):
        aux_genes = [g for g in extra_df["gene_id"].dropna().unique().tolist() if g]
        new_genes = [g for g in aux_genes if g not in gene_to_split]
        if new_genes:
            aux_split = _split_items(new_genes, seed + i, ratios)
            gene_to_split.update(aux_split)

    for df in dfs:
        for v, g in df[["variant_id", "gene_id"]].itertuples(index=False):
            if not v or v in variant_to_split or v in blocked_variants:
                continue
            split = gene_to_split.get(g) if g else None
            if split is None:
                unresolved_variants.add(v)
                continue
            variant_to_split[v] = split

    unresolved_variants = unresolved_variants - set(variant_to_split.keys())
    unresolved_variants = unresolved_variants - blocked_variants
    if unresolved_variants:
        variant_split = _split_items(sorted(unresolved_variants), seed + 99, ratios)
        variant_to_split.update(variant_split)

    if return_gene_split:
        return variant_to_split, gene_to_split
    return variant_to_split


def apply_split(
    df: pd.DataFrame,
    variant_col_or_index: str,
    split_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if variant_col_or_index == "index":
        variants = work.index.map(normalize_id)
    else:
        variants = work[variant_col_or_index].map(normalize_id)
    work["_split"] = variants.map(split_map)
    work = work.dropna(subset=["_split"])

    train_df = work[work["_split"] == "train"].drop(columns=["_split"])
    val_df = work[work["_split"] == "val"].drop(columns=["_split"])
    test_df = work[work["_split"] == "test"].drop(columns=["_split"])
    return train_df, val_df, test_df


def build_mappings(
    gene_x_df: pd.DataFrame,
    trait_x_df: pd.DataFrame,
    disease_df: pd.DataFrame,
) -> Dict[str, Dict[Any, Any]]:
    gene_ids = [normalize_id(v) for v in gene_x_df.index.tolist()]
    trait_ids = [normalize_id(v) for v in trait_x_df.index.tolist()]

    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    trait_to_idx = {t: i for i, t in enumerate(trait_ids)}

    disease_ids = sorted(disease_df["disease_index"].astype(int).unique().tolist())
    disease_to_idx = {d: i for i, d in enumerate(disease_ids)}

    return {
        "gene_to_idx": gene_to_idx,
        "idx_to_gene": {i: g for g, i in gene_to_idx.items()},
        "trait_to_idx": trait_to_idx,
        "idx_to_trait": {i: t for t, i in trait_to_idx.items()},
        "disease_to_idx": disease_to_idx,
        "idx_to_disease": {i: d for d, i in disease_to_idx.items()},
    }


def _edges_from_file(
    edge_path: str,
    src_mapping: Dict[str, int],
    dst_mapping: Dict[str, int],
) -> torch.Tensor:
    df = pd.read_csv(edge_path)
    if len(df.columns) < 2:
        raise ValueError(f"Edge file {edge_path} must have at least 2 columns")
    c0, c1 = df.columns[:2]
    src = df[c0].map(normalize_id)
    dst = df[c1].map(normalize_id)

    src_idx = src.map(src_mapping)
    dst_idx = dst.map(dst_mapping)
    valid = src_idx.notna() & dst_idx.notna()

    src_vals = src_idx[valid].astype(np.int64).to_numpy()
    dst_vals = dst_idx[valid].astype(np.int64).to_numpy()
    if len(src_vals) == 0:
        raise ValueError(f"No valid edges in {edge_path}")

    return torch.tensor(np.stack([src_vals, dst_vals], axis=0), dtype=torch.long)


def build_hetero_graph(
    gene_x_df: pd.DataFrame,
    trait_x_df: pd.DataFrame,
    edge_files: Dict[str, str],
    gene_mapping: Dict[str, int],
    trait_mapping: Dict[str, int],
) -> HeteroData:
    if _PYG_IMPORT_ERROR is not None:
        raise ImportError(
            "torch_geometric is required for graph construction"
        ) from _PYG_IMPORT_ERROR

    data = HeteroData()
    data["gene"].x = torch.tensor(gene_x_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    data["trait"].x = torch.tensor(
        trait_x_df.to_numpy(dtype=np.float32), dtype=torch.float32
    )

    data[("gene", "to", "gene")].edge_index = _edges_from_file(
        edge_files["gene_to_gene"], gene_mapping, gene_mapping
    )
    data[("gene", "to", "trait")].edge_index = _edges_from_file(
        edge_files["gene_to_trait"], gene_mapping, trait_mapping
    )
    data[("trait", "to", "trait")].edge_index = _edges_from_file(
        edge_files["trait_to_trait"], trait_mapping, trait_mapping
    )

    data = ToUndirected()(data)
    return data


def build_inductive_train_graph(
    full_graph: HeteroData,
    train_gene_indices: Set[int],
) -> HeteroData:
    """Keep only edges connected to train genes for graph relations touching gene nodes."""
    graph = deepcopy(full_graph)
    if not train_gene_indices:
        return graph

    gene_keep = torch.zeros(graph["gene"].x.shape[0], dtype=torch.bool)
    gene_keep[list(train_gene_indices)] = True

    for edge_type in list(graph.edge_index_dict.keys()):
        src_type, _, dst_type = edge_type
        edge_index = graph[edge_type].edge_index
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        if src_type == "gene":
            mask = mask & gene_keep[edge_index[0]]
        if dst_type == "gene":
            mask = mask & gene_keep[edge_index[1]]
        graph[edge_type].edge_index = edge_index[:, mask]

    return graph


def build_disease_to_traits_map(
    disease_df: pd.DataFrame,
    trait_mapping: Dict[str, int],
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for row in disease_df[["disease_index", "hpo_ids"]].itertuples(index=False):
        disease_id, raw = int(row[0]), row[1]
        trait_ids = [trait_mapping[t] for t in parse_hpo_ids(raw) if t in trait_mapping]
        if trait_ids:
            out[disease_id] = sorted(set(trait_ids))
    return out



@dataclass
class FeatureStore:
    variant_to_idx: Dict[str, int]
    idx_to_variant: Dict[int, str]
    variant_x: torch.Tensor
    protein_x: torch.Tensor


@dataclass
class VariantDiseaseKLTeacher:
    variant_to_disease: Dict[int, Tuple[np.ndarray, np.ndarray]]
    disease_to_variant: Dict[int, Tuple[np.ndarray, np.ndarray]]
    pool_variant_idx: np.ndarray
    pool_gene_idx: np.ndarray
    variant_idx_to_pool: Dict[int, int]
    variant_mapped_mass: Dict[int, float]
    variant_kl_weight: Dict[int, float]  # 1.0 for direct-mapped, alpha for gene-propagated
    stats: Dict[str, float]


def build_feature_store(
    variant_x_df: pd.DataFrame,
    protein_x_df: pd.DataFrame,
) -> FeatureStore:
    shared_ids = sorted(set(variant_x_df.index.tolist()) & set(protein_x_df.index.tolist()))
    if not shared_ids:
        raise ValueError("No overlapping variant ids between variant_x and protein_x")

    v_df = variant_x_df.loc[shared_ids]
    p_df = protein_x_df.loc[shared_ids]

    # Preserve a shared per-variant row index so variant_ids can address both tensors.
    variant_to_idx = {v: i for i, v in enumerate(shared_ids)}
    idx_to_variant = {i: v for v, i in variant_to_idx.items()}
    variant_x = torch.tensor(v_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    protein_x = torch.tensor(p_df.to_numpy(dtype=np.float32), dtype=torch.float32)

    return FeatureStore(
        variant_to_idx=variant_to_idx,
        idx_to_variant=idx_to_variant,
        variant_x=variant_x,
        protein_x=protein_x,
    )


def _resolve_optional_relative_path(path_text: str, base_dir: Path) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_topk_memmap_arrays(
    indices_path: str,
    values_path: str,
    metadata_path: str,
) -> Tuple[np.memmap, np.memmap, Dict[str, Any]]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n_variants = int(meta["n_variants"])
    topk = int(meta["topk"])
    value_dtype = np.float16 if str(meta.get("dtype", "float16")) == "float16" else np.float32
    idx_mm = np.memmap(indices_path, dtype=np.int32, mode="r", shape=(n_variants, topk))
    val_mm = np.memmap(values_path, dtype=value_dtype, mode="r", shape=(n_variants, topk))
    return idx_mm, val_mm, meta


def _add_concept_disease_edge(
    concept_to_disease_raw: Dict[int, Dict[int, float]],
    concept_idx: int,
    disease_col: int,
    weight: float,
) -> None:
    if weight <= 0 or not np.isfinite(weight):
        return
    disease_map = concept_to_disease_raw.setdefault(int(concept_idx), {})
    disease_map[int(disease_col)] = disease_map.get(int(disease_col), 0.0) + float(weight)


def _load_concept_to_disease_from_csv(
    disease_concept_map_csv: str,
    disease_to_col: Dict[int, int],
    direct_weight: float,
    concept_hpo_structural: Optional[Dict[int, int]] = None,
    disease_hpo_count: Optional[Dict[int, int]] = None,
    bridge_length_penalty_b: float = 0.0,
) -> Tuple[Dict[int, Dict[int, float]], Dict[str, float]]:
    concept_df = pd.read_csv(disease_concept_map_csv)
    required_cols = {"disease_index", "concept_idx"}
    if not required_cols.issubset(set(concept_df.columns)):
        raise ValueError(
            f"disease_concept_map_csv missing required columns: {required_cols}"
        )
    if "score" not in concept_df.columns:
        concept_df["score"] = 1.0

    out: Dict[int, Dict[int, float]] = {}
    n_rows_used = 0
    n_bridge_penalized = 0
    for row in concept_df[["disease_index", "concept_idx", "score"]].itertuples(index=False):
        disease_index, concept_idx, score = row
        d_col = disease_to_col.get(int(disease_index))
        if d_col is None:
            continue
        w = float(score) * float(direct_weight)
        if bridge_length_penalty_b > 0 and concept_hpo_structural is not None and disease_hpo_count is not None:
            c_hpo_count = concept_hpo_structural.get(int(concept_idx))
            if c_hpo_count is not None:
                d_hpo_count = disease_hpo_count.get(int(disease_index), 1)
                min_set = min(c_hpo_count, d_hpo_count)
                avg_set_size = 10.0
                length_penalty = (1.0 - bridge_length_penalty_b) + bridge_length_penalty_b * (min_set / avg_set_size)
                w *= length_penalty
                n_bridge_penalized += 1
        if w <= 0 or not np.isfinite(w):
            continue
        _add_concept_disease_edge(out, int(concept_idx), int(d_col), w)
        n_rows_used += 1

    stats = {
        "csv_rows_used": float(n_rows_used),
        "csv_mapped_concepts": float(len(out)),
        "csv_bridge_penalized": float(n_bridge_penalized),
    }
    return out, stats


def _load_concept_to_disease_from_hpo_semantic(
    disease_hpo_map: Dict[int, List[str]],
    disease_to_col: Dict[int, int],
    hpo_semantic_map_json: str,
    topk_per_hpo: int,
    min_similarity: float,
    semantic_weight: float,
    n_concepts: int,
    hpo_min_terms_for_full_weight: int = 0,
) -> Tuple[Dict[int, Dict[int, float]], Dict[str, float]]:
    out: Dict[int, Dict[int, float]] = {}
    if not hpo_semantic_map_json:
        return out, {
            "hpo_edges_used": 0.0,
            "hpo_mapped_diseases": 0.0,
            "hpo_mapped_concepts": 0.0,
            "hpo_json_found": 0.0,
        }
    path = Path(hpo_semantic_map_json)
    if (not path.exists()) or (not path.is_file()):
        return out, {
            "hpo_edges_used": 0.0,
            "hpo_mapped_diseases": 0.0,
            "hpo_mapped_concepts": 0.0,
            "hpo_json_found": 0.0,
        }
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mapping = payload.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}

    topk_per_hpo = max(int(topk_per_hpo), 1)
    min_similarity = float(min_similarity)
    semantic_weight = float(semantic_weight)
    n_edges = 0
    n_hpo_discounted = 0
    mapped_diseases: Set[int] = set()
    for disease_idx, hpo_terms in disease_hpo_map.items():
        d_col = disease_to_col.get(int(disease_idx))
        if d_col is None:
            continue
        n_hpo = len(hpo_terms)
        if hpo_min_terms_for_full_weight > 0:
            hpo_discount = min(1.0, n_hpo / float(hpo_min_terms_for_full_weight))
            if hpo_discount < 1.0:
                n_hpo_discounted += 1
        else:
            hpo_discount = 1.0
        any_edge_for_disease = False
        for hpo in hpo_terms:
            hpo_key = str(hpo).strip()
            entries = (
                mapping.get(hpo_key)
                or mapping.get(hpo_key.upper())
                or mapping.get(hpo_key.lower())
                or []
            )
            if not isinstance(entries, list) or not entries:
                continue
            kept = sorted(
                entries,
                key=lambda x: float(x.get("similarity", 0.0)),
                reverse=True,
            )[:topk_per_hpo]
            for item in kept:
                sim = float(item.get("similarity", 0.0))
                if not np.isfinite(sim) or sim < min_similarity:
                    continue
                concept_idx = item.get("phenotype_index")
                if concept_idx is None:
                    continue
                c_idx = int(concept_idx)
                if c_idx < 0 or c_idx >= int(n_concepts):
                    continue
                _add_concept_disease_edge(
                    out,
                    c_idx,
                    int(d_col),
                    float(semantic_weight) * sim * hpo_discount,
                )
                n_edges += 1
                any_edge_for_disease = True
        if any_edge_for_disease:
            mapped_diseases.add(int(disease_idx))

    stats = {
        "hpo_edges_used": float(n_edges),
        "hpo_mapped_diseases": float(len(mapped_diseases)),
        "hpo_mapped_concepts": float(len(out)),
        "hpo_json_found": 1.0,
        "hpo_discounted_diseases": float(n_hpo_discounted),
    }
    return out, stats


def _merge_concept_to_disease_maps(
    maps: Sequence[Dict[int, Dict[int, float]]],
    max_diseases_per_concept: int = 0,
    concept_specificity_weighting: bool = False,
    concept_quality_scaling: bool = False,
) -> Dict[int, List[Tuple[int, float]]]:
    merged_raw: Dict[int, Dict[int, float]] = {}
    for m in maps:
        for concept_idx, disease_map in m.items():
            for disease_col, weight in disease_map.items():
                _add_concept_disease_edge(merged_raw, concept_idx, disease_col, weight)

    merged: Dict[int, List[Tuple[int, float]]] = {}
    max_k = max(int(max_diseases_per_concept), 0)
    for concept_idx, disease_map in merged_raw.items():
        if max_k > 0 and len(disease_map) > max_k:
            top_items = sorted(
                disease_map.items(), key=lambda x: x[1], reverse=True,
            )[:max_k]
            disease_map = dict(top_items)
        total = float(sum(disease_map.values()))
        if total <= 0:
            continue
        n_diseases = len(disease_map)
        # IDF-like specificity: concepts mapping to fewer diseases get higher weight
        idf = 1.0 / math.log2(1.0 + n_diseases) if concept_specificity_weighting else 1.0
        # Quality scaling: multiply by max score so that weak single-disease
        # concepts (e.g. CUI SapBERT sim=0.80 → sigmoid=0.002) are not
        # inflated to 1.0 by per-concept normalization.
        quality = float(max(disease_map.values())) if concept_quality_scaling else 1.0
        pairs = sorted(
            ((int(d_col), float(w) / total * idf * quality) for d_col, w in disease_map.items()),
            key=lambda x: x[0],
        )
        merged[int(concept_idx)] = pairs
    return merged


def _build_concept_exclusion_set(
    concept_metadata_path: str,
    filter_level: int,
    disease_names: Optional[Sequence[str]] = None,
) -> Set[int]:
    """Build set of concept indices to exclude from teacher.

    Level 0: no filtering.
    Level 1: exclude PheCode concepts (diagnosis codes) + CUI concepts whose
             preferred name exactly matches a training disease name.
    Level 2: L1 + CUI concepts whose name has >=70% token overlap with any
             Orphanet/OMIM disease synonym (approximated by training disease names).
    """
    if filter_level <= 0:
        return set()
    try:
        with open(concept_metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        concept_matches = data.get("concept_matches", [])
    except Exception as e:
        print(f"concept filter: cannot load metadata {concept_metadata_path}: {e}")
        return set()

    excluded: Set[int] = set()
    disease_name_set: Set[str] = set()
    if disease_names:
        disease_name_set = {n.strip().lower() for n in disease_names if n}

    for cm in concept_matches:
        idx = int(cm.get("concept_idx", -1))
        cat = str(cm.get("category", "")).lower()
        name = str(cm.get("concept_name", "")).strip().lower()

        # L1: exclude all PheCode concepts (they are diagnosis codes)
        if cat == "phecode":
            excluded.add(idx)
            continue

        # L1: exclude CUI concepts whose name exactly matches a disease name
        if filter_level >= 1 and cat == "cui" and name and name in disease_name_set:
            excluded.add(idx)
            continue

        # L2: fuzzy token overlap >= 70% with any disease name
        if filter_level >= 2 and cat == "cui" and name:
            name_tokens = set(name.split())
            if len(name_tokens) >= 2:
                for dn in disease_name_set:
                    dn_tokens = set(dn.split())
                    if not dn_tokens:
                        continue
                    overlap = len(name_tokens & dn_tokens) / max(len(name_tokens), 1)
                    if overlap >= 0.7:
                        excluded.add(idx)
                        break

    n_phecode = sum(1 for cm in concept_matches if cm.get("category") == "phecode")
    print(
        f"concept_filter level={filter_level}: excluded {len(excluded)} / {len(concept_matches)} "
        f"concepts (phecode={n_phecode}, cui_match={len(excluded) - n_phecode})"
    )
    return excluded


def build_variant_disease_kl_teacher(
    train_main_records: Sequence[Dict[str, Any]],
    idx_to_variant: Dict[int, str],
    rsid_to_hgvs: Dict[str, str],
    disease_ids: Sequence[int],
    topk_indices_path: str,
    topk_values_path: str,
    topk_metadata_path: str,
    disease_concept_map_csv: str,
    disease_hpo_map: Optional[Dict[int, List[str]]] = None,
    hpo_semantic_map_json: Optional[str] = None,
    teacher_mode: str = "hybrid",
    hpo_topk_per_hpo: int = 5,
    hpo_min_similarity: float = 0.35,
    concept_direct_weight: float = 1.0,
    concept_semantic_weight: float = 0.35,
    min_variant_mapped_mass: float = 0.05,
    max_diseases_per_variant: int = 32,
    max_variants_per_disease: int = 256,
    concept_filter_level: int = 0,
    concept_metadata_path: str = "",
    disease_names: Optional[Sequence[str]] = None,
    shuffle_teacher: bool = False,
    shuffle_seed: int = 42,
    gene_propagation: bool = False,
    gene_propagation_alpha: float = 0.3,
    gene_propagation_distance_lambda: float = 0.01,
    gene_propagation_max_distance: int = 0,
    gene_propagation_adaptive_alpha: bool = False,
    gene_propagation_exclude_d2v: bool = True,
    gene_propagation_entropy_threshold: float = 0.0,
    gene_propagation_position_unknown_penalty: float = 0.3,
    gene_propagation_confidence_scaling: bool = False,
    max_diseases_per_concept: int = 0,
    concept_specificity_weighting: bool = False,
    bridge_length_penalty_b: float = 0.0,
    hpo_min_terms_for_full_weight: int = 0,
    teacher_max_concepts: int = 0,
    teacher_concept_temperature: float = 1.0,
    disease_score_concentration: float = 0.0,
    quality_row_weight: bool = False,
    prior_correction_alpha: float = 0.0,
    train_disease_freq: Optional[Dict[int, int]] = None,
    raw_corr_mode: bool = False,
    min_concept_corr: float = 0.01,
    disease_score_power: float = 1.0,
    concept_quality_scaling: bool = False,
) -> VariantDiseaseKLTeacher:
    """Build sparse teacher distributions for variant<->disease KL.

    Teacher source:
    1) variant->concept Top-K probabilities from MVP processed artifacts;
    2) concept->disease mapping from explicit concept map and/or HPO-semantic bridge.
    """
    teacher_mode = str(teacher_mode).strip().lower()
    if teacher_mode not in {"concept_map", "hpo_semantic", "hybrid"}:
        raise ValueError(f"Unknown vd_kl teacher_mode: {teacher_mode}")

    # --- Raw correlation mode overrides ---
    _min_corr = float(min_concept_corr) if raw_corr_mode else 0.0
    if raw_corr_mode:
        teacher_mode = "concept_map"  # skip HPO semantic bridge
        teacher_concept_temperature = 1.0  # no sharpening needed
        quality_row_weight = True  # auto-enable confidence weighting
        print(f"raw_corr_mode=True: teacher_mode→concept_map, min_concept_corr={_min_corr}, quality_row_weight→True")

    # --- Concept filtering (L0/L1/L2) ---
    excluded_concepts: Set[int] = set()
    if concept_filter_level > 0 and concept_metadata_path:
        excluded_concepts = _build_concept_exclusion_set(
            concept_metadata_path=concept_metadata_path,
            filter_level=concept_filter_level,
            disease_names=disease_names,
        )

    idx_mm, val_mm, topk_meta = _load_topk_memmap_arrays(
        indices_path=topk_indices_path,
        values_path=topk_values_path,
        metadata_path=topk_metadata_path,
    )

    # --- Shuffle teacher (negative control) ---
    # Cross-row permutation: each variant gets another variant's teacher row.
    # This preserves marginal teacher quality but breaks variant→concept association,
    # cleanly testing whether KL gains come from real clinical signal vs generic
    # regularization.  (Previous within-row shuffle was too weak — same concept set
    # stayed with each variant, only probability assignment changed.)
    if shuffle_teacher:
        print(f"SHUFFLE TEACHER: cross-row permutation (seed={shuffle_seed})")
        rng = np.random.RandomState(shuffle_seed)
        row_perm = np.arange(idx_mm.shape[0])
        rng.shuffle(row_perm)
        idx_mm = np.asarray(idx_mm)[row_perm]
        val_mm = np.asarray(val_mm)[row_perm]

    metadata_dir = Path(topk_metadata_path).resolve().parent
    variant_ids_file = topk_meta.get("variant_ids_file", "mvp_variant_ids.json")
    variant_ids_path = _resolve_optional_relative_path(str(variant_ids_file), metadata_dir)
    if not variant_ids_path.exists():
        raise FileNotFoundError(f"MVP top-k variant ids file not found: {variant_ids_path}")

    with open(variant_ids_path, "r", encoding="utf-8") as f:
        topk_row_rsids = [normalize_id(v) for v in json.load(f)]
    row_lookup = {v: i for i, v in enumerate(topk_row_rsids)}

    disease_to_col = {int(d): i for i, d in enumerate(disease_ids)}
    n_concepts = int(topk_meta.get("n_concepts", 0))
    concept_maps: List[Dict[int, Dict[int, float]]] = []
    map_stats: Dict[str, float] = {
        "concept_filter_level": float(concept_filter_level),
        "concepts_excluded": float(len(excluded_concepts)),
        "shuffle_teacher": float(shuffle_teacher),
    }

    concept_hpo_structural: Optional[Dict[int, int]] = None
    disease_hpo_count_map: Optional[Dict[int, int]] = None
    if bridge_length_penalty_b > 0:
        structural_json_path = Path(disease_concept_map_csv).resolve().parent / "concept_to_hpo_structural.json"
        if structural_json_path.exists():
            with open(structural_json_path, "r") as _f:
                _structural_data = json.load(_f)
            concept_hpo_structural = {
                int(k): len(v.get("hpo_terms", [])) for k, v in _structural_data.items()
            }
            print(f"bridge_length_penalty: loaded {len(concept_hpo_structural)} structural concepts from {structural_json_path}")
        if disease_hpo_map:
            disease_hpo_count_map = {int(k): len(v) for k, v in disease_hpo_map.items()}

    if teacher_mode in {"concept_map", "hybrid"}:
        concept_map_csv, csv_stats = _load_concept_to_disease_from_csv(
            disease_concept_map_csv=disease_concept_map_csv,
            disease_to_col=disease_to_col,
            direct_weight=float(concept_direct_weight),
            concept_hpo_structural=concept_hpo_structural,
            disease_hpo_count=disease_hpo_count_map,
            bridge_length_penalty_b=float(bridge_length_penalty_b),
        )
        concept_maps.append(concept_map_csv)
        map_stats.update(csv_stats)
    else:
        map_stats["csv_rows_used"] = 0.0
        map_stats["csv_mapped_concepts"] = 0.0

    if teacher_mode in {"hpo_semantic", "hybrid"}:
        hpo_map = disease_hpo_map or {}
        semantic_map, semantic_stats = _load_concept_to_disease_from_hpo_semantic(
            disease_hpo_map=hpo_map,
            disease_to_col=disease_to_col,
            hpo_semantic_map_json=str(hpo_semantic_map_json or ""),
            topk_per_hpo=int(hpo_topk_per_hpo),
            min_similarity=float(hpo_min_similarity),
            semantic_weight=float(concept_semantic_weight),
            n_concepts=n_concepts,
            hpo_min_terms_for_full_weight=int(hpo_min_terms_for_full_weight),
        )
        concept_maps.append(semantic_map)
        map_stats.update(semantic_stats)
    else:
        map_stats["hpo_edges_used"] = 0.0
        map_stats["hpo_mapped_diseases"] = 0.0
        map_stats["hpo_mapped_concepts"] = 0.0
        map_stats["hpo_json_found"] = 0.0

    concept_to_disease = _merge_concept_to_disease_maps(
        concept_maps,
        max_diseases_per_concept=int(max_diseases_per_concept),
        concept_specificity_weighting=bool(concept_specificity_weighting),
        concept_quality_scaling=bool(concept_quality_scaling),
    )

    hgvs_to_rsid: Dict[str, str] = {}
    for rsid_raw, hgvs_raw in rsid_to_hgvs.items():
        rsid = normalize_id(rsid_raw)
        hgvs = normalize_id(hgvs_raw)
        if rsid and hgvs and hgvs not in hgvs_to_rsid:
            hgvs_to_rsid[hgvs] = rsid

    variant_gene: Dict[int, int] = {}
    for rec in train_main_records:
        v_idx = int(rec["variant_idx"])
        g_idx = int(rec["gene_idx"])
        if v_idx not in variant_gene:
            variant_gene[v_idx] = g_idx

    variant_to_disease: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    variant_to_disease_raw: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    variant_mapped_mass: Dict[int, float] = {}
    mapped_mass_values: List[float] = []
    max_d = max(int(max_diseases_per_variant), 0)
    _conc = max(float(disease_score_concentration), 0.0)
    _conc_filtered_count = 0
    _max_c = max(int(teacher_max_concepts), 0)
    _c_temp = float(teacher_concept_temperature)
    for variant_idx, gene_idx in variant_gene.items():
        _ = gene_idx  # keeps intent explicit; gene_idx is used later via variant_gene map.
        hgvs = normalize_id(idx_to_variant.get(variant_idx, ""))
        if not hgvs:
            continue
        rsid = hgvs_to_rsid.get(hgvs)
        if not rsid:
            continue
        row_idx = row_lookup.get(rsid)
        if row_idx is None:
            continue

        concept_idx = idx_mm[row_idx]
        concept_prob = val_mm[row_idx].astype(np.float32, copy=False)
        disease_scores: Dict[int, float] = {}
        mapped_mass = 0.0

        mapped_concepts: List[Tuple[float, int, list]] = []
        for c_idx, p_c in zip(concept_idx, concept_prob):
            if p_c <= _min_corr:  # raw_corr_mode: use threshold; legacy: 0.0 (= p_c > 0)
                continue
            if int(c_idx) in excluded_concepts:
                continue
            mapped = concept_to_disease.get(int(c_idx))
            if not mapped:
                continue
            mapped_concepts.append((float(p_c), int(c_idx), mapped))

        if _max_c > 0 and len(mapped_concepts) > _max_c:
            mapped_concepts.sort(reverse=True)
            mapped_concepts = mapped_concepts[:_max_c]

        if mapped_concepts:
            raw_probs = np.array([mc[0] for mc in mapped_concepts], dtype=np.float64)
            mapped_mass = float(raw_probs.sum())
            if raw_corr_mode:
                # Raw correlation mode: use correlation values directly as weights.
                # No temperature sharpening, no concept-level renormalization.
                for p_c_val, c_idx_mc, mapped_mc in mapped_concepts:
                    for disease_col, w_cd in mapped_mc:
                        disease_scores[disease_col] = disease_scores.get(disease_col, 0.0) + float(p_c_val) * float(w_cd)
            else:
                probs = raw_probs.copy()
                if _c_temp > 0 and _c_temp != 1.0:
                    probs = np.power(probs, 1.0 / _c_temp)
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum
                for (_, c_idx_mc, mapped_mc), p_renorm in zip(mapped_concepts, probs):
                    for disease_col, w_cd in mapped_mc:
                        disease_scores[disease_col] = disease_scores.get(disease_col, 0.0) + p_renorm * float(w_cd)

        if not disease_scores:
            continue
        if float(mapped_mass) < float(min_variant_mapped_mass):
            continue
        # Concentration filter: remove diseases with score below C × uniform baseline.
        # When teacher distribution is flat, this shrinks S to only the "above-average" entries.
        if _conc > 0 and len(disease_scores) > 1:
            n_d = len(disease_scores)
            total_raw = sum(disease_scores.values())
            threshold = _conc * (total_raw / n_d)
            before_n = n_d
            disease_scores = {d: s for d, s in disease_scores.items() if s >= threshold}
            if len(disease_scores) < before_n:
                _conc_filtered_count += 1
            if not disease_scores:
                continue
        # Prior correction: boost rare diseases by dividing by (freq+μ)^α.
        # This counteracts frequency bias in teacher — rare diseases get
        # proportionally higher scores, making support set selection fairer.
        # Uses Laplace smoothing (μ=5) to avoid division by near-zero.
        if prior_correction_alpha > 0 and train_disease_freq:
            _mu = 5.0  # Laplace smoothing constant
            corrected: Dict[int, float] = {}
            for d, s in disease_scores.items():
                freq = train_disease_freq.get(d, 0)
                corrected[d] = s / max((freq + _mu) ** prior_correction_alpha, 1e-8)
            disease_scores = corrected
        if max_d > 0 and len(disease_scores) > max_d:
            top_items = sorted(
                disease_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:max_d]
            disease_scores = {int(k): float(v) for k, v in top_items}
        disease_cols = np.fromiter(disease_scores.keys(), dtype=np.int64)
        disease_vals_raw = np.fromiter(
            (disease_scores[k] for k in disease_scores.keys()),
            dtype=np.float32,
        )
        order = np.argsort(disease_cols)
        disease_cols = disease_cols[order]
        disease_vals_raw = disease_vals_raw[order]
        # Power transform: sharpen disease score distribution before normalization
        if disease_score_power > 1.0:
            disease_vals_raw = np.power(disease_vals_raw, disease_score_power)
        total = float(disease_vals_raw.sum())
        if total <= 0:
            continue
        disease_vals = disease_vals_raw / total
        variant_to_disease[variant_idx] = (disease_cols, disease_vals.astype(np.float32))
        variant_to_disease_raw[variant_idx] = (disease_cols, disease_vals_raw.astype(np.float32))
        variant_mapped_mass[int(variant_idx)] = float(mapped_mass)
        mapped_mass_values.append(float(mapped_mass))

    # --- Row weights: uniform (default) or quality-aware (mapped_mass) ---
    if quality_row_weight:
        variant_kl_weight: Dict[int, float] = {
            v: float(variant_mapped_mass.get(v, 1.0))
            for v in variant_to_disease
        }
    else:
        variant_kl_weight: Dict[int, float] = {v: 1.0 for v in variant_to_disease}
    propagated_count = 0
    filtered_by_entropy = 0
    pos_known_count = 0
    pos_unknown_count = 0
    alpha_values: List[float] = []
    distance_lambda = float(gene_propagation_distance_lambda)
    propagated_variant_set: Set[int] = set()
    if gene_propagation and gene_propagation_alpha > 0:
        # Step 0: Identify genes that have unmapped training variants (need propagation)
        needed_genes: Set[str] = set()
        for v_idx in variant_gene:
            if v_idx in variant_to_disease:
                continue
            hgvs_v = normalize_id(idx_to_variant.get(v_idx, ""))
            if hgvs_v:
                g = parse_hgvs_gene_hint(hgvs_v)
                if g:
                    needed_genes.add(g)

        # Step 1: Build disease scores for MVP variants in needed genes only
        #   Fully vectorized with numpy + scipy sparse matmul
        from scipy.sparse import csr_matrix as _csr

        rsid_to_hgvs_n: Dict[str, str] = {}
        for rsid_raw, hgvs_raw in rsid_to_hgvs.items():
            r_n = normalize_id(rsid_raw)
            h_n = normalize_id(hgvs_raw)
            if r_n and h_n:
                rsid_to_hgvs_n[r_n] = h_n

        # 1a) Identify eligible MVP rows and their gene/position
        _eligible_rows: List[int] = []
        _eligible_genes: List[str] = []
        _eligible_pos: List[Optional[int]] = []
        mvp_skipped_gene = 0
        for row_idx_mvp in range(len(topk_row_rsids)):
            rsid_mvp = topk_row_rsids[row_idx_mvp]
            hgvs_mvp = rsid_to_hgvs_n.get(rsid_mvp)
            if not hgvs_mvp:
                continue
            gene_name_mvp = parse_hgvs_gene_hint(hgvs_mvp)
            if not gene_name_mvp:
                continue
            if gene_name_mvp not in needed_genes:
                mvp_skipped_gene += 1
                continue
            _eligible_rows.append(row_idx_mvp)
            _eligible_genes.append(gene_name_mvp)
            _eligible_pos.append(parse_hgvs_protein_position(hgvs_mvp))

        # 1b) Build concept→disease sparse matrix (once)
        n_diseases = len(disease_ids)
        _c2d_rows_l: List[int] = []
        _c2d_cols_l: List[int] = []
        _c2d_vals_l: List[float] = []
        max_concept = 0
        for c_idx_int, mappings_list in concept_to_disease.items():
            if c_idx_int in excluded_concepts:
                continue
            for d_col, w_cd in mappings_list:
                _c2d_rows_l.append(int(c_idx_int))
                _c2d_cols_l.append(int(d_col))
                _c2d_vals_l.append(float(w_cd))
                if int(c_idx_int) > max_concept:
                    max_concept = int(c_idx_int)
        n_concepts = max_concept + 1
        c2d_sparse = _csr(
            (np.array(_c2d_vals_l, dtype=np.float32),
             (np.array(_c2d_rows_l, dtype=np.int64), np.array(_c2d_cols_l, dtype=np.int64))),
            shape=(n_concepts, n_diseases),
        )

        # 1c) Build variant→concept sparse matrix (fully vectorized, no Python loop)
        n_eligible = len(_eligible_rows)
        eligible_arr = np.array(_eligible_rows, dtype=np.int64)
        sel_idx = idx_mm[eligible_arr]  # [n_eligible, top_k]
        sel_val = val_mm[eligible_arr].astype(np.float32)  # [n_eligible, top_k]
        top_k = sel_idx.shape[1]

        flat_rows = np.repeat(np.arange(n_eligible, dtype=np.int64), top_k)
        flat_cols = sel_idx.ravel().astype(np.int64)
        flat_vals = sel_val.ravel()

        # Mask: positive values, within concept bounds, not excluded
        vmask = flat_vals > 0
        vmask &= flat_cols < n_concepts
        vmask &= flat_cols >= 0
        if excluded_concepts:
            excl_arr = np.array(sorted(excluded_concepts), dtype=np.int64)
            vmask &= ~np.isin(flat_cols, excl_arr)

        flat_rows = flat_rows[vmask]
        flat_cols = flat_cols[vmask]
        flat_vals = flat_vals[vmask]

        v2c_sparse = _csr(
            (flat_vals, (flat_rows, flat_cols)),
            shape=(n_eligible, n_concepts),
        )
        # mapped mass = sum of valid concept probs per variant
        _mass_vec = np.asarray(v2c_sparse.sum(axis=1)).ravel().astype(np.float32)
        # variant→disease score matrix via sparse matmul
        v2d_mat = (v2c_sparse @ c2d_sparse).tocsr()  # [n_eligible, n_diseases]

        # 1d) Group MVP entries by gene (lightweight: only indices, not dense matrices)
        _gene_mvp_entries: Dict[str, List[Tuple[Optional[int], int, float]]] = {}
        mvp_total_used = 0
        min_mass_f = float(min_variant_mapped_mass)
        v2d_indptr = v2d_mat.indptr
        for local_i in range(n_eligible):
            m_mass = float(_mass_vec[local_i])
            if m_mass < min_mass_f:
                continue
            start, end = v2d_indptr[local_i], v2d_indptr[local_i + 1]
            if start == end:
                continue
            gene_name_mvp = _eligible_genes[local_i]
            mvp_pos = _eligible_pos[local_i]
            _gene_mvp_entries.setdefault(gene_name_mvp, []).append(
                (mvp_pos, local_i, m_mass)
            )
            mvp_total_used += 1

        # Group training variants by gene (for co-processing with MVP data)
        _gene_train_variants: Dict[str, List[Tuple[int, Optional[int]]]] = {}
        for v_idx, g_idx in variant_gene.items():
            if v_idx in variant_to_disease:
                continue
            hgvs_t = normalize_id(idx_to_variant.get(v_idx, ""))
            if not hgvs_t:
                continue
            gene_name_t = parse_hgvs_gene_hint(hgvs_t)
            if not gene_name_t or gene_name_t not in _gene_mvp_entries:
                continue
            target_pos = parse_hgvs_protein_position(hgvs_t)
            _gene_train_variants.setdefault(gene_name_t, []).append((v_idx, target_pos))

        # Step 2: Per-gene processing — build dense matrix, propagate, release
        #   Peak memory: one gene's matrix (~42 × 1563 × 4B ≈ 256KB) instead of all (~580MB)
        max_prop_dist = int(gene_propagation_max_distance)
        pos_unknown_penalty = float(gene_propagation_position_unknown_penalty)
        entropy_thresh = float(gene_propagation_entropy_threshold)
        use_confidence_scaling = bool(gene_propagation_confidence_scaling)

        for gene_name, mvp_entries in _gene_mvp_entries.items():
            train_variants = _gene_train_variants.get(gene_name)
            if not train_variants:
                continue

            # Build dense matrix for this gene only
            n_ent = len(mvp_entries)
            gene_positions = np.empty(n_ent, dtype=np.float64)
            gene_masses = np.empty(n_ent, dtype=np.float64)
            gene_dmat = np.zeros((n_ent, n_diseases), dtype=np.float32)
            for j, (pos, local_i, mass) in enumerate(mvp_entries):
                gene_positions[j] = pos if pos is not None else np.nan
                gene_masses[j] = mass
                s, e = v2d_mat.indptr[local_i], v2d_mat.indptr[local_i + 1]
                if s < e:
                    cols = v2d_mat.indices[s:e]
                    vals = v2d_mat.data[s:e]
                    pm = vals > 0
                    gene_dmat[j, cols[pm]] = vals[pm]

            # Process all training variants in this gene
            for v_idx, target_pos in train_variants:
                n_src = n_ent

                # Compute distance-based weights (vectorized)
                if target_pos is not None:
                    known_mask = ~np.isnan(gene_positions)
                    dists = np.abs(gene_positions - target_pos)
                    weights = np.where(
                        known_mask,
                        np.exp(-distance_lambda * np.nan_to_num(dists, nan=0.0)) * gene_masses if distance_lambda > 0 else gene_masses,
                        gene_masses * pos_unknown_penalty,
                    )
                    if max_prop_dist > 0:
                        too_far = known_mask & (dists > max_prop_dist)
                        weights[too_far] = 0.0
                    pos_known_count += int(known_mask.sum())
                    pos_unknown_count += int((~known_mask).sum())
                    valid_dists = dists[known_mask]
                    min_dist = float(valid_dists.min()) if len(valid_dists) > 0 else float("inf")
                else:
                    weights = gene_masses * pos_unknown_penalty
                    pos_unknown_count += n_src
                    min_dist = float("inf")

                w_sum = float(weights.sum())
                n_sources_used = int((weights > 0).sum())
                if w_sum <= 0:
                    continue

                # Weighted sum: weights[n] @ gene_dmat[n, n_diseases] → agg_vec[n_diseases]
                agg_vec = weights @ gene_dmat

                # Extract nonzero entries from dense aggregation vector
                nz_mask = agg_vec > 0
                if not nz_mask.any():
                    continue
                d_cols_p = np.where(nz_mask)[0].astype(np.int64)
                d_vals_p = agg_vec[nz_mask].astype(np.float32)
                if max_d > 0 and len(d_cols_p) > max_d:
                    topk_idx = np.argpartition(d_vals_p, -max_d)[-max_d:]
                    d_cols_p = d_cols_p[topk_idx]
                    d_vals_p = d_vals_p[topk_idx]
                total_agg = float(d_vals_p.sum())
                if total_agg <= 0:
                    continue
                d_vals_normed = d_vals_p / total_agg

                # entropy filter
                if entropy_thresh > 0 and len(d_vals_normed) > 1:
                    ent = -float(np.sum(d_vals_normed * np.log(np.clip(d_vals_normed, 1e-12, None))))
                    max_ent = math.log(len(d_vals_normed))
                    if max_ent > 0 and ent / max_ent > entropy_thresh:
                        filtered_by_entropy += 1
                        continue

                order = np.argsort(d_cols_p)
                d_cols_p = d_cols_p[order]
                d_vals_normed = d_vals_normed[order]
                d_vals_raw_p = d_vals_p[order]

                # alpha: optionally decay with distance to nearest source
                if gene_propagation_adaptive_alpha and min_dist < float("inf") and distance_lambda > 0:
                    effective_alpha = gene_propagation_alpha * math.exp(-distance_lambda * min_dist)
                else:
                    effective_alpha = float(gene_propagation_alpha)

                # confidence scaling
                if use_confidence_scaling and n_sources_used > 0:
                    confidence = 1.0 - math.exp(-0.5 * n_sources_used)
                    effective_alpha *= confidence

                alpha_values.append(effective_alpha)
                variant_to_disease[v_idx] = (d_cols_p, d_vals_normed)
                variant_to_disease_raw[v_idx] = (d_cols_p, d_vals_raw_p)
                variant_mapped_mass[v_idx] = effective_alpha
                mapped_mass_values.append(effective_alpha)
                variant_kl_weight[v_idx] = effective_alpha
                propagated_variant_set.add(v_idx)
                propagated_count += 1
            # gene_dmat is released here (end of gene loop iteration)

        mvp_genes_used = len(_gene_mvp_entries)
        avg_mvp_per_gene = mvp_total_used / max(mvp_genes_used, 1)
        del _gene_mvp_entries, _gene_train_variants, v2d_mat, v2c_sparse, _mass_vec
        avg_alpha = float(np.mean(alpha_values)) if alpha_values else 0.0
        total_pos_pairs = pos_known_count + pos_unknown_count
        pos_parse_rate = pos_known_count / max(total_pos_pairs, 1)
        print(
            f"gene_propagation_v3: needed_genes={len(needed_genes)} "
            f"mvp_variants={mvp_total_used} mvp_skipped_gene={mvp_skipped_gene} "
            f"mvp_genes={mvp_genes_used} "
            f"avg_mvp_per_gene={avg_mvp_per_gene:.1f} "
            f"direct={len(variant_kl_weight) - propagated_count} "
            f"propagated={propagated_count} filtered_by_entropy={filtered_by_entropy} "
            f"total={len(variant_to_disease)} "
            f"alpha={gene_propagation_alpha} avg_effective_alpha={avg_alpha:.4f} "
            f"distance_lambda={distance_lambda} "
            f"max_distance={max_prop_dist} adaptive_alpha={gene_propagation_adaptive_alpha} "
            f"confidence_scaling={use_confidence_scaling} "
            f"entropy_threshold={entropy_thresh} "
            f"pos_parse_rate={pos_parse_rate:.2%} pos_unknown_penalty={pos_unknown_penalty} "
            f"exclude_d2v={gene_propagation_exclude_d2v}"
        )

    pool_variant_idx = np.asarray(sorted(variant_to_disease.keys()), dtype=np.int64)
    pool_gene_idx = np.asarray([variant_gene[v] for v in pool_variant_idx], dtype=np.int64)
    variant_idx_to_pool = {int(v): int(i) for i, v in enumerate(pool_variant_idx.tolist())}

    disease_bucket: Dict[int, List[Tuple[int, float]]] = {}
    _prop_set = propagated_variant_set if gene_propagation else set()
    _exclude_prop = bool(gene_propagation_exclude_d2v) and len(_prop_set) > 0
    # Build d2v from the same sparse joint scores before v2d row-normalization.
    for variant_idx, (disease_cols, disease_scores_raw) in variant_to_disease_raw.items():
        if _exclude_prop and int(variant_idx) in _prop_set:
            continue
        pool_pos = variant_idx_to_pool[int(variant_idx)]
        for d_col, score_vd in zip(disease_cols.tolist(), disease_scores_raw.tolist()):
            disease_bucket.setdefault(int(d_col), []).append((pool_pos, float(score_vd)))

    disease_to_variant: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    max_v = max(int(max_variants_per_disease), 0)
    for d_col, pairs in disease_bucket.items():
        if not pairs:
            continue
        if max_v > 0 and len(pairs) > max_v:
            pairs = sorted(
                pairs,
                key=lambda x: x[1],
                reverse=True,
            )[:max_v]
        pos = np.asarray([p for p, _ in pairs], dtype=np.int64)
        vals = np.asarray([v for _, v in pairs], dtype=np.float32)
        total = float(vals.sum())
        if total <= 0:
            continue
        vals = vals / total
        order = np.argsort(pos)
        disease_to_variant[d_col] = (pos[order], vals[order])

    mapped_variant_count = len(variant_to_disease)
    train_variant_count = len(variant_gene)
    mapped_mass_arr = np.asarray(mapped_mass_values, dtype=np.float32) if mapped_mass_values else np.asarray([], dtype=np.float32)
    v2d_support_sizes = np.asarray(
        [len(d_cols) for d_cols, _ in variant_to_disease.values()],
        dtype=np.float32,
    ) if variant_to_disease else np.asarray([], dtype=np.float32)
    stats = {
        "train_variants": float(train_variant_count),
        "mapped_variants": float(mapped_variant_count),
        "mapped_variant_ratio": float(mapped_variant_count / max(train_variant_count, 1)),
        "mapped_diseases": float(len(disease_to_variant)),
        "pool_size": float(len(pool_variant_idx)),
        "teacher_mode_code": float({"concept_map": 1, "hpo_semantic": 2, "hybrid": 3}[teacher_mode]),
        "concept_mapped_count": float(len(concept_to_disease)),
        "min_variant_mapped_mass": float(min_variant_mapped_mass),
        "max_diseases_per_variant": float(max_d),
        "max_variants_per_disease": float(max_v),
        "mapped_mass_mean": float(mapped_mass_arr.mean()) if mapped_mass_arr.size > 0 else 0.0,
        "mapped_mass_median": float(np.median(mapped_mass_arr)) if mapped_mass_arr.size > 0 else 0.0,
        "mapped_mass_p90": float(np.quantile(mapped_mass_arr, 0.9)) if mapped_mass_arr.size > 0 else 0.0,
        "gene_propagation": float(gene_propagation),
        "gene_propagation_alpha": float(gene_propagation_alpha) if gene_propagation else 0.0,
        "gene_propagation_avg_effective_alpha": float(np.mean(alpha_values)) if gene_propagation and alpha_values else 0.0,
        "gene_propagation_filtered_by_entropy": float(filtered_by_entropy) if gene_propagation else 0.0,
        "gene_propagation_pos_parse_rate": float(pos_known_count / max(pos_known_count + pos_unknown_count, 1)) if gene_propagation else 0.0,
        "propagated_variants": float(propagated_count),
        "direct_mapped_variants": float(len(variant_to_disease) - propagated_count),
        "v2d_support_size_mean": float(v2d_support_sizes.mean()) if v2d_support_sizes.size > 0 else 0.0,
        "v2d_support_size_median": float(np.median(v2d_support_sizes)) if v2d_support_sizes.size > 0 else 0.0,
        "v2d_support_size_max": float(v2d_support_sizes.max()) if v2d_support_sizes.size > 0 else 0.0,
        "v2d_support_size_p90": float(np.quantile(v2d_support_sizes, 0.9)) if v2d_support_sizes.size > 0 else 0.0,
        "disease_score_concentration": float(_conc),
        "concentration_filtered_variants": float(_conc_filtered_count),
        "raw_corr_mode": float(raw_corr_mode),
        "min_concept_corr": float(_min_corr),
        "disease_score_power": float(disease_score_power),
    }
    stats.update(map_stats)
    return VariantDiseaseKLTeacher(
        variant_to_disease=variant_to_disease,
        disease_to_variant=disease_to_variant,
        pool_variant_idx=pool_variant_idx,
        pool_gene_idx=pool_gene_idx,
        variant_idx_to_pool=variant_idx_to_pool,
        variant_mapped_mass=variant_mapped_mass,
        variant_kl_weight=variant_kl_weight,
        stats=stats,
    )


class DictDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def compute_disease_inv_freq_weights(
    main_df: pd.DataFrame,
    method: str = "sqrt_inv",
) -> Dict[int, float]:
    """Per-disease inverse frequency weights for sample rebalancing.

    Returns a dict mapping disease_index -> weight (mean-normalized to 1.0).
    Supported methods: "sqrt_inv" (1/sqrt(freq)), "log_inv" (1/log1p(freq)).
    """
    if method == "none":
        return {}
    freq = main_df.groupby("disease_index")["variant_id"].nunique()
    if method == "sqrt_inv":
        w = 1.0 / np.sqrt(freq.clip(lower=1).astype(float))
    elif method == "log_inv":
        w = 1.0 / np.log1p(freq.astype(float))
    else:
        raise ValueError(f"Unknown disease_freq_reweight method: {method}")
    w = w / w.mean()
    return {int(d): float(v) for d, v in w.items()}


def build_query_positive_disease_lookup(
    main_df: pd.DataFrame,
) -> Dict[Tuple[str, str], List[int]]:
    required_cols = {"variant_id", "gene_id", "disease_index"}
    if not required_cols.issubset(set(main_df.columns)):
        return {}

    grouped = main_df.groupby(["variant_id", "gene_id"], as_index=False).agg(
        disease_index=("disease_index", list),
    )
    lookup: Dict[Tuple[str, str], List[int]] = {}
    for variant_id, gene_id, disease_ids in grouped.itertuples(index=False):
        key = (normalize_id(variant_id), normalize_id(gene_id))
        pos = sorted({int(d) for d in disease_ids})
        if not key[0] or not key[1] or not pos:
            continue
        lookup[key] = pos
    return lookup


def make_main_records(
    main_df: pd.DataFrame,
    variant_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
    disease_id_to_col: Optional[Dict[int, int]] = None,
    positive_disease_lookup: Optional[Dict[Tuple[str, str], List[int]]] = None,
    disease_freq_weights: Optional[Dict[int, float]] = None,
    freq_weight_agg: str = "max",
    freq_weight_clip: float = 0.0,
    return_stats: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped = main_df.groupby(["variant_id", "gene_id"], as_index=False).agg(
        disease_index=("disease_index", list),
        confidence=("confidence", "mean"),
    )

    records: List[Dict[str, Any]] = []
    stats = {
        "input_rows": int(len(main_df)),
        "input_queries": int(len(grouped)),
        "records_emitted": 0,
        "dropped_missing_variant": 0,
        "dropped_missing_gene": 0,
        "dropped_empty_positive": 0,
        "queries_with_positive_override": 0,
        "override_extra_positives_added": 0,
    }
    for row in grouped.itertuples(index=False):
        variant_id, gene_id, disease_ids, confidence = row
        v_idx = variant_to_idx.get(variant_id)
        g_idx = gene_to_idx.get(gene_id)
        if v_idx is None:
            stats["dropped_missing_variant"] += 1
            continue
        if g_idx is None:
            stats["dropped_missing_gene"] += 1
            continue
        local_pos = sorted({int(d) for d in disease_ids})
        pos = list(local_pos)
        if positive_disease_lookup is not None:
            override_pos = positive_disease_lookup.get((normalize_id(variant_id), normalize_id(gene_id)))
            if override_pos is not None:
                pos = sorted({int(d) for d in override_pos} | set(local_pos))
                extra = max(len(pos) - len(local_pos), 0)
                if extra > 0:
                    stats["queries_with_positive_override"] += 1
                    stats["override_extra_positives_added"] += int(extra)
        if not pos:
            stats["dropped_empty_positive"] += 1
            continue
        freq_w = 1.0
        if disease_freq_weights:
            freq_candidates = [float(disease_freq_weights.get(d, 1.0)) for d in pos]
            if freq_weight_agg == "mean":
                freq_w = float(np.mean(freq_candidates)) if freq_candidates else 1.0
            elif freq_weight_agg == "max":
                freq_w = max(freq_candidates) if freq_candidates else 1.0
            else:
                raise ValueError(f"Unknown freq_weight_agg: {freq_weight_agg}")
            if float(freq_weight_clip) > 0:
                freq_w = min(float(freq_w), float(freq_weight_clip))
            freq_w = max(float(freq_w), 0.0)
        record = {
            "variant_idx": v_idx,
            "gene_idx": g_idx,
            "positive_disease_ids": pos,
            "confidence": float(confidence) * freq_w,
        }
        if disease_id_to_col is not None:
            record["positive_disease_cols"] = [
                int(disease_id_to_col[d]) for d in pos if d in disease_id_to_col
            ]
        records.append(record)
    stats["records_emitted"] = int(len(records))

    # Normalize sample weights to mean=1 under disease-frequency reweighting.
    if disease_freq_weights and records:
        w = np.asarray([float(r["confidence"]) for r in records], dtype=np.float64)
        mean_w = float(w.mean())
        if mean_w > 0:
            for r in records:
                r["confidence"] = float(r["confidence"]) / mean_w
    if return_stats:
        return records, stats
    return records


def make_domain_records(
    df: pd.DataFrame,
    variant_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
    return_stats: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "input_rows": int(len(df)),
        "records_emitted": 0,
        "dropped_missing_variant": 0,
        "dropped_missing_gene": 0,
    }
    for row in df[["variant_id", "gene_id", "domain_map"]].itertuples(index=False):
        v, g, y = row
        v_idx = variant_to_idx.get(v)
        g_idx = gene_to_idx.get(g)
        if v_idx is None:
            stats["dropped_missing_variant"] += 1
            continue
        if g_idx is None:
            stats["dropped_missing_gene"] += 1
            continue
        records.append({"variant_idx": v_idx, "gene_idx": g_idx, "label": int(y)})
    stats["records_emitted"] = int(len(records))
    if return_stats:
        return records, stats
    return records


def make_func_records(
    df: pd.DataFrame,
    variant_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
    target_cols: Optional[Sequence[str]] = None,
    mask_cols: Optional[Sequence[str]] = None,
    return_stats: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "input_rows": int(len(df)),
        "records_emitted": 0,
        "dropped_missing_variant": 0,
        "dropped_missing_gene": 0,
    }

    # 检测 v2 格式
    is_v2 = "phyloP100way_rs" in df.columns
    if is_v2:
        reg_cols = list(FUNC_REGRESSION_COLS)
        reg_mask_cols = list(FUNC_REGRESSION_MASK_COLS)
        mech_cols = [c for c in FUNC_MECHANISM_COLS if c in df.columns]
        has_mechanism = "mechanism_mask" in df.columns and len(mech_cols) > 0

        iter_cols = ["variant_id", "gene_id"] + reg_cols + reg_mask_cols
        if has_mechanism:
            iter_cols += mech_cols + ["mechanism_mask"]

        n_reg = len(reg_cols)
        n_mech = len(mech_cols)

        for row in df[iter_cols].itertuples(index=False):
            variant_id = row[0]
            gene_id = row[1]
            v_idx = variant_to_idx.get(variant_id)
            g_idx = gene_to_idx.get(gene_id)
            if v_idx is None:
                stats["dropped_missing_variant"] += 1
                continue
            if g_idx is None:
                stats["dropped_missing_gene"] += 1
                continue

            reg_target = np.asarray(row[2 : 2 + n_reg], dtype=np.float32)
            reg_mask = np.asarray(row[2 + n_reg : 2 + 2 * n_reg], dtype=np.float32)

            rec: Dict[str, Any] = {
                "variant_idx": v_idx,
                "gene_idx": g_idx,
                "regression_target": reg_target,
                "regression_mask": reg_mask,
            }

            if has_mechanism:
                offset = 2 + 2 * n_reg
                mech_target = np.asarray(row[offset : offset + n_mech], dtype=np.float32)
                mech_mask_val = float(row[offset + n_mech])
                rec["mechanism_target"] = mech_target
                rec["mechanism_mask"] = mech_mask_val
            else:
                rec["mechanism_target"] = np.zeros(n_mech, dtype=np.float32)
                rec["mechanism_mask"] = 0.0

            records.append(rec)
    else:
        # 旧 v1 fallback
        selected_targets = list(target_cols) if target_cols is not None else list(FUNC_TARGET_COLS)
        selected_masks = list(mask_cols) if mask_cols is not None else get_func_mask_cols(selected_targets)
        cols = ["variant_id", "gene_id"] + selected_targets + selected_masks
        for row in df[cols].itertuples(index=False):
            variant_id = row[0]
            gene_id = row[1]
            target = np.asarray(row[2 : 2 + len(selected_targets)], dtype=np.float32)
            mask = np.asarray(row[2 + len(selected_targets) :], dtype=np.float32)
            v_idx = variant_to_idx.get(variant_id)
            g_idx = gene_to_idx.get(gene_id)
            if v_idx is None:
                stats["dropped_missing_variant"] += 1
                continue
            if g_idx is None:
                stats["dropped_missing_gene"] += 1
                continue
            records.append({
                "variant_idx": v_idx,
                "gene_idx": g_idx,
                "regression_target": target,
                "regression_mask": mask,
                "mechanism_target": np.zeros(len(FUNC_MECHANISM_COLS), dtype=np.float32),
                "mechanism_mask": 0.0,
            })

    stats["records_emitted"] = int(len(records))
    if return_stats:
        return records, stats
    return records


def make_vd_d2v_records(
    train_main_records: Sequence[Dict[str, Any]],
    disease_ids: Sequence[int],
    vd_kl_teacher: VariantDiseaseKLTeacher,
    return_stats: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, float]]:
    disease_positive_pool: Dict[int, Set[int]] = {}
    stats = {
        "input_queries": float(len(train_main_records)),
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
    for rec in train_main_records:
        v_idx = int(rec["variant_idx"])
        pool_pos = vd_kl_teacher.variant_idx_to_pool.get(v_idx)
        if pool_pos is None:
            continue
        stats["mapped_queries"] += 1.0
        positive_cols = rec.get("positive_disease_cols", [])
        for d_col in positive_cols:
            disease_positive_pool.setdefault(int(d_col), set()).add(int(pool_pos))

    records: List[Dict[str, Any]] = []
    positive_sizes: List[int] = []
    anchor_sizes: List[int] = []
    teacher_sizes: List[int] = []
    for d_col, positive_pool in disease_positive_pool.items():
        stats["candidate_diseases"] += 1.0
        if not positive_pool:
            stats["dropped_no_positive_pool"] += 1.0
            continue
        teacher_item = vd_kl_teacher.disease_to_variant.get(int(d_col))
        if teacher_item is None or len(teacher_item[0]) == 0:
            stats["dropped_no_teacher"] += 1.0
            continue
        teacher_idx_np, teacher_prob_np = teacher_item
        teacher_pairs = sorted(
            zip(teacher_idx_np.tolist(), teacher_prob_np.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        if not teacher_pairs:
            stats["dropped_no_teacher"] += 1.0
            continue
        disease_id = int(disease_ids[int(d_col)])
        positive_list = sorted(int(p) for p in positive_pool)
        teacher_support_set = {int(idx) for idx, _ in teacher_pairs}
        anchor_list = [int(p) for p in positive_list if int(p) in teacher_support_set]
        if not anchor_list:
            stats["dropped_no_anchor_overlap"] += 1.0
            continue
        records.append(
            {
                "disease_id": disease_id,
                "disease_col": int(d_col),
                "positive_pool_pos": positive_list,
                "anchor_pool_pos": anchor_list,
                "teacher_pool_pos": [int(idx) for idx, _ in teacher_pairs],
                "teacher_probs": [float(prob) for _, prob in teacher_pairs],
            }
        )
        positive_sizes.append(len(positive_list))
        anchor_sizes.append(len(anchor_list))
        teacher_sizes.append(len(teacher_pairs))

    stats["records_emitted"] = float(len(records))
    if positive_sizes:
        positive_arr = np.asarray(positive_sizes, dtype=np.float32)
        stats["positive_pool_mean"] = float(positive_arr.mean())
        stats["positive_pool_median"] = float(np.median(positive_arr))
    if anchor_sizes:
        anchor_arr = np.asarray(anchor_sizes, dtype=np.float32)
        stats["anchor_pool_mean"] = float(anchor_arr.mean())
        stats["anchor_pool_median"] = float(np.median(anchor_arr))
    if teacher_sizes:
        teacher_arr = np.asarray(teacher_sizes, dtype=np.float32)
        stats["teacher_support_mean"] = float(teacher_arr.mean())
        stats["teacher_support_median"] = float(np.median(teacher_arr))
    if return_stats:
        return records, stats
    return records


def _collate_main(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "variant_idx": torch.tensor([b["variant_idx"] for b in batch], dtype=torch.long),
        "gene_idx": torch.tensor([b["gene_idx"] for b in batch], dtype=torch.long),
        "positive_disease_ids": [b["positive_disease_ids"] for b in batch],
        "positive_disease_cols": [
            b.get("positive_disease_cols", b["positive_disease_ids"]) for b in batch
        ],
        "confidence": torch.tensor([b.get("confidence", 1.0) for b in batch], dtype=torch.float32),
    }


def _collate_classification(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "variant_idx": torch.tensor([b["variant_idx"] for b in batch], dtype=torch.long),
        "gene_idx": torch.tensor([b["gene_idx"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def _collate_func(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "variant_idx": torch.tensor([b["variant_idx"] for b in batch], dtype=torch.long),
        "gene_idx": torch.tensor([b["gene_idx"] for b in batch], dtype=torch.long),
        "regression_target": torch.tensor(np.stack([b["regression_target"] for b in batch]), dtype=torch.float32),
        "regression_mask": torch.tensor(np.stack([b["regression_mask"] for b in batch]), dtype=torch.float32),
        "mechanism_target": torch.tensor(np.stack([b["mechanism_target"] for b in batch]), dtype=torch.float32),
        "mechanism_mask": torch.tensor([b["mechanism_mask"] for b in batch], dtype=torch.float32),
    }


def _collate_vd_d2v(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "disease_id": [int(b["disease_id"]) for b in batch],
        "disease_col": [int(b["disease_col"]) for b in batch],
        "positive_pool_pos": [list(b["positive_pool_pos"]) for b in batch],
        "anchor_pool_pos": [list(b["anchor_pool_pos"]) for b in batch],
        "teacher_pool_pos": [list(b["teacher_pool_pos"]) for b in batch],
        "teacher_probs": [list(b["teacher_probs"]) for b in batch],
    }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_for_task(
    task_name: str,
    records: List[Dict[str, Any]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> DataLoader:
    dataset = DictDataset(records)
    if task_name == "main":
        collate_fn = _collate_main
    elif task_name == "domain":
        collate_fn = _collate_classification
    elif task_name == "func":
        collate_fn = _collate_func
    elif task_name == "vd_d2v":
        collate_fn = _collate_vd_d2v
    else:
        raise ValueError(f"Unknown task_name: {task_name}")

    effective_shuffle = shuffle and len(dataset) > 0
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": effective_shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader_kwargs["generator"] = generator
        loader_kwargs["worker_init_fn"] = _seed_worker
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        **loader_kwargs,
    )


def summarize_split(
    task_splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    variant_col: str = "variant_id",
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for task_name, split_frames in task_splits.items():
        train_df, val_df, test_df = split_frames

        def _unique_count(df: pd.DataFrame) -> int:
            if variant_col not in df.columns:
                return int(len(df))
            values = df[variant_col].map(normalize_id)
            values = values[values != ""]
            return int(values.nunique())

        summary[task_name] = {
            "train": _unique_count(train_df),
            "val": _unique_count(val_df),
            "test": _unique_count(test_df),
        }
    return summary


def compute_train_test_overlap(
    task_splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
) -> Dict[str, int]:
    train_union: Set[str] = set()
    test_union: Set[str] = set()
    for train_df, _, test_df in task_splits.values():
        train_union.update(train_df["variant_id"].tolist())
        test_union.update(test_df["variant_id"].tolist())

    return {
        "train_variants": len(train_union),
        "test_variants": len(test_union),
        "train_test_overlap": len(train_union & test_union),
    }


def compute_func_target_scales(
    func_train_df: pd.DataFrame,
    target_cols: Optional[Sequence[str]] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """计算回归目标列的 per-column std。v2 格式使用 rankscore (均在 [0,1])。"""
    is_v2 = "phyloP100way_rs" in func_train_df.columns
    if is_v2:
        selected_targets = list(FUNC_REGRESSION_COLS)
        selected_masks = list(FUNC_REGRESSION_MASK_COLS)
    else:
        selected_targets = list(target_cols) if target_cols is not None else list(FUNC_TARGET_COLS)
        selected_masks = get_func_mask_cols(selected_targets)

    scales: List[float] = []
    for target_col, mask_col in zip(selected_targets, selected_masks):
        if target_col not in func_train_df.columns or mask_col not in func_train_df.columns:
            scales.append(1.0)
            continue
        valid = func_train_df[mask_col] > 0
        vals = pd.to_numeric(func_train_df.loc[valid, target_col], errors="coerce").dropna()
        if len(vals) == 0:
            scales.append(1.0)
        else:
            scales.append(float(max(vals.std(ddof=0), eps)))
    return np.asarray(scales, dtype=np.float32)


def build_disease_frequency_buckets(
    main_train_df: pd.DataFrame,
    rare_max: int = 3,
    frequent_min: int = 21,
) -> Dict[str, Set[int]]:
    if main_train_df.empty:
        return {"rare": set(), "medium": set(), "frequent": set()}
    freq = (
        main_train_df.groupby("disease_index")["variant_id"]
        .nunique()
        .to_dict()
    )
    rare = {int(d) for d, c in freq.items() if int(c) <= rare_max}
    frequent = {int(d) for d, c in freq.items() if int(c) >= frequent_min}
    medium = {int(d) for d, c in freq.items() if rare_max < int(c) < frequent_min}
    return {"rare": rare, "medium": medium, "frequent": frequent}
