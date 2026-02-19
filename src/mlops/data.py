from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from .utils import sha256_file, safe_mkdir

SUPPORTED_FORMATS = {"csv", "json", "parquet"}

@dataclass
class DatasetArtifacts:
    dataset_hash: str
    sample_path: Path
    schema_path: Path
    feature_stats_path: Path

def load_dataset(path: Union[str, Path], fmt: str) -> pd.DataFrame:
    path = Path(path)
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported dataset format: {fmt}. Supported: {sorted(SUPPORTED_FORMATS)}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "json":
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)

def detect_target(df: pd.DataFrame, explicit_target: Optional[str] = None) -> str:
    if explicit_target and explicit_target in df.columns:
        return explicit_target

    candidates = [c for c in df.columns if c.lower() in {"target", "label", "y", "outcome", "class", "diabetes"}]
    if candidates:
        return candidates[0]
    # fallback: last column
    return df.columns[-1]

def detect_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target])
    cat_cols = []
    num_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return cat_cols, num_cols

def schema_from_df(df: pd.DataFrame) -> Dict:
    schema = {"columns": []}
    for c in df.columns:
        dtype = str(df[c].dtype)
        schema["columns"].append({"name": c, "dtype": dtype, "n_null": int(df[c].isna().sum())})
    schema["n_rows"] = int(df.shape[0])
    schema["n_cols"] = int(df.shape[1])
    return schema

def feature_stats(df: pd.DataFrame, target: str) -> Dict:
    stats = {"target": target, "features": {}}
    for c in df.columns:
        s = df[c]
        entry = {"dtype": str(s.dtype), "n_null": int(s.isna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            entry.update({
                "mean": float(np.nanmean(s.to_numpy(dtype=float))),
                "std": float(np.nanstd(s.to_numpy(dtype=float))),
                "min": float(np.nanmin(s.to_numpy(dtype=float))),
                "max": float(np.nanmax(s.to_numpy(dtype=float))),
                "p01": float(np.nanpercentile(s.to_numpy(dtype=float), 1)),
                "p99": float(np.nanpercentile(s.to_numpy(dtype=float), 99)),
            })
        else:
            entry.update({
                "n_unique": int(s.nunique(dropna=True)),
                "top": None if s.dropna().empty else str(s.dropna().value_counts().idxmax()),
            })
        stats["features"][c] = entry
    return stats

def clean_dataframe(df: pd.DataFrame, target: str, cat_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    # Basic cleaning: drop fully empty rows, coerce numeric, cap outliers by winsorization (1%-99%)
    df = df.dropna(how="all").copy()

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        low = df[c].quantile(0.01)
        high = df[c].quantile(0.99)
        df[c] = df[c].clip(lower=low, upper=high)

    # Ensure target exists
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not in dataframe columns.")
    return df

def save_reproducibility_artifacts(df: pd.DataFrame, dataset_path: Path, out_dir: Path) -> DatasetArtifacts:
    safe_mkdir(out_dir)
    dataset_hash = sha256_file(dataset_path)

    sample_path = out_dir / "dataset_sample.csv"
    schema_path = out_dir / "schema.json"
    feature_stats_path = out_dir / "feature_stats.json"

    df.sample(min(len(df), 200), random_state=42).to_csv(sample_path, index=False)

    import json
    schema = schema_from_df(df)
    schema["dataset_hash"] = dataset_hash
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    stats = feature_stats(df, target=schema.get("target", "unknown"))
    feature_stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return DatasetArtifacts(
        dataset_hash=dataset_hash,
        sample_path=sample_path,
        schema_path=schema_path,
        feature_stats_path=feature_stats_path,
    )
