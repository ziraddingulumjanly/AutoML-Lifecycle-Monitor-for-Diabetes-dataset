from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import joblib

from src.mlops.registry import ModelRegistry

REGISTRY_DIR = Path("registry")

class LoadedModel:
    def __init__(self, version: str, model, preprocessor, metadata: Dict, metrics: Dict, reference_csv: Optional[Path]):
        self.version = version
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.metrics = metrics
        self.reference_csv = reference_csv

    def predict_df(self, df: pd.DataFrame):
        X = df[self.metadata["input_features"]]
        X_t = self.preprocessor.transform(X)
        y = self.model.predict(X_t)
        return y

def load_latest() -> LoadedModel:
    reg = ModelRegistry(REGISTRY_DIR)
    version = reg.resolve_version()
    if not version:
        raise FileNotFoundError("No model found in registry. Train first with scripts/train.py.")
    model, pre, metadata, metrics, ref = reg.load(version)
    return LoadedModel(version, model, pre, metadata, metrics, ref)

def validate_input(df: pd.DataFrame, metadata: Dict) -> None:
    needed = set(metadata["input_features"])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
