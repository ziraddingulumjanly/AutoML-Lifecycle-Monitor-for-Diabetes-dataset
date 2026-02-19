from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

@dataclass
class FeatureSpec:
    target: str
    categorical: List[str]
    numerical: List[str]
    task: str  # classification | regression

def build_preprocessor(categorical: List[str], numerical: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numerical),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def infer_task(y: pd.Series, explicit_task: str = "auto") -> str:
    if explicit_task in {"classification", "regression"}:
        return explicit_task
    # auto: if few uniques and integer/bool -> classification
    if y.dtype == bool:
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 20 and set(y.dropna().unique()).issubset(set(range(int(y.min()), int(y.max()) + 1))):
            return "classification"
        return "regression"
    return "classification"
