from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

from .data import load_dataset, detect_target, detect_feature_types, clean_dataframe
from .features import build_preprocessor, infer_task
from .evaluate import full_report, pick_metric
from .registry import ModelRegistry
from .utils import sha256_file, safe_mkdir, now_utc_iso, json_dumps

@dataclass
class TrainResult:
    best_model_name: str
    best_metric_name: str
    best_metric_value: float
    leaderboard: List[Dict]
    version: Optional[str]

def _build_models(task: str):
    models = {}
    if task == "classification":
        models["ridge"] = RidgeClassifier()
        models["random_forest"] = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        models["gradient_boosting"] = GradientBoostingClassifier(random_state=42)
        # Optional XGBoost
        try:
            import xgboost as xgb
            models["xgboost_optional"] = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                random_state=42, n_jobs=-1, eval_metric="logloss"
            )
        except Exception:
            pass
    else:
        models["ridge"] = Ridge()
        models["random_forest"] = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        models["gradient_boosting"] = GradientBoostingRegressor(random_state=42)
        try:
            import xgboost as xgb
            models["xgboost_optional"] = xgb.XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                random_state=42, n_jobs=-1
            )
        except Exception:
            pass
    return models

def train_from_config(config: Dict) -> TrainResult:
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    registry_cfg = config["registry"]

    dataset_path = Path(dataset_cfg["path"])
    fmt = dataset_cfg.get("format", "csv")
    df = load_dataset(dataset_path, fmt)
    target = detect_target(df, dataset_cfg.get("target"))

    cat_cols, num_cols = detect_feature_types(df, target=target)
    df = clean_dataframe(df, target=target, cat_cols=cat_cols, num_cols=num_cols)

    y = df[target]
    task = infer_task(y, explicit_task=train_cfg.get("task", "auto"))
    X = df.drop(columns=[target])

    # splits
    test_size = float(dataset_cfg.get("test_size", 0.2))
    val_size = float(dataset_cfg.get("val_size", 0.2))
    seed = int(dataset_cfg.get("random_seed", 42))

    stratify = y if task == "classification" else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)

    # val split from trainval
    val_ratio = val_size / (1.0 - test_size)
    stratify_tv = y_trainval if task == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio, random_state=seed, stratify=stratify_tv)

    pre = build_preprocessor(cat_cols, num_cols)
    models = _build_models(task)

    leaderboard = []
    best = None

    cv_folds = int(train_cfg.get("cv_folds", 5))
    metric_pref = train_cfg.get("metric", "auto")

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])

        # CV score (rough selection)
        if task == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            scoring = "roc_auc" if metric_pref in {"auto", "roc_auc"} else "f1_weighted"
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            scoring = "neg_root_mean_squared_error" if metric_pref in {"auto", "rmse"} else "r2"

        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
            cv_score = float(np.mean(scores))
        except Exception as e:
            leaderboard.append({"model": name, "cv_scoring": scoring, "cv_score": None, "error": str(e)})
            continue

        pipe.fit(X_train, y_train)

        # eval on val
        y_pred = pipe.predict(X_val)
        y_proba = None
        if task == "classification":
            # proba if available
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                y_proba = pipe.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) == 2 else None

        metric_name, metric_val = pick_metric(task, y_val, y_pred, y_proba=y_proba, explicit=metric_pref)

        entry = {
            "model": name,
            "cv_scoring": scoring,
            "cv_score": cv_score,
            "val_metric": metric_name,
            "val_metric_value": float(metric_val),
        }
        leaderboard.append(entry)

        if best is None:
            best = (name, metric_name, float(metric_val), pipe)
        else:
            # maximize for classification metrics; minimize for rmse
            _, bm_name, bm_val, _ = best
            if metric_name == "rmse":
                if metric_val < bm_val:
                    best = (name, metric_name, float(metric_val), pipe)
            else:
                if metric_val > bm_val:
                    best = (name, metric_name, float(metric_val), pipe)

    if best is None:
        raise RuntimeError("Training failed: no model could be trained successfully.")

    best_name, best_metric, best_metric_value, best_pipe = best

    # final test report
    y_test_pred = best_pipe.predict(X_test)
    y_test_proba = None
    if task == "classification" and hasattr(best_pipe.named_steps["model"], "predict_proba") and len(np.unique(y_test)) == 2:
        y_test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_report = full_report(task, y_test, y_test_pred, y_proba=y_test_proba)

    # write artifacts
    out_art = Path("artifacts")
    safe_mkdir(out_art)
    (out_art / "leaderboard.json").write_text(json_dumps(leaderboard), encoding="utf-8")
    (out_art / "metrics.json").write_text(json_dumps({"task": task, "best_model": best_name, "val": {best_metric: best_metric_value}, "test": test_report}), encoding="utf-8")
    (out_art / "training_config.json").write_text(json_dumps(config), encoding="utf-8")

    # reference data snapshot (for drift): use trainval sample
    ref_path = out_art / "reference_dataset.csv"
    pd.concat([X_trainval, y_trainval.rename(target)], axis=1).sample(min(len(X_trainval), 500), random_state=seed).to_csv(ref_path, index=False)

    # registry
    registry = ModelRegistry(Path(registry_cfg.get("dir", "registry")))
    dataset_hash = sha256_file(dataset_path)

    metadata = {
        "task": task,
        "target": target,
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "dataset_hash": dataset_hash,
        "trained_at_utc": now_utc_iso(),
        "input_features": list(X.columns),
    }

    metrics = {
        "val_metric": {best_metric: best_metric_value},
        "test_report": test_report,
        "leaderboard": leaderboard,
    }

    version = registry.register(
        model_obj=best_pipe.named_steps["model"],
        preprocessor_obj=best_pipe.named_steps["pre"],
        metadata=metadata,
        metrics=metrics,
        reference_data_path=ref_path,
    ).version

    if bool(registry_cfg.get("auto_approve", True)):
        registry.approve(version)

    return TrainResult(
        best_model_name=best_name,
        best_metric_name=best_metric,
        best_metric_value=best_metric_value,
        leaderboard=leaderboard,
        version=version,
    )
