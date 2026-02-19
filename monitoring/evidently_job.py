from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.mlops.registry import ModelRegistry
from src.mlops.utils import safe_mkdir, now_utc_iso

# Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REGISTRY_DIR = Path("registry")
LOGS = Path("logs") / "predictions.jsonl"
REPORTS_DIR = Path("reports")
TRIGGERS_DIR = Path("monitoring/triggers")
safe_mkdir(REPORTS_DIR)
safe_mkdir(TRIGGERS_DIR)

DRIFT_THRESHOLD_SHARE = 0.35  # if >35% columns drift -> trigger
MIN_ROWS = 50

def _load_reference(reg: ModelRegistry) -> Optional[pd.DataFrame]:
    version = reg.resolve_version()
    if not version:
        return None
    ref = reg.models_dir / version / "reference.csv"
    if not ref.exists():
        return None
    return pd.read_csv(ref)

def _load_current(metadata) -> Optional[pd.DataFrame]:
    if not LOGS.exists():
        return None
    rows = []
    with LOGS.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                rows.append(j["features"])
            except Exception:
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # align features
    feats = metadata.get("input_features", [])
    if feats:
        for c in feats:
            if c not in df.columns:
                df[c] = None
        df = df[feats]
    return df

def decide_and_trigger(drift_share: float, detail: dict) -> None:
    if drift_share >= DRIFT_THRESHOLD_SHARE:
        payload = {
            "triggered_at_utc": now_utc_iso(),
            "reason": "data_drift",
            "drift_share": drift_share,
            "threshold": DRIFT_THRESHOLD_SHARE,
            "detail": detail,
        }
        (TRIGGERS_DIR / "retrain_trigger.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

def run_once() -> None:
    reg = ModelRegistry(REGISTRY_DIR)
    version = reg.resolve_version()
    if not version:
        return
    model, pre, metadata, metrics, ref_path = reg.load(version)
    ref = _load_reference(reg)
    cur = _load_current(metadata)
    if ref is None or cur is None:
        return
    if len(cur) < MIN_ROWS or len(ref) < MIN_ROWS:
        return

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref[metadata["input_features"]], current_data=cur)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_html = REPORTS_DIR / f"data_drift_{version}_{ts}.html"
    report.save_html(str(out_html))

    # parse drift share
    j = report.as_dict()
    # Evidently dict structure can vary by version; handle defensively.
    drifted_cols = 0
    total_cols = len(metadata["input_features"])
    try:
        metrics_list = j.get("metrics", [])
        # find data drift metric result
        for m in metrics_list:
            if "result" in m and isinstance(m["result"], dict):
                if "drift_by_columns" in m["result"]:
                    drift_by = m["result"]["drift_by_columns"]
                    drifted_cols = sum(1 for _, v in drift_by.items() if isinstance(v, dict) and v.get("drift_detected") is True)
                    break
    except Exception:
        drifted_cols = 0
    drift_share = (drifted_cols / total_cols) if total_cols else 0.0

    decide_and_trigger(drift_share, {"drifted_cols": drifted_cols, "total_cols": total_cols, "report": out_html.name})

def main():
    interval_seconds = int(Path("monitoring/interval_seconds.txt").read_text().strip()) if Path("monitoring/interval_seconds.txt").exists() else 60
    while True:
        try:
            run_once()
        except Exception:
            pass
        time.sleep(interval_seconds)

if __name__ == "__main__":
    main()
