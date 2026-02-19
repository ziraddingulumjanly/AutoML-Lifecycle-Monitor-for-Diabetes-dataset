from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib

from .utils import safe_mkdir, now_utc_iso, json_dumps

@dataclass
class RegisteredModel:
    version: str
    model_path: Path
    preprocessor_path: Path
    metadata_path: Path
    metrics_path: Path

class ModelRegistry:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.models_dir = self.root_dir / "models"
        self.production_pointer = self.root_dir / "production.json"
        safe_mkdir(self.models_dir)

    def _next_version(self) -> str:
        existing = sorted([p.name for p in self.models_dir.glob("v*/") if p.is_dir()])
        if not existing:
            return "v0001"
        last = existing[-1]
        n = int(last.replace("v", ""))
        return f"v{n+1:04d}"

    def register(
        self,
        model_obj,
        preprocessor_obj,
        metadata: Dict,
        metrics: Dict,
        reference_data_path: Optional[Path] = None,
    ) -> RegisteredModel:
        version = self._next_version()
        out = self.models_dir / version
        safe_mkdir(out)

        model_path = out / "model.joblib"
        pre_path = out / "preprocessor.joblib"
        meta_path = out / "metadata.json"
        metrics_path = out / "metrics.json"

        joblib.dump(model_obj, model_path)
        joblib.dump(preprocessor_obj, pre_path)

        metadata = dict(metadata)
        metadata.update({
            "version": version,
            "registered_at_utc": now_utc_iso(),
        })
        meta_path.write_text(json_dumps(metadata), encoding="utf-8")
        metrics_path.write_text(json_dumps(metrics), encoding="utf-8")

        if reference_data_path and reference_data_path.exists():
            (out / "reference.csv").write_bytes(reference_data_path.read_bytes())

        return RegisteredModel(version, model_path, pre_path, meta_path, metrics_path)

    def list_versions(self):
        return sorted([p.name for p in self.models_dir.glob("v*/") if p.is_dir()])

    def latest(self) -> Optional[str]:
        versions = self.list_versions()
        return versions[-1] if versions else None

    def approve(self, version: str) -> None:
        pointer = {"production_version": version, "approved_at_utc": now_utc_iso()}
        self.production_pointer.write_text(json_dumps(pointer), encoding="utf-8")

    def get_production_version(self) -> Optional[str]:
        if not self.production_pointer.exists():
            return None
        d = json.loads(self.production_pointer.read_text(encoding="utf-8"))
        return d.get("production_version")

    def resolve_version(self) -> Optional[str]:
        return self.get_production_version() or self.latest()

    def load(self, version: Optional[str] = None):
        version = version or self.resolve_version()
        if not version:
            raise FileNotFoundError("No model version found in registry (no models registered yet).")
        p = self.models_dir / version
        model = joblib.load(p / "model.joblib")
        pre = joblib.load(p / "preprocessor.joblib")
        metadata = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((p / "metrics.json").read_text(encoding="utf-8"))
        ref_csv = p / "reference.csv"
        return model, pre, metadata, metrics, (ref_csv if ref_csv.exists() else None)
