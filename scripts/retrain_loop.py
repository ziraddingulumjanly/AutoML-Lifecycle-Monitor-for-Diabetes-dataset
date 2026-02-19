from __future__ import annotations

import json
from pathlib import Path
import shutil
import yaml

from src.mlops.registry import ModelRegistry
from src.mlops.train import train_from_config

TRIGGER = Path("monitoring/triggers/retrain_trigger.json")

def compare_models(reg: ModelRegistry, cand_version: str, prod_version: str) -> bool:
    # Simple compare: prefer better val metric if available
    import json
    cand_metrics = json.loads((reg.models_dir / cand_version / "metrics.json").read_text(encoding="utf-8"))
    prod_metrics = json.loads((reg.models_dir / prod_version / "metrics.json").read_text(encoding="utf-8"))

    def score(m):
        vm = m.get("val_metric", {})
        if not vm:
            return None, None
        name = list(vm.keys())[0]
        val = vm[name]
        return name, float(val)

    c_name, c_val = score(cand_metrics)
    p_name, p_val = score(prod_metrics)

    if c_name is None or p_name is None:
        return False

    if c_name != p_name:
        # fallback: don't auto-promote if incomparable
        return False

    if c_name == "rmse":
        return c_val < p_val
    return c_val > p_val

def main():
    if not TRIGGER.exists():
        print("No retrain trigger present. Exiting.")
        return

    cfg = yaml.safe_load(Path("configs/train.yaml").read_text(encoding="utf-8"))
    reg = ModelRegistry(Path(cfg["registry"]["dir"]))

    prod = reg.get_production_version()
    if not prod:
        prod = reg.latest()

    print("Trigger found. Retraining...")
    res = train_from_config(cfg)
    cand = res.version
    print("Candidate version:", cand)

    if prod:
        promote = compare_models(reg, cand, prod)
    else:
        promote = True

    if promote:
        reg.approve(cand)
        print(f"Promoted {cand} to production")
    else:
        print(f"Kept production {prod}; candidate {cand} not better")

    # archive trigger
    TRIGGER.rename(TRIGGER.with_name("retrain_trigger.processed.json"))

if __name__ == "__main__":
    main()
