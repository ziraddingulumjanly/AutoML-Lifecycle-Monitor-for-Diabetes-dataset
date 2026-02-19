from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from src.mlops.train import train_from_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    res = train_from_config(cfg)
    print(f"Registered model version: {res.version}")
    print(f"Best model: {res.best_model_name} | {res.best_metric_name}={res.best_metric_value:.5f}")

if __name__ == "__main__":
    main()
