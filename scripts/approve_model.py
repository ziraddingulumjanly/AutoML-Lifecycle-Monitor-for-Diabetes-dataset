from __future__ import annotations
import argparse
from pathlib import Path
from src.mlops.registry import ModelRegistry

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="registry")
    p.add_argument("--version", default=None)
    p.add_argument("--latest", action="store_true")
    args = p.parse_args()

    reg = ModelRegistry(Path(args.registry))
    version = args.version
    if args.latest:
        version = reg.latest()
    if not version:
        raise SystemExit("No version specified and no latest model found.")
    reg.approve(version)
    print(f"Approved {version} as production")

if __name__ == "__main__":
    main()
