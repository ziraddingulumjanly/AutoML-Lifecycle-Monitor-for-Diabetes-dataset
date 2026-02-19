from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Local path or URL to dataset")
    p.add_argument("--dest", default="data/dataset.csv", help="Destination path under the repo")
    args = p.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if args.source.startswith("http://") or args.source.startswith("https://"):
        r = requests.get(args.source, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"Downloaded dataset to {dest}")
    else:
        src = Path(args.source)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dest)
        print(f"Copied dataset to {dest}")

if __name__ == "__main__":
    main()
