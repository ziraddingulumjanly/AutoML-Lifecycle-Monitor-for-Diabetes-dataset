from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------
# core sender
# ---------------------------------------------------
def send_batches(api: str, records: list[dict]):
    for i in range(0, len(records), 25):
        batch = records[i:i + 25]
        r = requests.post(api, json={"records": batch}, timeout=30)
        r.raise_for_status()
        print("sent", i, "->", i + len(batch), "version", r.json().get("version"))


# ---------------------------------------------------
# create records from dataset
# ---------------------------------------------------
def generate_records(data_path: str, n: int) -> list[dict]:
    df = pd.read_csv(data_path)

    # detect likely target column
    candidates = [c for c in df.columns if c.lower() in {"target", "label", "y", "outcome", "class", "diabetes"}]
    target = candidates[0] if candidates else df.columns[-1]

    sample = df.sample(min(len(df), n))
    sample = sample.drop(columns=[target], errors="ignore")

    return sample.to_dict(orient="records")


# ---------------------------------------------------
# batch mode (one-time)
# ---------------------------------------------------
def simulate_once(api: str, data: str, n: int):
    records = generate_records(data, n)
    send_batches(api, records)


# ---------------------------------------------------
# continuous production users
# ---------------------------------------------------
def live_mode(api: str, data: str):
    print("Starting continuous traffic (CTRL+C to stop)")

    while True:
        try:
            # random number of users each cycle
            n = random.randint(5, 40)
            records = generate_records(data, n)
            send_batches(api, records)

            # human-like pause
            sleep_time = random.uniform(0.5, 3.0)
            print(f"sleep {sleep_time:.2f}s")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped live traffic.")
            break
        except Exception as e:
            print("temporary error:", e)
            time.sleep(2)


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000/predict")
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--live", action="store_true")   # <-- new argument

    args = parser.parse_args()

    if args.live:
        live_mode(args.api, args.data)
    else:
        simulate_once(args.api, args.data, args.n)


if __name__ == "__main__":
    main()
