# End-to-End Production-Style MLOps Project (AutoML + Registry + FastAPI + Docker + K8s + Monitoring)

This repository is generated to run an **end-to-end lifecycle**:
<img width="1919" height="308" alt="Screenshot 2026-02-19 190622" src="https://github.com/user-attachments/assets/9d91ff15-f6fc-4259-8829-faf60db25688" />

**DATA → TRAIN → PACKAGE → SERVE → CONTAINERIZE → DEPLOY → MONITOR → DECIDE → RETRAIN LOOP**

It is designed for **tabular datasets** out of the box. (Text/images can be added as extensions; scaffolding is included.)

---

## 0) Quickstart (your provided dataset)

A sample dataset file is expected at:

- `data/dataset.csv`

If you have a CSV already (e.g. `diabetes_prediction_dataset.csv`), copy it:

```powershell
mkdir data
Copy-Item -Path .\diabetes_prediction_dataset.csv -Destination .\data\dataset.csv
```

Then train + register a model:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts\train.py --config configs\train.yaml
python scripts\approve_model.py --latest
```

Start the stack:

```powershell
docker compose up --build
```

API:
- http://localhost:8000/health
- http://localhost:8000/docs
- http://localhost:8000/metrics

Prometheus:
- http://localhost:9090

Grafana:
- http://localhost:3000 (admin/admin)

Evidently HTML drift reports:
- `reports/` (mounted volume)

---

## 1) Project Structure

- `src/mlops/` — ingestion, schema, preprocessing, training, evaluation, registry
- `api/` — FastAPI service (dynamic model loading, validation, logging, Prometheus metrics)
- `monitoring/` — Evidently drift job + decision rules + trigger file
- `scripts/` — CLI utilities: ingest, train, approve, simulate production, retrain loop
- `docker/` — Dockerfiles, Prometheus config, Grafana provisioning + dashboard
- `k8s/` — Kubernetes manifests (Deployment, Service, ConfigMap, HPA)

---

## 2) Model Registry (File-based)

Registry layout:

```
registry/
  models/
    v0001/
      model.joblib
      preprocessor.joblib
      metadata.json
      metrics.json
  production.json     # pointer to approved model version
```

The API always loads the model referenced by `registry/production.json` (if present), otherwise the latest model.

---

## 3) Monitoring + Decision Logic

- API logs all predictions as JSON lines to `logs/predictions.jsonl`
- Monitoring job loads:
  - reference dataset snapshot from the registry
  - recent production inputs/predictions from logs
- Evidently generates drift reports to `reports/`
- Decision logic writes a trigger file if drift exceeds threshold:
  - `monitoring/triggers/retrain_trigger.json`

The `scripts/retrain_loop.py` demonstrates an automated retrain + compare + promote flow.

---

## 4) Kubernetes

Manifests are in `k8s/`.

Typical flow:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

For an end-to-end cluster monitoring stack, use your existing Prometheus/Grafana operator or adapt `docker/` configs.

---

## 5) Notes

- XGBoost is treated as optional; the pipeline works without it.
- This project is intentionally self-contained (file-based registry) to run anywhere.

