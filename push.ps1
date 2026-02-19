# ===== CONFIG =====
$repo = "https://github.com/ziraddingulumjanly/AutoML-Lifecycle-Monitor-for-Diabetes-dataset.git"
$commit = "Initial MLOps diabetes pipeline"

# ===== GIT INIT =====
git init
git branch -M main

# ===== CREATE GITIGNORE =====
@"
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
env/

data/
*.csv
*.parquet

models/
artifacts/
*.pkl
*.joblib

logs/
mlruns/

.DS_Store
Thumbs.db
.vscode/
*.tar
.env
.env.*
"@ | Out-File -Encoding utf8 .gitignore

# ===== CONNECT REPO =====
git remote remove origin 2>$null
git remote add origin $repo

# ===== COMMIT =====
git add .
git commit -m $commit

# ===== PUSH =====
git push -u origin main