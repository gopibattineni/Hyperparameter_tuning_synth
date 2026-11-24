#!/usr/bin/env bash
# Per-generator status for Standard TPE / Multivariate TPE / Gaussian Process.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
"$PY" - <<'PY'
from pathlib import Path
import json
import optuna
from optuna.trial import TrialState
from scripts.run_experiments import RESEARCH_GENERATORS

DATASETS = [("cancer", "1_cancer"), ("alzheimers", "2_alzheimers")]
SAMPLERS = [
    ("Standard TPE", "standard_tpe/results"),
    ("Multivariate TPE", "multivariate_tpe/results"),
    ("Gaussian Process", "gaussian_process/results"),
]

def trial_counts(db: Path):
    if not db.is_file():
        return None
    try:
        st = optuna.study.get_all_study_summaries(storage=f"sqlite:///{db.resolve()}")
        if not st:
            return None
        study = optuna.load_study(study_name=st[0].study_name, storage=f"sqlite:///{db.resolve()}")
        c = {s.name: sum(1 for t in study.trials if t.state == s) for s in TrialState}
        vals = [t.value for t in study.trials if t.value is not None]
        return len(study.trials), c, (max(vals) if vals else None)
    except Exception as e:
        return ("err", str(e), None)

print(f"{'method':<18} {'dataset':<12} {'generator':<18} {'status':<12} {'trials':<18} best")
print("-" * 95)
for label, root in SAMPLERS:
    for ds, folder in DATASETS:
        for gen in RESEARCH_GENERATORS:
            d = Path(root) / "classification" / folder / gen
            if (d / "completed.json").is_file():
                meta = json.loads((d / "completed.json").read_text())
                best = meta.get("best_value")
                status, trials = "completed", "-"
            elif (d / "failed.json").is_file():
                status, best, trials = "failed", None, "-"
            elif d.is_dir():
                info = trial_counts(d / "optuna.db")
                if info and info[0] != "err":
                    n, c, best = info
                    status = "running" if c.get("RUNNING") else "started"
                    trials = f"{c.get('COMPLETE', 0)}C/{c.get('RUNNING', 0)}R/{n}"
                else:
                    status, best, trials = "pending", None, "-"
            else:
                status, best, trials = "missing", None, "-"
            b = f"{best:.4f}" if isinstance(best, float) else (best or "—")
            print(f"{label:<18} {ds:<12} {gen:<18} {status:<12} {trials:<18} {b}")

# Forest workers live check
print()
print("live forest processes:")
import subprocess
r = subprocess.run(["pgrep", "-af", "generators forest_diffusion|forest_diffusion --quiet"], capture_output=True, text=True)
print(r.stdout.strip() or "(none)")
PY
