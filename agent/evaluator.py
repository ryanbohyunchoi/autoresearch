import json
from pathlib import Path


def read_metric(exp_dir: Path, metric_key: str = "auroc") -> float | None:
    """
    Read the primary metric from results.json written by train.py.
    Expected format: {"auroc": 0.82, "auprc": 0.71, ...}
    Returns None if the file is missing or the key is absent.
    """
    results_file = exp_dir / "results.json"
    if not results_file.exists():
        return None
    try:
        data = json.loads(results_file.read_text())
        return float(data[metric_key])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def load_history(results_root: Path, metric_key: str = "auroc") -> list[dict]:
    """Reconstruct experiment history from all completed exp_* directories."""
    history = []
    for exp_dir in sorted(results_root.glob("exp_*")):
        metric = read_metric(exp_dir, metric_key)
        if metric is None:
            continue
        desc_file = exp_dir / "description.txt"
        description = desc_file.read_text().strip() if desc_file.exists() else "unknown"
        history.append({"exp": exp_dir.name, "metric": metric, "description": description})
    return history
