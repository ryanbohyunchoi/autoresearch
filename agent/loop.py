"""
Main autoresearch agent loop.

Usage:
    python -m agent.loop \
        --target targets/example \
        --results results/my_run \
        --slurm slurm/train_job.sh \
        --metric auroc \
        --n-experiments 20
"""
import argparse
import json
import sys
from pathlib import Path

from agent import evaluator, llm, runner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="Dir with train.py + program.md")
    p.add_argument("--results", required=True, help="Output dir for experiments")
    p.add_argument("--slurm", required=True, help="SLURM job template (job.sh)")
    p.add_argument("--metric", default="auroc", help="Key in results.json to maximize")
    p.add_argument("--n-experiments", type=int, default=20)
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--poll-interval", type=int, default=60, help="Seconds between squeue polls")
    p.add_argument("--job-timeout", type=int, default=7200, help="Max seconds to wait per job")
    return p.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.target)
    results_root = Path(args.results)
    slurm_template = Path(args.slurm)
    results_root.mkdir(parents=True, exist_ok=True)

    program_md = (target_dir / "program.md").read_text()
    baseline_train = (target_dir / "train.py").read_text()
    current_train = baseline_train

    history = evaluator.load_history(results_root, args.metric)
    best_metric = max((h["metric"] for h in history), default=float("-inf"))
    print(f"[loop] Resuming. {len(history)} past experiments. Best {args.metric}: {best_metric:.4f}")

    for i in range(len(history), len(history) + args.n_experiments):
        exp_name = f"exp_{i:03d}"
        exp_dir = results_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[loop] === Experiment {exp_name} ===")

        # Propose modification
        print("[loop] Asking LLM for modification...")
        modified_train = llm.propose_modification(
            program_md=program_md,
            current_code=current_train,
            history=history,
            model=args.model,
        )
        description = llm.describe_change(current_train, modified_train)
        print(f"[loop] Change: {description}")

        # Save modified script and metadata
        (exp_dir / "train.py").write_text(modified_train)
        (exp_dir / "description.txt").write_text(description)
        (exp_dir / "parent.txt").write_text(
            history[-1]["exp"] if history else "baseline"
        )

        # Submit + wait
        print("[loop] Submitting SLURM job...")
        try:
            job_id = runner.submit_job(exp_dir, slurm_template)
            print(f"[loop] Job {job_id} submitted. Waiting...")
            success = runner.wait_for_job(
                job_id,
                poll_interval=args.poll_interval,
                timeout=args.job_timeout,
            )
        except subprocess.CalledProcessError as e:
            print(f"[loop] sbatch failed: {e}. Skipping.")
            continue

        if not success:
            print("[loop] Job failed or timed out. Skipping.")
            continue

        # Evaluate
        metric = evaluator.read_metric(exp_dir, args.metric)
        if metric is None:
            print("[loop] No results.json found. Skipping.")
            continue

        print(f"[loop] {args.metric} = {metric:.4f}  (best so far: {best_metric:.4f})")
        entry = {"exp": exp_name, "metric": metric, "description": description}
        history.append(entry)

        if metric > best_metric:
            best_metric = metric
            current_train = modified_train  # keep improving from best
            print(f"[loop] New best! Continuing from this version.")
        else:
            print(f"[loop] No improvement. Keeping previous best as base.")

    print(f"\n[loop] Done. Best {args.metric}: {best_metric:.4f}")


if __name__ == "__main__":
    import subprocess  # needed for runner
    main()
