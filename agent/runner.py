import subprocess
import time
import shutil
from pathlib import Path


def submit_job(exp_dir: Path, slurm_template: Path) -> str:
    """Copy the SLURM template into exp_dir and submit it; return the job ID."""
    job_script = exp_dir / "job.sh"
    content = slurm_template.read_text()
    content = content.replace("{{EXP_DIR}}", str(exp_dir))
    job_script.write_text(content)

    result = subprocess.run(
        ["sbatch", str(job_script)],
        capture_output=True, text=True, check=True,
    )
    # sbatch output: "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    return job_id


def wait_for_job(job_id: str, poll_interval: int = 30, timeout: int = 3600) -> bool:
    """Block until the SLURM job finishes. Returns True if succeeded."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True,
        )
        state = result.stdout.strip()
        if not state:
            return True  # job no longer in queue → finished
        if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            return False
        time.sleep(poll_interval)
    return False  # timed out


def copy_train_script(src: Path, exp_dir: Path) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    dst = exp_dir / "train.py"
    shutil.copy2(src, dst)
    return dst
