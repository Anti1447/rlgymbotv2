import json, re
from pathlib import Path
from datetime import datetime
import os
import shutil

def make_run_dir(root: Path, prefix="rlgym-ppo-run"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{prefix}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def latest_checkpoint_folder(run_dir: Path):
    if not run_dir or not run_dir.exists():
        return None
    nums = [int(p.name) for p in run_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    return run_dir / str(max(nums)) if nums else None

def read_meta(run_dir: Path):
    try:
        with open(run_dir / "meta.json", "r") as f:
            return json.load(f)
    except Exception:
        return None

def write_meta(run_dir: Path, **data):
    with open(run_dir / "meta.json", "w") as f:
        json.dump(data, f, indent=2)

def shapes_match(meta, obs_dim, policy_layers, critic_layers):
    if not meta:
        return False
    return (
        meta.get("obs_dim") == obs_dim and
        meta.get("policy_layers") == policy_layers and
        meta.get("critic_layers") == critic_layers
    )

def summarize_checkpoints(root: Path):
    """
    Print a concise summary of each run and its latest checkpoint.
    Includes timesteps, mean reward, and date if available.
    """
    runs = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    print("\n=== Checkpoint Summary ===")
    for i, run in enumerate(runs):
        meta = read_meta(run)
        ckpt = latest_checkpoint_folder(run)
        steps = None
        reward = None
        timestamp = None
        if meta:
            steps = meta.get("total_timesteps") or "?"
            reward = meta.get("avg_reward") or "?"
            timestamp = meta.get("timestamp") or run.stat().st_mtime
        print(f"[{i}] {run.name}")
        print(f"    ├─ Checkpoint: {ckpt.name if ckpt else 'N/A'}")
        print(f"    ├─ Steps: {steps:,}" if isinstance(steps, int) else f"    ├─ Steps: {steps}")
        print(f"    ├─ Avg Reward: {reward}")
        print(f"    └─ Date: {timestamp}")

def choose_checkpoint(root: Path, include_milestones=False):
    runs = []
    # Gather main training runs
    runs.extend(sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name != "milestones"],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    ))

    # Optionally include milestone folders
    if include_milestones:
        milestone_root = root / "milestones"
        if milestone_root.exists():
            for phase_dir in milestone_root.iterdir():
                if phase_dir.is_dir():
                    runs.extend(sorted(
                        [p for p in phase_dir.iterdir() if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    ))

    print("\nAvailable checkpoints:")
    for i, run in enumerate(runs):
        ckpt = latest_checkpoint_folder(run)
        category = "Milestone" if "milestones" in str(run) else "Checkpoint"
        print(f"[{i}] ({category}) {run.relative_to(root)} -> {ckpt.name if ckpt else 'N/A'}")
    sel = input("Select checkpoint index to resume (Enter for latest): ").strip()
    if sel.isdigit():
        run = runs[int(sel)]
        return run, latest_checkpoint_folder(run)
    return runs[0], latest_checkpoint_folder(runs[0])

