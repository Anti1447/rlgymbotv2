from pathlib import Path
import shutil
from mysim.training_utils.checkpoint_utils import latest_checkpoint_folder, read_meta


def make_milestone_dir(root: Path, phase: str, total_steps: int) -> Path:
    milestone_root = root / "milestones" / phase
    milestone_root.mkdir(parents=True, exist_ok=True)
    tag = f"{int(total_steps // 1_000_000):03d}M"
    milestone_dir = milestone_root / tag
    milestone_dir.mkdir(exist_ok=True)
    return milestone_dir

def promote_to_release(milestone_dir: Path):
    phase_folder = milestone_dir.parent
    release_folder = phase_folder.with_name(phase_folder.name + "-Releases")
    release_folder.mkdir(parents=True, exist_ok=True)
    shutil.copytree(milestone_dir, release_folder / milestone_dir.name, dirs_exist_ok=True)
    print(f"[release] 🚀 Promoted {milestone_dir.name} → {release_folder.name}")

def choose_milestone(root: Path):
    """
    Interactive selector for milestone checkpoints only.

    Searches all phase subfolders under data/checkpoints/milestones/,
    lists each milestone (e.g., Phase1-BallTouching/018M),
    and lets the user choose one to resume from.
    """
    milestone_root = root / "milestones"
    if not milestone_root.exists():
        print("[error] No milestones folder found.")
        return None, None

    # Collect all milestone directories
    milestones = []
    for phase_dir in milestone_root.iterdir():
        if not phase_dir.is_dir():
            continue
        for milestone_dir in sorted(
            [p for p in phase_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        ):
            milestones.append(milestone_dir)

    if not milestones:
        print("[warning] No milestone checkpoints found.")
        return None, None

    # Display menu
    print("\nAvailable milestones:")

    for i, m in enumerate(milestones):
        meta = read_meta(m)
        phase = meta.get("phase") if meta else "?"
        steps = meta.get("total_timesteps") if meta else "?"
        reward = meta.get("avg_reward") if meta else "?"
        print(f"[{i}] {m.relative_to(root)}  |  Steps: {steps}  |  Reward: {reward}")


    sel = input("Select milestone index to resume (Enter for latest): ").strip()
    if sel.isdigit():
        chosen = milestones[int(sel)]
    else:
        chosen = milestones[0]

    ckpt = latest_checkpoint_folder(chosen)
    if ckpt is None:
        # Milestone folder is itself the checkpoint root
        ckpt = chosen
    print(f"[resume] Using milestone: {chosen.relative_to(root)}")
    return chosen, ckpt

