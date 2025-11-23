from pathlib import Path
import shutil
import re
from rlgymbotv2.mysim.training_utils.checkpoint_utils import latest_checkpoint_folder, read_meta
from rlgymbotv2.mysim.debug_config import dprint, global_debug_mode, debug_checkpoints


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

def choose_milestone(checkpoint_root: Path):
    """
    Interactive selector for milestone-based checkpoint resuming.

    Expected directory structure:

        data/checkpoints/
            milestones/
                Phase1-BallTouching/
                    005M/
                    010M/
                    ...
                Phase2-BronzeSkills1/
                    015M/
                    020M/
                Phase3-SilverSkills1/
                    005M/
                    ...

    Returns:
        (run_dir, checkpoint_folder)
    """
    def dbg(*args, **kwargs):
        # Only print if checkpoint debugging is on or forced
        if debug_checkpoints or global_debug_mode:
            dprint(*args, **kwargs)

    dbg("\n[DEBUG] choose_milestone() called")
    dbg("[DEBUG] checkpoint_root =", checkpoint_root.resolve())

    milestone_root = checkpoint_root / "milestones"
    dbg("[DEBUG] milestone_root =", milestone_root.resolve())

    dbg("[DEBUG] Listing contents of milestone_root:")
    if milestone_root.exists():
        for p in milestone_root.iterdir():
            dbg("   [DEBUG] -", repr(p.name), "| is_dir:", p.is_dir())
    else:
        dbg("[DEBUG] milestone_root DOES NOT EXIST")
        print("[milestones] No milestone root found.")
        return None, None

    # --- 1. Detect valid milestone directories ---
    milestone_dirs = []

    dbg("\n[DEBUG] Scanning for Phase*- folders:")
    for d in milestone_root.iterdir():
        dbg("   [DEBUG] Checking folder:", repr(d.name))
        if not d.is_dir():
            dbg("   [DEBUG]   -> NOT a directory")
            continue

        name = d.name.strip()
        dbg("   [DEBUG]   stripped name:", repr(name))

        m = re.match(r"^Phase(\d+)-(.+)$", name)
        dbg("   [DEBUG]   regex match:", bool(m))

        if not m:
            dbg("   [DEBUG]   SKIPPING", repr(name))
            continue

        phase = int(m.group(1))
        display_name = m.group(2)
        milestone_dirs.append((phase, display_name, d))
        dbg("   [DEBUG]   VALID PHASE FOLDER FOUND")

    if not milestone_dirs:
        print("[milestones] No valid milestones found.")
        return None, None

    # --- 2. Sort by phase number ---
    milestone_dirs.sort(key=lambda x: x[0])

    print("\n=== Available Milestones ===")
    flat_list = []  # (phase_dir, checkpoint_dir)

    # --- 3. Display phases and their checkpoints ---
    for i, (phase, name, pdir) in enumerate(milestone_dirs):
        label = f"Phase{phase}-{name}"
        print(f"[{i}] {label}")
        dbg(f"[DEBUG]   Looking inside phase folder {repr(pdir.name)} (path={pdir.resolve()})")

        ckpts = []
        for c in pdir.iterdir():
            dbg("       [DEBUG] subfolder:", repr(c.name), "| is_dir:", c.is_dir())
            if not c.is_dir():
                continue

            if re.match(r"^\d+M$", c.name):
                dbg("       [DEBUG]       -> MATCHED checkpoint folder")
                ckpts.append(c)
            else:
                dbg("       [DEBUG]       -> not a checkpoint")

        # sort checkpoint names numerically (005M, 010M, etc.)
        ckpts.sort(key=lambda p: int(re.match(r"(\d+)M", p.name).group(1)) if re.match(r"(\d+)M", p.name) else 0)

        for j, ck in enumerate(ckpts):
            print(f"    ├─ ({i}.{j}) {ck.name}")
            flat_list.append((pdir, ck))

        if not ckpts:
            print("    ├─ No checkpoint subfolders found")

    if not flat_list:
        print("[milestones] No checkpoint folders found inside milestones.")
        return None, None

    # --- 4. Prompt user for selection ---
    choice = input("\nSelect checkpoint index (e.g., 0.3): ").strip()

    dbg("[DEBUG] User raw choice:", repr(choice))

    if "." not in choice:
        print("[milestones] Invalid format. Expected something like 0.2")
        return None, None

    phase_idx_str, ckpt_idx_str = choice.split(".", 1)
    try:
        phase_idx = int(phase_idx_str)
        ckpt_idx = int(ckpt_idx_str)
    except ValueError:
        print("[milestones] Invalid numeric selection.")
        return None, None

    dbg(f"[DEBUG] Parsed indices: phase_idx={phase_idx}, ckpt_idx={ckpt_idx}")

    # Map (phase_idx, ckpt_idx) to actual directory
    for i, (phase, name, pdir) in enumerate(milestone_dirs):
        ckpts = sorted(
            [c for c in pdir.iterdir() if c.is_dir() and re.match(r"^\d+M$", c.name)],
            key=lambda p: int(re.match(r"(\d+)M", p.name).group(1)) if re.match(r"(\d+)M", p.name) else 0
        )

        if i == phase_idx:
            if ckpt_idx >= len(ckpts):
                print("[milestones] Checkpoint index out of range.")
                dbg(f"[DEBUG] ckpt_idx={ckpt_idx}, len(ckpts)={len(ckpts)}")
                return None, None

            selected = ckpts[ckpt_idx]
            dbg(f"[DEBUG] Selected phase dir: {pdir.resolve()}")
            dbg(f"[DEBUG] Selected checkpoint dir: {selected.resolve()}")
            return pdir, selected

    print("[milestones] Selection failed.")
    return None, None

