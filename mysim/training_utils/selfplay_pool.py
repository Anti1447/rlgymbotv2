from pathlib import Path
import re
import json
from typing import List, Optional, Tuple


class SelfPlayOpponentPool:
    """
    Keeps track of candidate opponent checkpoints for self-play evaluation.
    Works off your existing milestone structure:

        data/checkpoints/milestones/
            Phase1-BallTouching/005M
            Phase1-BallTouching/010M
            ...
            Phase3-SilverSkills1/025M
            ...

    Each leaf dir is expected to contain:
        - PPO_POLICY.pt
        - PPO_CRITIC.pt
        - meta.json (optional, but we try to read it)
    """

    def __init__(self, checkpoint_root: Path, min_steps: int = 5_000_000):
        self.checkpoint_root = checkpoint_root
        self.milestone_root = checkpoint_root / "milestones"
        self.min_steps = int(min_steps)
        self._cache: List[Tuple[int, Path]] = []  # (total_steps, folder)
        self.refresh()

    def refresh(self):
        self._cache.clear()
        if not self.milestone_root.exists():
            return

        for phase_dir in self.milestone_root.iterdir():
            if not phase_dir.is_dir():
                continue
            # Expect PhaseX-Name
            name = phase_dir.name.strip()
            if not re.match(r"^Phase\d+-", name):
                continue

            for ckpt_dir in phase_dir.iterdir():
                if not ckpt_dir.is_dir():
                    continue
                # Expect something like 005M, 010M, ...
                m = re.match(r"^(\d+)M$", ckpt_dir.name)
                if not m:
                    continue

                steps = int(m.group(1)) * 1_000_00  # 005M -> 500000, etc.
                # If you saved at exact ts (e.g., 5_000_000), you can
                # instead read meta.json:
                meta_path = ckpt_dir / "meta.json"
                if meta_path.exists():
                    try:
                        with meta_path.open("r") as f:
                            meta = json.load(f)
                        steps = int(meta.get("total_timesteps", steps))
                    except Exception:
                        pass

                if steps < self.min_steps:
                    continue

                policy_file = ckpt_dir / "PPO_POLICY.pt"
                if not policy_file.exists():
                    continue

                self._cache.append((steps, ckpt_dir))

        # sort oldest → newest
        self._cache.sort(key=lambda x: x[0])

    def list_opponents(self) -> List[Tuple[int, Path]]:
        return list(self._cache)

    def best_so_far(self) -> Optional[Path]:
        if not self._cache:
            return None
        # right now "best" = latest; later you can use avg_reward from meta
        return self._cache[-1][1]

    def pick_opponent(self, strategy: str = "latest") -> Optional[Path]:
        """
        strategy:
            'latest'  -> newest checkpoint
            'random'  -> random from pool
            'older'   -> sample from older 50% of pool (for variety)
        """
        import random

        if not self._cache:
            return None

        if strategy == "latest":
            return self._cache[-1][1]
        elif strategy == "random":
            return random.choice(self._cache)[1]
        elif strategy == "older":
            half = max(1, len(self._cache) // 2)
            return random.choice(self._cache[:half])[1]
        else:
            return self._cache[-1][1]
