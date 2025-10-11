################################
# Date: 10/10/2025
# Revision: 1.0.0
# Description: Initial Release
# Title: policy_loader.py
# Author: NJH
################################
"""
Policy loading and inference wrapper for RLBot agent.
Supports two artifact styles from `export_policy.py`:
- SB3 zip (manifest.model_type == 'sb3')
- Raw torch state_dict (manifest.model_type == 'torch')

You must provide `build_observation(packet)` in agent.py to map RLBot
packets to your observation tensor.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover
    PPO = None


@dataclass
class LoadedPolicy:
    kind: str  # 'sb3' or 'torch'
    model: Any
    obs_norm: Optional[dict]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.kind == "sb3":
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        elif self.kind == "torch":
            if torch is None:
                raise RuntimeError("PyTorch not available to run torch policy")
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                out = self.model(x)
                # Assume model returns action tensor in [-1,1] or logits
                return out.squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unknown policy kind: {self.kind}")


def load_latest_artifact(artifacts_root: Path = Path("artifacts")) -> LoadedPolicy:
    manifest_path = artifacts_root / "latest" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No artifacts/latest/manifest.json found. Run export_policy.py first.")
    manifest = json.loads(manifest_path.read_text())
    kind = manifest["model_type"]

    if kind == "sb3":
        if PPO is None:
            raise RuntimeError("stable-baselines3 not installed; cannot load SB3 artifact")
        zip_name = manifest["files"]["sb3_zip"]
        model = PPO.load(str(manifest_path.parent / zip_name))
        return LoadedPolicy(kind="sb3", model=model, obs_norm=manifest.get("obs_norm"))

    if kind == "torch":
        if torch is None:
            raise RuntimeError("PyTorch not installed; cannot load torch artifact")
        # You must import/instantiate your policy architecture here
        from training.ppo_learner import Policy  # <-- ensure this exists
        state_file = (manifest_path.parent / manifest["files"].get("torch_state_dict", "policy_state.pt"))
        policy = Policy()  # adjust constructor args as needed
        state = torch.load(state_file, map_location="cpu")
        policy.load_state_dict(state, strict=False)
        policy.eval()
        return LoadedPolicy(kind="torch", model=policy, obs_norm=manifest.get("obs_norm"))

    raise ValueError(f"Unsupported model_type in manifest: {kind}")