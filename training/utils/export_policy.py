################################
# Date: 10/10/2025
# Revision: 1.0.0
# Description: Initial Release. Exports latest PPO to on-disk policy
# Title: export_policy.py
# Author: NJH
################################
"""
Export the latest PPO checkpoint to a versioned `artifacts/` directory
along with normalization stats and a small manifest so the RLBot agent
can load it later.

Supported formats (auto-detected by file extension):
- Stable-Baselines3 PPO: .zip
- Raw PyTorch: .pt / .pth (expects a dict with 'model_state', optional 'obs_rms')

This script is intentionally lightweight and defensive. It does not assume
your exact learner implementation. Adjust the `CUSTOM_TORCH_POLICY_CLASS`
import to point at your policy class if you're exporting raw PyTorch.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Optional imports; we gate them to keep this script usable without SB3/torch
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # noqa: N816

try:
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover
    PPO = None  # noqa: N816

# If exporting raw torch, point this to your policy class
CUSTOM_TORCH_POLICY_CLASS = None  # e.g., from training.ppo_learner import Policy

CHECKPOINTS_DIR = Path("training/checkpoints")
ARTIFACTS_ROOT = Path("artifacts")
DEFAULT_EXPORT_NAME = "latest"


@dataclass
class ExportManifest:
    model_type: str  # 'sb3' or 'torch'
    created_utc: str
    source_checkpoint: str
    obs_norm: Optional[dict]
    files: dict
    obs_space: Optional[dict] = None
    action_space: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _find_latest_checkpoint(patterns=("*.zip", "*.pt", "*.pth")) -> Path:
    candidates = []
    for pat in patterns:
        candidates.extend(CHECKPOINTS_DIR.rglob(pat))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {CHECKPOINTS_DIR}/")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def export_latest(to: Optional[str] = None) -> Path:
    ckpt = _find_latest_checkpoint()
    ts = _timestamp_slug()
    version_dir = ARTIFACTS_ROOT / ts
    version_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    obs_norm = None

    if ckpt.suffix == ".zip":  # Stable-Baselines3 PPO
        if PPO is None:
            raise RuntimeError("stable-baselines3 is not installed but a .zip checkpoint was found.")
        model = PPO.load(str(ckpt))
        # Save the SB3 zip as-is (fast path) and a small manifest
        target_zip = version_dir / ckpt.name
        shutil.copy2(ckpt, target_zip)
        files["sb3_zip"] = target_zip.name

        # Try to capture running mean/std if present (VecNormalize)
        try:
            vecnorm = getattr(model, "get_vec_normalize_env", None)
            if callable(vecnorm):
                venv = vecnorm()
                if venv is not None:
                    obs_norm = {
                        "obs_rms_mean": getattr(venv.obs_rms, "mean", None).tolist() if getattr(venv, "obs_rms", None) is not None else None,
                        "obs_rms_var": getattr(venv.obs_rms, "var", None).tolist() if getattr(venv, "obs_rms", None) is not None else None,
                        "clip_obs": getattr(venv, "clip_obs", None),
                    }
        except Exception:
            pass

        manifest = ExportManifest(
            model_type="sb3",
            created_utc=ts,
            source_checkpoint=str(ckpt),
            obs_norm=obs_norm,
            files=files,
        )

    elif ckpt.suffix in {".pt", ".pth"}:  # Raw PyTorch
        if torch is None:
            raise RuntimeError("PyTorch is not installed but a .pt/.pth checkpoint was found.")
        raw = torch.load(str(ckpt), map_location="cpu")
        # Expect either: whole model, or a dict with model_state / obs_rms
        if isinstance(raw, dict) and "model_state" in raw:
            model_state = raw["model_state"]
            obs_norm = raw.get("obs_rms")
            model_file = version_dir / "policy_state.pt"
            torch.save(model_state, model_file)
            files["torch_state_dict"] = model_file.name
        else:
            # Unknown format; copy as-is
            target_pt = version_dir / ckpt.name
            torch.save(raw, target_pt)
            files["torch_blob"] = target_pt.name

        manifest = ExportManifest(
            model_type="torch",
            created_utc=ts,
            source_checkpoint=str(ckpt),
            obs_norm=obs_norm,
            files=files,
        )
    else:
        raise ValueError(f"Unsupported checkpoint type: {ckpt.suffix}")

    # Write manifest
    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(manifest.to_json())

    # Update/refresh artifacts/latest symlink or folder
    latest_link = ARTIFACTS_ROOT / DEFAULT_EXPORT_NAME
    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        latest_link.symlink_to(version_dir.name)
    except Exception:
        # Fallback: copy (Windows without dev-mode symlinks)
        if latest_link.exists():
            shutil.rmtree(latest_link)
        shutil.copytree(version_dir, latest_link)

    print(f"Exported to: {version_dir}")
    print(f"Manifest: {manifest_path}")
    return version_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="Optional artifacts/<version> directory name override")
    args = parser.parse_args()
    export_latest(args.out)