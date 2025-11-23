<!-- Auto-generated guidance for coding agents. Update as the project evolves. -->
# RLGymBot — Copilot Instructions

**Purpose:** Quick, actionable rules and project-specific context for AI coding agents working in this repository.

**Big Picture:**
- **Project layout:** main RL pipeline lives under `mysim/`, training code under `training/`, runtime/agent integration under `rlbot/`, configs in `configs/`, helper scripts in `scripts/`, and checkpoints/artifacts in `data/checkpoints` or `training/checkpoints`.
- **Core flow:** YAML config (`configs/*.yaml`) → training entrypoint (`python -m training.ppo_learner --config configs/training.yaml`) → checkpoints in `training/checkpoints` → `scripts/export_latest_policy.py` to create an artifact that RLBot/GUI loads.
- **External integration:** RocketSim (native/CMake project at `RocketSim/`) and RLBot are external dependencies; confirm builds for native code before assuming simulator availability.

**Key developer workflows (explicit commands):**
- Create & activate venv (Windows PowerShell):
```
python -m venv .venv; .venv\Scripts\activate
```
- Install deps and run tests:
```
pip install -r requirements.txt
pytest -q
```
- Train (example):
```
python -m training.ppo_learner --config configs/training.yaml
```
- Export latest policy (after training):
```
python scripts/export_latest_policy.py
```

**Project-specific conventions & patterns (do not change without checking usages):**
- `mysim/` is organized by responsibilities: `obs_builders/`, `reward_functions/`, `action_parsers/`, `state_setters/`, `terminal_conditions/`. Configs refer to implementations by name in `configs/*.yaml` (for example `env.reward.name: strong_hit_sqrt` in `configs/training.yaml`). When adding a new component, place it in the appropriate subpackage and mirror the config name.
- `tests/conftest.py` injects project root onto `sys.path` so tests import `mysim` directly. Run `pytest` from the `rlgymbotv2` root.
- Checkpoint / export locations: `training/checkpoints` (config `checkpoint_dir`) and exported artifacts are expected by the RLBot agent loader (see README quickstart).
- Small helper scripts live in `scripts/`; prefer adding small utility CLIs here and referencing them in the README.

**Code / change guidance for agents:**
- When you need concrete examples, inspect `mysim/obs_builders/` and `mysim/reward_functions/` for existing implementation patterns (function/class names, signature shapes). Match their public API rather than introducing deviating signatures.
- Config-driven behavior: prefer reading and updating `configs/*.yaml` for behavioral changes rather than scattering constants across modules.
- Tests: add pytest tests under `tests/` and ensure imports work without modifying `conftest.py`. Keep tests runnable via `pytest -q`.

**Integration & build notes:**
- RocketSim contains native code and uses CMake. If your change requires rebuilding the simulator, follow instructions in `RocketSim/README.md` before running training.
- Check `requirements.txt` for Python dependencies (RLBot, rlgym, wandb). Confirm versions before upgrading.

**Files to reference when editing or extending:**
- `configs/training.yaml` — canonical training config example.
- `training/ppo_learner.py` (entrypoint is `python -m training.ppo_learner`) — change here to affect training loop.
- `mysim/obs_builders/` and `mysim/reward_functions/` — add new observation or reward implementations.
- `scripts/export_latest_policy.py` — packaging/export logic for RLBot usage.
- `tests/conftest.py` — test import behavior.

**How to handle ambiguous cases:**
- If a config key exists (see `configs/*.yaml`), prefer modifying configs first. If behavior still requires code changes, place the implementation in `mysim/` and update the config to reference it.
- For simulator failures, check whether RocketSim native binaries exist; consult `RocketSim/README.md` for build steps and do not assume RocketSim is prebuilt in CI.

If anything here is unclear or you want more detail about a specific area (training loop, YAML schema, or native build steps), tell me which section to expand and I'll iterate.
