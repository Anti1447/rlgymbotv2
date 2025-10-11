--------------------------------------
Date: 10/10/2025
Revision: 1.0.0
Description: Initial Release
Title: Roadmap
Author: NJH
--------------------------------------
# RLGymBotv2 — Project Mind Map, Goals & Roadmap

> High-level objective: Build a reproducible RL pipeline (training → export → playable bot) with strong version control, documentation, and measurable milestones.

---

## 0) Suggested Repository Structure

```
rlgym-bot/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ CHANGELOG.md
├─ ROADMAP.md
├─ pyproject.toml / requirements.txt
├─ configs/
│  ├─ training.yaml
│  └─ eval.yaml
├─ training/
│  ├─ ppo_learner.py            # renamed from example.py
│  ├─ envs/                     # custom RLGym env wrappers
│  │  ├─ __init__.py
│  │  └─ training_env.py
│  ├─ rewards/
│  │  ├─ __init__.py
│  │  └─ strong_hit_sqrt.py
│  ├─ checkpoints/              # saved PPO checkpoints
│  └─ utils/
│     ├─ rollout.py
│     └─ export_policy.py       # exports latest PPO → on-disk policy
├─ rlbot/
│  ├─ agents/
│  │  └─ ppo_hivemind_agent/
│  │     ├─ agent.py            # loads exported policy for live play
│  │     ├─ policy_loader.py
│  │     └─ requirements.txt
│  └─ training_playlists/       # RLBotTraining scripts
│     ├─ shadow_defense_training.py
│     ├─ kickoff_reach_training.py
│     └─ custom_playlist.py
├─ tests/
│  ├─ test_export_pipeline.py
│  └─ test_training_playlists.py
└─ scripts/
   ├─ run_training.sh / .ps1
   ├─ export_latest_policy.py
   └─ run_rlbot_agent.sh / .ps1
```

---

## 1) Goal: Create & Maintain the GitHub Repository

**Outcome**: Versioned, documented project with CI checks and clear contribution flow.

### Subtasks

* **Initialize repo**

  * Create remote GitHub repo `rlgym-bot` (private to start).
  * Add `.gitignore` (Python, RLBot, venv, checkpoints), `LICENSE` (MIT or Apache-2.0), and `README.md`.
  * Commit initial scaffolding (repo tree above).
* **Branching & conventions**

  * Main branch: `main` (protected).
  * Feature branches: `feat/*`, `fix/*`, `chore/*`.
  * Commit style: Conventional Commits (`feat:`, `fix:`, etc.).
* **CI (optional but recommended)**

  * GitHub Actions: run `pytest` on `tests/`, lint with `ruff`/`flake8`, type-check with `mypy` (optional).
* **Releases & tags**

  * Use semantic versioning (start at `v0.1.0`).
  * Generate release notes from CHANGELOG.

---

## 2) Goal: Organize Code & Rename Ambiguous Files

**Outcome**: Clear file names & structure to reduce cognitive load.

### Subtasks

* **Rename `example.py` → `ppo_learner.py`** (in `training/`).
* **Create `export_policy.py`**: script to save the latest PPO weights (policy + normalization stats) to disk in a format loadable by RLBot agent.
* **Create `policy_loader.py` + `agent.py`** under `rlbot/agents/ppo_hivemind_agent/` to load exported policy and act in RLBotGUI.
* **Add configs** in `configs/` for training/eval.

---

## 3) Goal: Include RLGym Simulator & Training Assets

**Outcome**: Reproducible training environment for experimentation.

### Subtasks

* **Environment wrappers**

  * `training/envs/training_env.py` for customized spawns, obs, action space, and reward assembly.
* **Rewards**

  * Implement `strong_hit_sqrt.py` (square-root scaled hit strength) and any others.
* **Dependencies**

  * Pin RLGym, numpy, torch, PPO library (e.g., SB3 or your custom learner) in `requirements.txt`/`pyproject.toml`.
* **Unit tests**

  * `tests/test_export_pipeline.py` ensures a dummy policy can export + reload.

---

## 4) Goal: RLBot Playable Agent from Latest PPO Policy

**Outcome**: “One command” to export the newest checkpoint and play in RLBotGUI against humans/bots.

### Subtasks

* **Export script**

  * `scripts/export_latest_policy.py`: finds latest checkpoint in `training/checkpoints/`, calls `export_policy.py`.
* **RLBot agent**

  * `rlbot/agents/ppo_hivemind_agent/agent.py`: loads exported weights and runs forward pass for decisions.
  * `policy_loader.py`: handles device selection (CPU/GPU), normalization, and action translation.
* **Launchers**

  * `scripts/run_rlbot_agent.sh` or PS1 to start RLBotGUI with the agent registered.
* **Smoke test**

  * `tests/test_training_playlists.py`: run a tiny RLBotTraining playlist and assert Pass/Fail grades to validate integrations.

---

## 5) Goal: Document Everything (Change Log + Roadmap)

**Outcome**: Clear, measurable progress that’s easy to revisit.

### Subtasks

* **CHANGELOG.md**

  * Use Keep a Changelog format.
  * Update per PR with Conventional Commit scopes.
* **ROADMAP.md**

  * Break down milestones, KPIs, and target capabilities.
  * Mark items with status (Planned / In Progress / Done).
* **README.md**

  * Quickstart (install, train, export, play).
  * Repo structure, links to ROADMAP and CHANGELOG.

---

## 6) Goal: Use ChatGPT with the GitHub Repo

**Outcome**: Faster iteration via AI assistance with context from code.

### Subtasks

* **Connect repo to ChatGPT**

  * Add the GitHub repo to a ChatGPT project so the assistant can read code and propose diffs.
* **PR Review workflow**

  * Ask ChatGPT to summarize PRs, suggest tests, and draft doc updates.
* **Issue drafting**

  * Generate issues from ROADMAP items with acceptance criteria.
* **Coding assist**

  * Use ChatGPT to propose refactors, write unit tests, and convert TODOs into concrete changes.

> Note: If you prefer, also enable GitHub’s AI pair-programming (Copilot) for inline code completion.

---

## 7) Suggested Measurable Milestones (KPIs)

* **M1 — Repo Online**

  * ✅ Repo initialized with structure, CI green, README/ROADMAP/CHANGELOG created.
* **M2 — Training Loop**

  * ✅ `ppo_learner.py` runs a short training session and saves checkpoints.
  * KPI: completes N episodes without error; average reward improves > X% vs baseline.
* **M3 — Export → Play**

  * ✅ `export_latest_policy.py` generates a loadable artifact.
  * ✅ RLBot agent loads artifact and can play a full match.
  * KPI: passes `test_training_playlists.py`; reaches kickoff cone in ≤ 2.0s drill.
* **M4 — Basic Competency**

  * KPI: win rate ≥ 60% vs a baseline bot over 50 games; saves per game ≥ threshold.
* **M5 — Documentation & Automation**

  * ✅ CHANGELOG reflects features; release `v0.1.0` with notes.

---

## 8) First Implementation Sprint — Concrete Tasks

1. Create GitHub repo; push scaffolding; enable branch protections.
2. Add `requirements.txt`/`pyproject.toml` and `.gitignore`.
3. Rename `example.py` → `training/ppo_learner.py`; update imports.
4. Implement `training/utils/export_policy.py` and `scripts/export_latest_policy.py`.
5. Create `rlbot/agents/ppo_hivemind_agent/` with `agent.py` + `policy_loader.py`.
6. Add initial tests: `test_export_pipeline.py` + `test_training_playlists.py`.
7. Write `ROADMAP.md` and `CHANGELOG.md`; commit with Conventional Commits.
8. (Optional) Add GitHub Action to run tests on push/PR.

---

## 9) Git Commands Cheat Sheet (No secrets committed)

```bash
# Initialize
git init
git remote add origin git@github.com:<you>/rlgym-bot.git

# Create base structure and first commit
git checkout -b feat/scaffold
git add .
git commit -m "feat(scaffold): initial project structure with docs and scripts"
git push -u origin feat/scaffold

# Open a PR on GitHub → merge into main when CI passes

# Tag a release later
git tag -a v0.1.0 -m "First playable PPO export and RLBot agent"
git push origin v0.1.0
```

---

## 10) Definitions of Done (per goal)

* **Repo/Docs**: Main is protected; README, ROADMAP, CHANGELOG exist; CI green.
* **Training**: `ppo_learner.py` runs end-to-end; checkpoints written.
* **Export**: Export script finds latest checkpoint and writes a versioned artifact.
* **Agent**: RLBot agent loads artifact; passes smoke playlist tests.
* **Automation**: One-step scripts in `scripts/` for train/export/run.
* **Metrics**: KPIs logged and tracked in ROADMAP updates.

---

If you want, we can immediately generate `README.md`, `CHANGELOG.md`, and `ROADMAP.md` skeletons plus a `.gitignore` tailored for Python/RLBot, and draft the `export_policy.py` + `agent.py` stubs next.
