# Roadmap

## Milestones

### M1 — Repo Online
- Repo initialized with CI (tests, lint/types, pre-commit).
- Docs: README, CHANGELOG, ROADMAP.

### M2 — Training Loop
- `training/ppo_learner.py` runs and saves checkpoints.
- KPI: average reward improves > X% vs baseline in N episodes.

### M3 — Export → Play
- `scripts/export_latest_policy.py` writes artifact.
- RLBot agent loads artifact and completes full match.
- KPI: passes RLBotTraining drills; kickoff cone ≤ 2.0s.

### M4 — Basic Competency
- KPI: win rate ≥ 60% vs baseline bot across 50 games.

### M5 — Release & Docs
- Tag v0.1.0, publish release notes.
- CHANGELOG updated; Quickstart validated.

## Backlog / Ideas
- Add Shadow Defense & Kickoff drills as acceptance tests.
- Policy evaluation harness with Elo vs multiple bots.
- Experiment tracking (TensorBoard or Weights & Biases).
- Auto-upload artifacts to GitHub Releases.