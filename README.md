# RLGym Bot

A reproducible pipeline to train a Rocket League agent (RLGym), export the latest PPO policy, and play it in RLBotGUI.

## Quickstart

```bash
# 1) Create & activate venv (example)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run tests
pytest -q

# 4) Train (example, adjust)
python -m training.ppo_learner --config configs/training.yaml

# 5) Export latest policy
python scripts/export_latest_policy.py

# 6) Run RLBot agent (see RLBotGUI docs)
# Agent loads exported artifact from artifacts/<version>/ or latest export path
