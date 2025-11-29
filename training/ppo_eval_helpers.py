# rlgymbotv2/training/ppo_eval_helpers.py
# ---------------------------------------
# Helper utilities for evaluation / skill-gap tests.
#
# Keep this file free of heavy training logic to avoid circular imports.
# Anything that:
#   - builds short-lived eval envs
#   - runs snapshot-vs-snapshot matches
#   - aggregates win/goal/reward stats
# can live here.

# Revisions:
# 2025-11-27 – Rev 1.0.0 – Initial extraction of skill-gap eval helpers from ppo_learner.py. – NJH
# 2025-11-28 – Rev 1.1.0 – Added Elo JSON output + WandB logging hooks for eval runs. – NJH


from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from rlgymbotv2.selfplay.pool import load_frozen_opponent


EnvBuilder = Callable[[], Any]


def run_skill_gap_eval(
    blue_path: Path,
    orange_path: Path,
    device: str,
    env_builder: EnvBuilder,
    n_episodes: int = 50,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a short 1v1 evaluation between two frozen policies.

    Args:
        blue_path:   Path to BLUE policy .pt file.
        orange_path: Path to ORANGE policy .pt file.
        device:      "cpu" / "cuda" / "cuda:0" etc.
        env_builder: Callable that returns a freshly-built env instance.
                     Example from ppo_learner:
                         lambda: build_rocketsim_env(spawn_opponents=True)
        n_episodes:  Number of evaluation episodes to play.
        max_steps:   Optional hard cap on steps per episode
                     (None = let env episode termination decide).

    Returns:
        Dict with win/loss/draw counts, goals, and mean rewards.
    """
    blue_path = Path(blue_path)
    orange_path = Path(orange_path)

    blue_policy = load_frozen_opponent(blue_path, device=device)
    orange_policy = load_frozen_opponent(orange_path, device=device)

    env = env_builder()

    total_blue_goals = 0
    total_orange_goals = 0
    total_blue_rew = 0.0
    total_orange_rew = 0.0

    blue_wins = 0
    orange_wins = 0
    draws = 0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False

        blue_ep_rew = 0.0
        orange_ep_rew = 0.0
        blue_goals = 0
        orange_goals = 0

        steps = 0

        while not done:
            # Expect obs structure: (blue_obs, orange_obs) or list of length 2
            if isinstance(obs, (list, tuple)) and len(obs) == 2:
                blue_obs, orange_obs = obs
            else:
                # If you ever change env layout to multi-agent dicts, update here.
                blue_obs = obs[0]
                orange_obs = obs[1]

            a_blue = blue_policy.act(blue_obs)
            a_orange = orange_policy.act(orange_obs)

            obs, rewards, done, info = env.step([a_blue, a_orange])

            # rlgym-ppo multi-agent reward: [blue, orange]
            if isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) == 2:
                r_blue, r_orange = float(rewards[0]), float(rewards[1])
            else:
                r_blue = float(rewards)
                r_orange = 0.0

            blue_ep_rew += r_blue
            orange_ep_rew += r_orange

            # Pull current score from GameState if present
            state = info.get("state")
            if state is not None:
                blue_goals = int(state.blue_score)
                orange_goals = int(state.orange_score)

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        total_blue_goals += blue_goals
        total_orange_goals += orange_goals
        total_blue_rew += blue_ep_rew
        total_orange_rew += orange_ep_rew

        if blue_goals > orange_goals:
            blue_wins += 1
        elif orange_goals > blue_goals:
            orange_wins += 1
        else:
            draws += 1

        print(
            f"Episode {ep:03d}: "
            f"blue_goals={blue_goals} orange_goals={orange_goals} "
            f"blue_rew={blue_ep_rew:.3f} orange_rew={orange_ep_rew:.3f}"
        )

    stats = {
        "episodes": n_episodes,
        "blue_wins": blue_wins,
        "orange_wins": orange_wins,
        "draws": draws,
        "blue_goals": total_blue_goals,
        "orange_goals": total_orange_goals,
        "blue_reward_mean": total_blue_rew / n_episodes,
        "orange_reward_mean": total_orange_rew / n_episodes,
    }

    print("\n=== Skill Gap Eval (inline) ===")
    print(f"Episodes:        {n_episodes}")
    print(f"Blue wins:       {blue_wins}")
    print(f"Orange wins:     {orange_wins}")
    print(f"Draws:           {draws}")
    print(
        f"Avg goals/game:  "
        f"blue={total_blue_goals / n_episodes:.3f}, "
        f"orange={total_orange_goals / n_episodes:.3f}"
    )
    print(
        f"Avg reward/game: blue={stats['blue_reward_mean']:.3f}, "
        f"orange={stats['orange_reward_mean']:.3f}"
    )

    return stats
