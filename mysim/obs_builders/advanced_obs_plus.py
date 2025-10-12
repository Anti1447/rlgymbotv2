# advanced_obs_plus.py
# A compact, learning-friendly observation builder for RLGym-style envs.
# Usage:
# from advanced_obs_plus import AdvancedObsPlus
# 
# obs_builder = AdvancedObsPlus(
#     max_allies=2,
#     max_opponents=3,
#     k_nearest_pads_me=4,
#     k_nearest_pads_ball=2,
#     stack_size=1,                 # try 3–4 later for a small performance bump
#     include_prev_action=True,
# )
# env = rlgym.make(obs_builder=obs_builder, ...)  # or your project’s env constructor

from collections import deque
from typing import List, Tuple
import numpy as np

try:
    # RLGym public names
    from rlgym.utils.obs_builders import ObsBuilder
    from rlgym.utils.gamestates import GameState, PlayerData, PhysicsObject
except Exception:
    # Fall back to your local project layout if needed
    from obs_builder import ObsBuilder  # noqa
    from mysim.gamestates import GameState, PlayerData, PhysicsObject  # noqa


def _norm_clip(x, lo=-1.0, hi=1.0):
    return np.clip(x, lo, hi)


class AdvancedObsPlus(ObsBuilder):
    """
    Key ideas:
      - Team invariance: mirror Orange to Blue's frame.
      - Car frame: rotate world -> my car frame (my forward = +X, up = +Z).
      - Stable ordering: others sorted by distance to BALL.
      - Small nearest-pad summary instead of whole pad map.
      - Light game context + explicit action mask bits.
      - Optional tiny frame stack for dynamics (delta-like signal).
    """

    # Physical scales for normalization (rough but consistent)
    POS_MAX = 6000.0         # uu (≈ field half-diagonal ~ sqrt(4096^2+5120^2))
    VEL_MAX = 2300.0         # uu/s (car top speed)
    ANG_VEL_MAX = 5.5        # rad/s (approx car angular velocity cap)
    BALL_VEL_MAX = 6000.0    # uu/s (ceiling for big clears)
    BOOST_MAX = 100.0
    TIME_MAX = 300.0         # normalize game clock to 0..1 for 5 min games

    def __init__(
        self,
        max_allies: int = 2,
        max_opponents: int = 3,
        k_nearest_pads_me: int = 4,
        k_nearest_pads_ball: int = 2,
        stack_size: int = 1,           # set to 3 or 4 to enable tiny temporal stacking
        include_prev_action: bool = True,
    ):
        super().__init__()
        self.max_allies = max_allies
        self.max_opponents = max_opponents
        self.k_nearest_pads_me = k_nearest_pads_me
        self.k_nearest_pads_ball = k_nearest_pads_ball
        self.stack_size = max(1, int(stack_size))
        self.include_prev_action = include_prev_action

        # per-player rolling buffer (by car_id) if stacking is used
        self._stacks = {}
        self._cached = None     # filled in pre_step
        self._obs_size = None   # populated after first build

    # ---------- Helpers ----------

    @staticmethod
    def _team_sign(p: PlayerData) -> int:
        # +1 for blue perspective (default), -1 to mirror orange into blue frame
        return 1 if p.team_num == 0 else -1

    @staticmethod
    def _rot_mat(fwd: np.ndarray, up: np.ndarray) -> np.ndarray:
        # Orthonormal basis: X=fwd, Z=up, Y=Z×X (right-handed X,Y,Z)
        x = fwd / (np.linalg.norm(fwd) + 1e-9)
        z = up  / (np.linalg.norm(up)  + 1e-9)
        y = np.cross(z, x)
        y /= (np.linalg.norm(y) + 1e-9)
        z = np.cross(x, y)
        return np.stack([x, y, z], axis=1)  # columns are basis vectors

    @staticmethod
    def _to_car_frame(R: np.ndarray, v: np.ndarray) -> np.ndarray:
        # World -> car frame (R columns are car basis in world)
        return R.T @ v

    # ---------- ObsBuilder API ----------

    def reset(self, initial_state: GameState):
        self._stacks.clear()
        self._cached = None
        self._obs_size = None

    def pre_step(self, state: GameState):
        """
        Do once-per-tick work we can reuse for every agent.
        Cache:
          - mirrored state flag per player
          - ball/world quantities
          - stable player order (others sorted by dist to ball)
          - boost pad positions & availability
        """
        ball = state.ball
        ball_pos = ball.position.astype(np.float32)
        ball_vel = ball.linear_velocity.astype(np.float32)

        # Player ordering by distance to BALL in their own (mirrored) world
        def pd_with_world_car(p: PlayerData) -> Tuple[PlayerData, PhysicsObject]:
            car = p.inverted_car_data if p.team_num == 1 else p.car_data
            return p, car

        players_world = [pd_with_world_car(p) for p in state.players]
        # For sorting, use mirrored world (so "distance to ball" is team-invariant)
        sorted_by_ball = sorted(
            players_world, key=lambda pc: np.linalg.norm(pc[1].position - ball_pos)
        )

        # Boost pads: collect (pos, is_big, is_active, respawn_time if available)
        pads_meta = []
        if hasattr(state, "boost_pads") and state.boost_pads is not None:
            for bp in state.boost_pads:
                # Expecting fields: position, is_active, is_full_boost (names vary a bit across forks)
                pos = getattr(bp, "position", None)
                if pos is None:
                    continue
                pos = np.asarray(pos, dtype=np.float32)
                is_big = bool(
                    getattr(bp, "is_full_boost", getattr(bp, "is_big", False))
                )
                active = bool(getattr(bp, "is_active", True))
                timer = float(getattr(bp, "timer", 0.0))
                pads_meta.append((pos, is_big, active, timer))

        self._cached = {
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "sorted_players": sorted_by_ball,  # list of (PlayerData, world_car)
            "pads": pads_meta,
            "time_left": float(getattr(state, "seconds_remaining", 0.0)),
            "is_kickoff": bool(getattr(state, "is_kickoff_pause", False)),
            "scores": (
                int(getattr(state, "blue_score", 0)),
                int(getattr(state, "orange_score", 0)),
            ),
        }

    def get_obs_space(self):
        # Filled after the first build. Keeps things flexible across action sizes.
        try:
            import gym
            from gym.spaces import Box
        except Exception:
            return None
        if self._obs_size is None:
            # A conservative placeholder – updated after first build
            return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        return Box(low=-1.0, high=1.0, shape=(self._obs_size,), dtype=np.float32)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        c = self._cached
        assert c is not None, "pre_step must run before build_obs"

        sign = self._team_sign(player)

        # Choose mirrored car data to make Blue the canonical frame
        my_car = player.inverted_car_data if sign == -1 else player.car_data

        # Rotation to my car frame
        R = self._rot_mat(my_car.forward(), my_car.up())
        to_car = lambda v: self._to_car_frame(R, v)

        # --- Ball (relative to me, in my car frame) ---
        rel_ball_pos = to_car(c["ball_pos"] - my_car.position) / self.POS_MAX
        rel_ball_vel = to_car(c["ball_vel"] - my_car.linear_velocity)
        rel_ball_vel = rel_ball_vel / self.BALL_VEL_MAX

        # --- Me (small absolute in my car frame) ---
        my_vel_cf = to_car(my_car.linear_velocity) / self.VEL_MAX
        my_angvel_cf = to_car(my_car.angular_velocity) / self.ANG_VEL_MAX
        my_boost = np.array([player.boost_amount / self.BOOST_MAX], dtype=np.float32)
        has_flip = np.array([float(getattr(player, "has_flip", True))], dtype=np.float32)

        # --- Game context (tiny, high-value) ---
        # Score diff from my perspective (Blue canonical frame)
        blue, orange = c["scores"]
        score_diff = (blue - orange) * (1 if sign == 1 else -1)
        ctx = np.array([
            float(c["is_kickoff"]),
            _norm_clip(c["time_left"] / self.TIME_MAX, 0.0, 1.0),
            np.tanh(score_diff / 3.0),  # squash into ~[-1,1]
        ], dtype=np.float32)

        # --- Action awareness ---
        act_bits = []
        if self.include_prev_action and previous_action is not None:
            # Clip to [-1,1], also add legal-useful flags
            prev = np.asarray(previous_action, dtype=np.float32).ravel()
            prev = _norm_clip(prev, -1.0, 1.0)
            act_bits.append(prev)
        # Minimal action mask bits
        act_bits.append(np.array([
            my_boost[0] > 0.0,    # can_boost
            has_flip[0] > 0.5     # can_flip
        ], dtype=np.float32))
        action_feat = np.concatenate(act_bits, dtype=np.float32)

        # --- Others (sorted by distance to BALL, then split into allies/opps) ---
        # In mirrored world, "team_num==0" are allies for me.
        me_id = player.car_id
        others = [(p, car) for (p, car) in c["sorted_players"] if p.car_id != me_id]
        allies = [(p, car) for (p, car) in others if p.team_num == player.team_num]
        opps   = [(p, car) for (p, car) in others if p.team_num != player.team_num]

        def encode_other(pc_list: List[Tuple[PlayerData, PhysicsObject]], k: int) -> np.ndarray:
            feats = []
            for i in range(k):
                if i < len(pc_list):
                    p, car = pc_list[i]
                    # Relative to ME in my car frame
                    rel_p = to_car(car.position - my_car.position) / self.POS_MAX
                    rel_v = to_car(car.linear_velocity - my_car.linear_velocity) / self.VEL_MAX
                    rel_w = to_car(car.angular_velocity - my_car.angular_velocity) / self.ANG_VEL_MAX
                    b = getattr(p, "boost_amount", 0.0) / self.BOOST_MAX
                    alive = 1.0 - float(getattr(p, "is_demoed", False))
                    feats.append(np.hstack([rel_p, rel_v, rel_w, [b, alive]]))
                else:
                    feats.append(np.zeros(3 + 3 + 3 + 2, dtype=np.float32))
            return np.concatenate(feats, dtype=np.float32)

        ally_feat = encode_other(allies, self.max_allies)
        opp_feat  = encode_other(opps,   self.max_opponents)

        # --- Boost pads summary (nearest to me & to ball) ---
        pad_feat = []
        pads = c["pads"]
        if pads:
            # distances in my *mirrored* world to keep invariance
            my_pos = my_car.position
            bp = np.asarray([p[0] for p in pads], dtype=np.float32)
            d_me = np.linalg.norm(bp - my_pos, axis=1)
            d_ball = np.linalg.norm(bp - c["ball_pos"], axis=1)

            idx_me = np.argsort(d_me)[: self.k_nearest_pads_me]
            idx_ball = np.argsort(d_ball)[: self.k_nearest_pads_ball]

            def enc(indices):
                out = []
                for idx in indices:
                    pos, is_big, active, timer = pads[idx]
                    rel = to_car(pos - my_pos) / self.POS_MAX
                    out.append(np.array([
                        rel[0], rel[1], rel[2],
                        float(is_big),
                        float(active),
                        _norm_clip(timer / 10.0, 0.0, 1.0)  # rough 10s scale
                    ], dtype=np.float32))
                return np.concatenate(out, dtype=np.float32) if out else np.zeros(0, dtype=np.float32)

            pm = enc(idx_me)
            pb = enc(idx_ball)
            # If fewer than requested, right-pad with zeros
            need_me = self.k_nearest_pads_me * 6 - pm.size
            need_ball = self.k_nearest_pads_ball * 6 - pb.size
            if need_me > 0:
                pm = np.pad(pm, (0, need_me))
            if need_ball > 0:
                pb = np.pad(pb, (0, need_ball))
            pad_feat = [pm, pb]
        pad_feat = np.concatenate(pad_feat, dtype=np.float32) if pad_feat else np.zeros(self.k_nearest_pads_me*6 + self.k_nearest_pads_ball*6, dtype=np.float32)

        # --- Goal geometry (ball->their goal, ball->my goal) in my car frame ---
        my_goal_world    = np.array([0.0, -5120.0, 0.0]) if player.team_num == 0 else np.array([0.0, 5120.0, 0.0])
        their_goal_world = -my_goal_world
        g_vecs = np.concatenate([
            to_car(their_goal_world - c["ball_pos"]) / self.POS_MAX,
            to_car(my_goal_world    - c["ball_pos"]) / self.POS_MAX,
        ], dtype=np.float32)

        # --- Assemble one-step vector ---
        step_vec = np.concatenate([
            rel_ball_pos, rel_ball_vel,
            my_vel_cf, my_angvel_cf, my_boost, has_flip,
            ctx,
            action_feat,
            ally_feat, opp_feat,
            pad_feat,
            g_vecs,
        ], dtype=np.float32)

        # --- Optional tiny temporal stack (per-agent) ---
        if self.stack_size > 1:
            dq = self._stacks.get(player.car_id)
            if dq is None:
                dq = deque([np.zeros_like(step_vec) for _ in range(self.stack_size - 1)] + [step_vec], maxlen=self.stack_size)
                self._stacks[player.car_id] = dq
            else:
                dq.append(step_vec)
            vec = np.concatenate(list(dq), dtype=np.float32)
        else:
            vec = step_vec

        if self._obs_size is None:
            self._obs_size = int(vec.size)
        return vec
