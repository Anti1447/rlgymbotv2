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
from rlgymbotv2.mysim.obs_builders import ObsBuilder  # noqa
from rlgymbotv2.mysim.gamestates import GameState, PlayerData, PhysicsObject  # noqa
from typing import Optional
from rlgymbotv2.mysim.common_values import BOOST_LOCATIONS



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
        k_nearest_pads_me: int = 5,
        k_nearest_pads_ball: int = 3,
        stack_size: int = 3,           # set to 3 or 4 to enable tiny temporal stacking
        include_prev_action: bool = True,
        action_lookup: Optional[np.ndarray] = None,  # for LUT index -> 8-dim action conversion
    ):
        super().__init__()
        self.max_allies = max_allies
        self.max_opponents = max_opponents
        self.k_nearest_pads_me = k_nearest_pads_me
        self.k_nearest_pads_ball = k_nearest_pads_ball
        self.stack_size = max(1, int(stack_size))
        self.include_prev_action = include_prev_action
        self.action_lookup = action_lookup

        # per-player rolling buffer (by car_id) if stacking is used
        self._stacks = {}
        self._cached = None     # filled in pre_step
        self._obs_size = self._calc_obs_len()

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

        # --- Boost pads: RocketSim-style (state.boost_pads is an array of 0/1 flags) ---
        pads_meta = []
        pads_flags = getattr(state, "boost_pads", None)

        if pads_flags is not None:
            pads_flags = np.asarray(pads_flags, dtype=np.float32).ravel()

            # BOOST_LOCATIONS is a list/array of 34 (x, y, z) pad positions on standard map
            n = min(len(BOOST_LOCATIONS), pads_flags.shape[0])

            for i in range(n):
                pos = np.asarray(BOOST_LOCATIONS[i], dtype=np.float32)

                # infer "big" based on height (RL convention: big pads are the tall ones)
                is_big = bool(pos[2] > 72.0)

                # RocketSim exposes active flag as 0.0 / 1.0
                active = bool(pads_flags[i] > 0.5)

                # we don't get a cooldown timer from GameState, so just stub 0.0
                timer = 0.0

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

    
    def _calc_obs_len(self) -> int:
        # per-step (no stacking) feature counts
        ball_rel = 3 + 3                        
        # rel pos + rel vel
        # me = 3 + 3 + 1 + 1   # vel + angvel + boost + has_flip   (CURRENT)
        # Add: on_ground (+1) and up_z (+1)  -> +2 total
        me = 3 + 3 + 1 + 1 + 1 + 1
        # If instead you want full up vector (+3) replace the last +1 with +3.
        ctx = 3                                 # kickoff, time_left, score_diff
        act = (8 if self.include_prev_action else 0) + 2   # prev action (8) + [can_boost, can_flip]
        other_per = 3 + 3 + 3 + 2               # rel_p, rel_v, rel_w, [boost, alive] = 11
        allies = self.max_allies * other_per
        opps = self.max_opponents * other_per
        pads = (self.k_nearest_pads_me * 6) + (self.k_nearest_pads_ball * 6)  # each: [rel(3), is_big, active, timer] = 6
        # goals: ball->their, ball->my, car->their, car->my (all in car frame)
        goals = 3 * 4
        base = ball_rel + me + ctx + act + allies + opps + pads + goals
        return base * self.stack_size


    def get_obs_space(self):
        try:
            import gym
            from gym.spaces import Box
        except Exception:
            return None
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
        
        # Robust has_flip
        _has_flip = getattr(player, "has_flip", None)
        if _has_flip is None and hasattr(player, "car_data"):
            _has_flip = getattr(player.car_data, "has_flip", None)
        if _has_flip is None:
            _has_flip = True  # conservative default
        has_flip = np.array([float(_has_flip)], dtype=np.float32)

        # Robust on_ground
        _on_ground = getattr(my_car, "on_ground", None)
        if _on_ground is None:
            _on_ground = getattr(player, "on_ground", False)
        on_ground = np.array([float(bool(_on_ground))], dtype=np.float32)

        # Car up() in world frame
        up_world = np.asarray(my_car.up(), dtype=np.float32)
        up_z = np.array([float(up_world[2])], dtype=np.float32)   # preferred minimal feature
        # If you want the full vector, normalize (should already be unit) and use up_world itself.

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
            prev = np.asarray(previous_action)

            # If RepeatAction is used, take the last tick
            if prev.ndim == 2:            # (T,1) or (T,8)
                prev = prev[-1]

            prev = prev.ravel()            # (1,) index OR (8,) vector

            if prev.size == 1 and self.action_lookup is not None:
                # scalar LUT index -> 8-dim action
                idx = int(prev[0])
                idx = max(0, min(idx, len(self.action_lookup) - 1))
                prev = self.action_lookup[idx]
            elif prev.size != 8:
                # Unexpected shape: fall back to zeros to keep obs length deterministic
                prev = np.zeros(8, dtype=np.float32)

            prev = _norm_clip(prev, -1.0, 1.0).astype(np.float32)
            act_bits.append(prev)

        # Minimal action mask bits
        act_bits.append(np.array([
            1.0 if my_boost[0] > 0.0 else 0.0,   # can_boost
            1.0 if has_flip[0] > 0.5 else 0.0    # can_flip
        ], dtype=np.float32))

        action_feat = np.concatenate(act_bits, dtype=np.float32)

        # --- Others (sorted by distance to BALL, then split into allies/opps) ---
        # In mirrored world, "team_num==0" are allies for me.
        me_id = player.car_id
        others = [(p, car) for (p, car) in c["sorted_players"] if p.car_id != me_id]
        allies = [(p, car) for (p, car) in others if p.team_num == player.team_num]
        opps   = [(p, car) for (p, car) in others if p.team_num != player.team_num]

        def encode_other(pc_list: List[Tuple[PlayerData, PhysicsObject]], k: int) -> np.ndarray:
            # If caller requested zero slots (e.g., 1v0 setup), return empty vector.
            if k <= 0:
                return np.zeros(0, dtype=np.float32)

            feats = []
            FEAT_LEN = 3 + 3 + 3 + 2  # rel_p, rel_v, rel_w, [boost, alive] = 11
            for i in range(k):
                if i < len(pc_list):
                    p, car = pc_list[i]
                    rel_p = to_car(car.position - my_car.position) / self.POS_MAX
                    rel_v = to_car(car.linear_velocity - my_car.linear_velocity) / self.VEL_MAX
                    rel_w = to_car(car.angular_velocity - my_car.angular_velocity) / self.ANG_VEL_MAX
                    b = getattr(p, "boost_amount", 0.0) / self.BOOST_MAX
                    alive = 1.0 - float(getattr(p, "is_demoed", False))
                    feats.append(np.hstack([rel_p, rel_v, rel_w, [b, alive]]).astype(np.float32))
                else:
                    feats.append(np.zeros(FEAT_LEN, dtype=np.float32))
            # feats is guaranteed non-empty here because k>0
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

        # --- Goal geometry (ball & car → goals) in my car frame ---
        # RL standard map: back walls at y = ±5120. We treat the goal as a
        # center point on that wall; width/height are captured implicitly via
        # ball/car X,Z already present in other features.
        my_goal_world    = np.array([0.0, -5120.0, 0.0]) if player.team_num == 0 else np.array([0.0,  5120.0, 0.0])
        their_goal_world = -my_goal_world

        ball_pos_world = c["ball_pos"]
        car_pos_world  = my_car.position.astype(np.float32)

        # Ball → goals (in my car frame)
        ball_to_their = to_car(their_goal_world - ball_pos_world) / self.POS_MAX
        ball_to_my    = to_car(my_goal_world   - ball_pos_world) / self.POS_MAX

        # Car → goals (in my car frame)
        car_to_their  = to_car(their_goal_world - car_pos_world) / self.POS_MAX
        car_to_my     = to_car(my_goal_world   - car_pos_world) / self.POS_MAX

        g_vecs = np.concatenate([
            ball_to_their, ball_to_my,
            car_to_their,  car_to_my,
        ], dtype=np.float32)


        # Enforce team invariance: map both teams into the canonical "blue frame"
        g_vecs = g_raw * sign


        # --- Assemble one-step vector ---
        step_vec = np.concatenate([
            rel_ball_pos, rel_ball_vel,
            my_vel_cf, my_angvel_cf, my_boost, has_flip,
            on_ground, up_z,              # <— add here (or replace up_z with up_world)
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
        vec = np.clip(vec, -1.0, 1.0).astype(np.float32) #That keeps the data consistent with the advertised observation space, which some algos assume.
        return vec
