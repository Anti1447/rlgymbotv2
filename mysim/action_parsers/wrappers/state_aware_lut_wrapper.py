from typing import Optional, Dict, Any, List
import numpy as np
from ._compat import safe_get_action_space
from typing import Optional, Dict, Any, List


class StateAwareLUTWrapper:
    """
    Ensures actions chosen from the LUT are legal given the car's state.
    Falls back to a safe index (idle) when a chosen action is invalid.
    """
    def __init__(self, inner_lut_parser, fallback_index: int = 0):
        self.inner = inner_lut_parser
        self.fallback_index = int(fallback_index)

        table = getattr(inner_lut_parser, "lookup_table",
                 getattr(inner_lut_parser, "_lookup_table", None))
        if table is None:
            raise ValueError("StateAwareLUTWrapper requires inner_lut_parser with a .lookup_table ndarray")

        # 🔧 store as attribute so other methods can use it
        self._table = np.asarray(table, dtype=np.float32)

        # Masks used for legality checks
        self._is_flip   = (self.table[:, 5] > 0.5)  # jump == 1 → flip-like
        self._is_aerial = (np.abs(self.table[:, 2:5]).sum(axis=1) > 0) & (self.table[:, 5] < 0.5)

        # Precompute recovery candidates (Fix B)
        self._recovery = self._build_recovery_fallbacks(self.table)

    # (optional) provide a read-only alias so older code using `.table` keeps working
    @property
    def table(self):
        return self._table

    def _row_matches(self, row: np.ndarray, target: np.ndarray, tol: float = 1e-6) -> bool:
        """Exact-ish row match with tolerance for float LUTs."""
        return np.allclose(row, target, atol=tol)

    def _find_first(self, table: np.ndarray, targets: list) -> Optional[int]:
        """Return the first index in `table` that matches any of `targets` (list of 8-vectors)."""
        for k, r in enumerate(table):
            for t in targets:
                if self._row_matches(r, t):
                    return k
        return None

    def _build_recovery_fallbacks(self, table: np.ndarray) -> dict:
        """
        Precompute useful recovery indices from the LUT.
        All rows are in RLBot control order:
          [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        # Core ground pops
        idx_jump_hb  = self._find_first(table, [
            np.array([0,0,0,0,0,1,0,1], dtype=np.float32),   # jump + handbrake
        ])
        idx_jump     = self._find_first(table, [
            np.array([0,0,0,0,0,1,0,0], dtype=np.float32),   # jump only
        ])

        # Jump + small roll (snap-up)
        idx_jump_roll_left  = self._find_first(table, [
            np.array([0,0,0,0,-1,1,0,0], dtype=np.float32),
            np.array([0,0,0,0,-0.5,1,0,0], dtype=np.float32),
        ])
        idx_jump_roll_right = self._find_first(table, [
            np.array([0,0,0,0, 1,1,0,0], dtype=np.float32),
            np.array([0,0,0,0, 0.5,1,0,0], dtype=np.float32),
        ])

        # Optional: torque-only roll (for in-air righting)
        idx_torque_roll_left  = self._find_first(table, [
            np.array([0,0,0,0,-1,0,0,0], dtype=np.float32),
        ])
        idx_torque_roll_right = self._find_first(table, [
            np.array([0,0,0,0, 1,0,0,0], dtype=np.float32),
        ])

        # Prioritized candidate list for ground turtle (strongest first)
        ground_candidates = [x for x in [
            idx_jump_hb, idx_jump_roll_left, idx_jump_roll_right, idx_jump
        ] if x is not None]

        # Candidates for in-air upside-down
        air_candidates = [x for x in [
            idx_torque_roll_left, idx_torque_roll_right, idx_jump
        ] if x is not None]

        return {
            "jump_hb": idx_jump_hb,
            "jump": idx_jump,
            "jump_roll_left": idx_jump_roll_left,
            "jump_roll_right": idx_jump_roll_right,
            "torque_roll_left": idx_torque_roll_left,
            "torque_roll_right": idx_torque_roll_right,
            "ground_candidates": ground_candidates,
            "air_candidates": air_candidates,
        }

    def _choose_recovery_index(self, p, up_z: float, on_ground: bool) -> Optional[int]:
        """
        Choose the best cached recovery index for this player state.
        Prefers a direction based on the car's right-vector z component if available.
        """
        if on_ground:
            # Prefer a strong ground pop list
            cand = list(self._recovery["ground_candidates"])
            # Nudge left/right ordering based on right().z to bias roll direction
            try:
                rz = float(p.car_data.right()[2])
                if rz > 0 and self._recovery["jump_roll_right"] in cand:
                    # move right option earlier
                    cand.remove(self._recovery["jump_roll_right"])
                    cand.insert(0, self._recovery["jump_roll_right"])
                elif rz < 0 and self._recovery["jump_roll_left"] in cand:
                    cand.remove(self._recovery["jump_roll_left"])
                    cand.insert(0, self._recovery["jump_roll_left"])
            except Exception:
                pass
            return cand[0] if cand else None
        else:
            cand = list(self._recovery["air_candidates"])
            # Similar directional nudge in air
            try:
                rz = float(p.car_data.right()[2])
                if rz > 0 and self._recovery["torque_roll_right"] in cand:
                    cand.remove(self._recovery["torque_roll_right"])
                    cand.insert(0, self._recovery["torque_roll_right"])
                elif rz < 0 and self._recovery["torque_roll_left"] in cand:
                    cand.remove(self._recovery["torque_roll_left"])
                    cand.insert(0, self._recovery["torque_roll_left"])
            except Exception:
                pass
            return cand[0] if cand else None

    def get_action_space(self, agent=None):
        return safe_get_action_space(self.inner, agent)

    def reset(self, *args, **kwargs):
        if hasattr(self.inner, "reset"):
            self.inner.reset(*args, **kwargs)

    def parse_actions(
        self,
        actions,
        state,
        shared_info: Optional[Dict[str, Any]] = None,
        *_,
        **__
    ) -> List[np.ndarray]:
        """
        Handle scalar, list, dict, or batched numpy actions robustly.
        Maps invalid actions to fallback_index before passing to inner parser.
        """
        # --- Normalize input to numpy array form ---
        if isinstance(actions, dict):
            # Dictionary input (multi-agent case)
            per_agent_idxs: List[np.ndarray] = []
            for p in state.players:
                if p.car_id in actions:
                    per_agent_idxs.append(np.atleast_1d(np.asarray(actions[p.car_id], dtype=int)))
        else:
            a = np.asarray(actions)
            # Handle scalar or empty case
            if a.ndim == 0:
                a = np.expand_dims(a, axis=0)
            elif a.size == 0:
                a = np.array([self.fallback_index], dtype=int)

            # Convert to per-agent list
            if a.ndim == 1:
                per_agent_idxs = [np.atleast_1d(int(x)) for x in a]
            elif a.ndim == 2:
                per_agent_idxs = [np.atleast_1d(a[i].astype(int)) for i in range(a.shape[0])]
            else:
                raise ValueError(f"Unexpected action array shape: {a.shape}")

        table = self._table
        remapped: List[np.ndarray] = []
        
        for i, idxs in enumerate(per_agent_idxs):
            if i >= len(state.players):
                break
            
            p = state.players[i]
            # respect blue/orange inversion once; don’t overwrite later
            car = getattr(p, "inverted_car_data", None) if p.team_num == 1 else getattr(p, "car_data", None)
        
            # probe state once
            up_z = float(car.up()[2]) if car is not None else 1.0
            pos  = getattr(car, "position", None)
            z    = float(pos[2]) if pos is not None else 999.0
            on_ground = bool(getattr(p, "on_ground", 0))
            has_flip  = bool(getattr(p, "has_flip", True))
            boost_amt = float(getattr(p, "boost_amount", 0.0))
        
            fixed_indices = []
            # clip to table range; ensure Python ints
            for raw_idx in np.atleast_1d(np.clip(idxs, 0, len(table) - 1)).astype(int):
                idx = int(raw_idx)
                illegal = False  # ✅ initialize for each candidate
        
                # --- gating rules ---
                if self._is_flip[idx] and not has_flip:
                    illegal = True
        
                # block “aerial” torque at low Z only when upright-ish; allow when side/turtle
                if self._is_aerial[idx] and (z < 100.0) and (up_z > 0.4):
                    illegal = True
        
                # boost gating
                if table[idx, 6] > 0.5 and boost_amt <= 0.0:
                    illegal = True
        
                # --- Fix B: recovery substitution instead of idle ---
                if illegal and ( (on_ground and up_z < 0.30) or ((not on_ground) and up_z < 0.0) ):
                    sub = self._choose_recovery_index(p, up_z, on_ground=on_ground)
                    if sub is not None:
                        idx = int(sub)
                        illegal = False
                        # optional: re-check boost only (if your recovery rows never use boost, skip)
                        if table[idx, 6] > 0.5 and boost_amt <= 0.0:
                            illegal = True
        
                fixed_indices.append(self.fallback_index if illegal else idx)
        
            remapped.append(np.asarray(fixed_indices, dtype=int))
        
        # pass down
        return self.inner.parse_actions(remapped, state, shared_info)

