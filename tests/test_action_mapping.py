import numpy as np
import pytest

from rlgymbotv2.mysim.action_parsers.simple_discrete_hybrid_action import SimpleHybridDiscreteAction

def test_action_mapping_basic_shape_and_linking():
    tbl = SimpleHybridDiscreteAction().lookup_table
    assert tbl.ndim == 2 and tbl.shape[1] == 8, "Action vectors must be (N, 8)"

    # steer==yaw rows should exist (linked turning)
    linked_rows = np.isclose(tbl[:, 1], tbl[:, 3])
    assert np.count_nonzero(linked_rows) > 0, "Expected some rows with steer == yaw (linked)."

    # Allow some rows to have steer!=yaw (yaw-only torque rows, etc.)
    # But ensure not *all* rows are mismatched.
    assert np.count_nonzero(~linked_rows) < tbl.shape[0], "All rows have steer!=yaw; mapping/order likely wrong."
