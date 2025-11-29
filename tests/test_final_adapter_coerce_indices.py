import numpy as np
from rlgymbotv2.mysim.action_parsers.wrappers.final_rocketsim_adapter import FinalRocketSimAdapter

def test_coerce_indices_mixed_scalar_and_array():
    adapter = FinalRocketSimAdapter(... )  # fill with minimal ctor args or a dummy subclass

    cases = [
        5,
        np.array(6),
        [7, 8],
        [np.array([9]), 10],
        np.array([np.array([11]), 12], dtype=object),
        np.array([[13], [14]]),
    ]

    for case in cases:
        out = adapter._coerce_indices(case)
        assert out.ndim == 1
        assert out.dtype == np.int64
        assert all(isinstance(x, (int, np.integer)) for x in out)
