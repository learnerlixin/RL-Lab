import numpy as np
from algorithms.tabular.monte_carlo import first_visit_mc_prediction


def test_first_visit_only_counts_first_state_visit():
    values, counts = first_visit_mc_prediction([[(0, 1.0), (0, 2.0)]], 1, gamma=1.0)
    assert np.allclose(values, [3.0])
    assert counts.tolist() == [1]
