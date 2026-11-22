import numpy as np
import pytest

from algorithms.offline_rl.dataset import OfflineTransitionDataset


def make_dataset():
    return OfflineTransitionDataset(
        observations=np.zeros((5, 3)), actions=np.zeros((5, 1)),
        rewards=np.arange(5), next_observations=np.ones((5, 3)),
        terminals=[0, 0, 1, 0, 0], timeouts=[0, 1, 0, 0, 0],
    )


def test_dataset_round_trip_npz_and_preserves_timeout(tmp_path):
    dataset = make_dataset()
    path = tmp_path / "nested" / "demo.npz"
    dataset.save_npz(path)
    loaded = OfflineTransitionDataset.from_npz(path)
    assert len(loaded) == 5
    assert loaded[1]["timeouts"].item() == 1.0
    assert loaded[2]["terminals"].item() == 1.0
    assert loaded[3]["observations"].shape == (3,)


def test_dataset_rejects_inconsistent_lengths_and_nonfinite_data():
    with pytest.raises(ValueError):
        OfflineTransitionDataset(np.zeros((2, 3)), np.zeros((1, 1)), [0, 0],
                                np.zeros((2, 3)), [0, 0])
    bad = np.zeros((2, 3)); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        OfflineTransitionDataset(bad, np.zeros((2, 1)), [0, 0],
                                np.zeros((2, 3)), [0, 0])


def test_from_npz_requires_documented_fields(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, observations=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        OfflineTransitionDataset.from_npz(path)
