import numpy as np
import pytest

from qrcshots.config import apply_overrides, derive_seeds, load_config
from qrcshots.dataset import TestSetAccessedBeforeFreeze, build_dataset
from qrcshots.tasks import narma5_target, primary_target, product3_target


@pytest.fixture
def cfg():
    cfg = load_config("configs/base.yaml")
    return apply_overrides(cfg, ["signal.T=600"])


def test_test_split_raises_before_freeze(cfg):
    seeds = derive_seeds(cfg)
    ds = build_dataset(cfg, seeds)
    with pytest.raises(TestSetAccessedBeforeFreeze):
        _ = ds.test


def test_test_split_accessible_after_freeze(cfg):
    seeds = derive_seeds(cfg)
    ds = build_dataset(cfg, seeds)
    ds.freeze()
    split = ds.test
    assert split.windows.shape[0] == split.y.shape[0]


def test_splits_are_temporally_ordered_and_disjoint(cfg):
    seeds = derive_seeds(cfg)
    ds = build_dataset(cfg, seeds)
    ds.freeze()
    train_idx = ds.train.t_indices
    val_idx = ds.val.t_indices
    test_idx = ds.test.t_indices
    assert train_idx.max() < val_idx.min()
    assert val_idx.max() < test_idx.min()
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(set(all_idx.tolist())) == len(all_idx)


def test_windows_have_correct_length_and_target_alignment(cfg):
    cfg = apply_overrides(cfg, ["task.name=primary"])
    seeds = derive_seeds(cfg)
    ds = build_dataset(cfg, seeds)
    window_L = cfg["reservoir"]["window_L"]
    split = ds.train
    assert split.windows.shape[1] == window_L
    # a última entrada da janela é u_t; para a tarefa primária, y_t = u_{t-2}*u_{t-5}
    t0 = split.t_indices[0]
    u_t = split.windows[0, -1]
    u_t_minus_2 = split.windows[0, -3]
    u_t_minus_5 = split.windows[0, -6]
    assert np.isclose(split.y[0], u_t_minus_2 * u_t_minus_5)


def test_reproducibility_same_config_same_numbers(cfg):
    seeds1 = derive_seeds(cfg)
    seeds2 = derive_seeds(cfg)
    ds1 = build_dataset(cfg, seeds1)
    ds2 = build_dataset(cfg, seeds2)
    ds1.freeze()
    ds2.freeze()
    assert np.array_equal(ds1.test.windows, ds2.test.windows)
    assert np.array_equal(ds1.test.y, ds2.test.y)


def test_primary_target_formula():
    u = np.arange(10, dtype=float)
    y = primary_target(u)
    assert np.isnan(y[:5]).all()
    for t in range(5, 10):
        assert y[t] == u[t - 2] * u[t - 5]


def test_product3_target_formula():
    u = np.arange(10, dtype=float)
    y = product3_target(u, lags=(1, 3, 7))
    assert np.isnan(y[:7]).all()
    for t in range(7, 10):
        assert y[t] == u[t - 1] * u[t - 3] * u[t - 7]


def test_narma5_target_matches_recurrence_and_is_bounded():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 0.2, size=200)
    y = narma5_target(u)
    assert np.isnan(y[:5]).all()
    assert np.isfinite(y[5:]).all()
    # recorrência manual para os primeiros passos após o warmup
    y_manual = np.zeros_like(u)
    for t in range(5, 12):
        y_manual[t] = 0.3 * y_manual[t - 1] + 0.05 * y_manual[t - 1] * np.sum(y_manual[t - 5 : t]) + 1.5 * u[t - 5] * u[t - 1] + 0.1
        assert np.isclose(y[t], y_manual[t])
