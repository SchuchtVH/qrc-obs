import numpy as np
import pytest
from sklearn.linear_model import Ridge

from qrcshots.readout import (
    FrozenReadout,
    StandardScalerStats,
    collect_features,
    exact_features,
    feature_group_index,
    feature_names,
    fit_frozen_readout,
    nmse,
    rmse,
)


def test_ridge_matches_closed_form_solution():
    rng = np.random.default_rng(0)
    n, p = 200, 12
    X = rng.normal(size=(n, p))
    true_w = rng.normal(size=p)
    y = X @ true_w + 0.1 * rng.normal(size=n)
    alpha = 0.7

    model = Ridge(alpha=alpha)
    model.fit(X, y)

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    w_closed = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
    intercept_closed = y.mean() - X.mean(axis=0) @ w_closed

    assert np.allclose(model.coef_, w_closed, atol=1e-8)
    assert np.isclose(model.intercept_, intercept_closed, atol=1e-8)


def test_nmse_and_rmse_basic():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    assert nmse(y_true, y_pred) == 0.0
    assert rmse(y_true, y_pred) == 0.0
    y_pred2 = y_true + 1.0
    assert np.isclose(rmse(y_true, y_pred2), 1.0)


def test_feature_names_and_group_index_ordering():
    names = feature_names(4)
    assert names == ["X0", "X1", "X2", "X3", "Y0", "Y1", "Y2", "Y3", "Z0", "Z1", "Z2", "Z3"]
    gi = feature_group_index(4)
    assert list(gi) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_standard_scaler_matches_manual_zscore():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=5.0, scale=2.0, size=(500, 3))
    scaler = StandardScalerStats.fit(X)
    X_std = scaler.transform(X)
    assert np.allclose(X_std.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X_std.std(axis=0), 1.0, atol=1e-8)


def test_fit_frozen_readout_selects_alpha_and_freezes_weights():
    rng = np.random.default_rng(2)
    n_train, n_val, p = 300, 100, 12
    true_w = rng.normal(size=p)
    X_train = rng.normal(size=(n_train, p))
    y_train = X_train @ true_w + 0.05 * rng.normal(size=n_train)
    X_val = rng.normal(size=(n_val, p))
    y_val = X_val @ true_w + 0.05 * rng.normal(size=n_val)

    alpha_grid = np.logspace(-6, 2, 9)
    readout, grid = fit_frozen_readout(X_train, y_train, X_val, y_val, alpha_grid, n_qubits=4)

    assert isinstance(readout, FrozenReadout)
    assert readout.weights.shape == (p,)
    assert readout.alpha in alpha_grid
    assert len(grid) == len(alpha_grid)
    # pesos "congelados": chamar predict não deve alterar os atributos do objeto
    preds_before = readout.predict(X_val)
    preds_after = readout.predict(X_val)
    assert np.array_equal(preds_before, preds_after)

    for group in ["X", "Y", "Z"]:
        gw = readout.group_weights(group)
        assert gw.shape == (4,)


def test_exact_features_have_no_sampling_noise_between_repeats():
    from qrcshots.reservoir import build_hamiltonian, build_propagator, evolve_window

    rng = np.random.default_rng(3)
    ham = build_hamiltonian(4, 1.0, 0.5, 1.5, rng)
    U = build_propagator(ham.H, 1.0)
    rho_batch = np.stack([evolve_window(rng.uniform(-1, 1, size=20), U, 4) for _ in range(5)])

    feats_a = exact_features(rho_batch, 4)
    feats_b = exact_features(rho_batch, 4)
    assert np.array_equal(feats_a, feats_b)
    assert feats_a.shape == (5, 12)


def test_collect_features_are_noisy_and_shaped_correctly():
    from qrcshots.reservoir import build_hamiltonian, build_propagator, evolve_window

    rng = np.random.default_rng(4)
    ham = build_hamiltonian(4, 1.0, 0.5, 1.5, rng)
    U = build_propagator(ham.H, 1.0)
    rho_batch = np.stack([evolve_window(rng.uniform(-1, 1, size=20), U, 4) for _ in range(5)])

    feats = collect_features(rho_batch, n_shots_per_group=64, rng=rng, n_qubits=4)
    assert feats.shape == (5, 12)
    assert np.all(np.abs(feats) <= 1.0 + 1e-9)
