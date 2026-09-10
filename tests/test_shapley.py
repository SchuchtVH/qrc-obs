import numpy as np

from qrcshots.readout import FrozenReadout, StandardScalerStats
from qrcshots.shapley import aggregate_phi_by_group, analytic_phi_formula, compute_shap_phi


def _make_readout(p, seed):
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=p)
    scaler = StandardScalerStats(mean=np.zeros(p), scale=np.ones(p))
    return FrozenReadout(weights=weights, intercept=0.3, scaler=scaler, alpha=1.0, n_qubits=p // 3)


def test_shap_matches_analytic_formula_tight_tolerance():
    rng = np.random.default_rng(1)
    n, p = 400, 12
    # features correlacionadas entre si (como as reais, vindas do mesmo reservatório)
    base = rng.normal(size=(n, 1))
    X = base + 0.4 * rng.normal(size=(n, p))
    readout = _make_readout(p, seed=2)

    phi_shap = compute_shap_phi(readout, X)
    phi_formula = analytic_phi_formula(readout, X)

    assert np.allclose(phi_shap, phi_formula, atol=1e-9)


def test_aggregate_by_group_sums_correctly():
    phi = np.arange(12, dtype=float)
    agg = aggregate_phi_by_group(phi, n_qubits=4)
    assert agg["X"] == sum(range(0, 4))
    assert agg["Y"] == sum(range(4, 8))
    assert agg["Z"] == sum(range(8, 12))
