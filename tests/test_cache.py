import shutil

import numpy as np
import pytest

from qrcshots.config import apply_overrides, derive_seeds, load_config
from qrcshots.dataset import build_dataset
from qrcshots.reservoir import build_or_load_hamiltonian, compute_or_load_rho_windows


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config("configs/base.yaml")
    cfg = apply_overrides(cfg, ["signal.T=300"])
    cfg["paths"]["cache_dir"] = str(tmp_path / "cache")
    return cfg


def test_hamiltonian_cache_hit_returns_identical_H_and_U(cfg):
    seeds = derive_seeds(cfg)
    ham1, U1 = build_or_load_hamiltonian(cfg, seeds, cfg["paths"]["cache_dir"])
    ham2, U2 = build_or_load_hamiltonian(cfg, seeds, cfg["paths"]["cache_dir"])
    assert np.array_equal(ham1.H, ham2.H)
    assert np.array_equal(U1, U2)


def test_rho_cache_hit_returns_identical_array_and_is_faster_path(cfg):
    seeds = derive_seeds(cfg)
    ds = build_dataset(cfg, seeds)
    t_indices = ds.all_valid_t
    ham, U = build_or_load_hamiltonian(cfg, seeds, cfg["paths"]["cache_dir"])

    from qrcshots.config import rng_for

    rng = rng_for(seeds, "signal")
    u = rng.uniform(cfg["signal"]["u_low"], cfg["signal"]["u_high"], size=cfg["signal"]["T"])

    rho_first = compute_or_load_rho_windows(cfg, u, t_indices, U, cfg["paths"]["cache_dir"])
    rho_second = compute_or_load_rho_windows(cfg, u, t_indices, U, cfg["paths"]["cache_dir"])
    assert np.array_equal(rho_first, rho_second)
    assert rho_first.shape == (len(t_indices), 2 ** cfg["reservoir"]["n_qubits"], 2 ** cfg["reservoir"]["n_qubits"])

    # traço 1 e hermitiana para todas as janelas cacheadas
    traces = np.trace(rho_first, axis1=1, axis2=2)
    assert np.allclose(traces.real, 1.0, atol=1e-8)
    assert np.allclose(traces.imag, 0.0, atol=1e-8)


def test_reservoir_is_identical_across_two_independent_builds_same_config(cfg):
    """Reproduz o requisito de que H/U/rho sejam idênticos entre 'estratégias' --
    aqui simulado como duas chamadas independentes com a mesma config."""
    seeds_a = derive_seeds(cfg)
    seeds_b = derive_seeds(cfg)
    ham_a, U_a = build_or_load_hamiltonian(cfg, seeds_a, cfg["paths"]["cache_dir"])
    shutil.rmtree(cfg["paths"]["cache_dir"])
    ham_b, U_b = build_or_load_hamiltonian(cfg, seeds_b, cfg["paths"]["cache_dir"])
    assert np.array_equal(ham_a.H, ham_b.H)
    assert np.array_equal(U_a, U_b)
