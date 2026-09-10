import numpy as np
import pytest

from qrcshots.measure import (
    counts_from_indices,
    draw_shot_indices,
    estimates_from_counts,
    exact_expectations,
    exact_group_covariance,
    sample_group,
)
from qrcshots.reservoir import (
    build_hamiltonian,
    build_propagator,
    evolve_window,
    kron_uniform,
    ket0_density,
)

N_QUBITS = 4


def _random_physical_rho(seed):
    rng = np.random.default_rng(seed)
    ham = build_hamiltonian(N_QUBITS, J=1.0, h_low=0.5, h_high=1.5, rng=rng)
    U = build_propagator(ham.H, tau=1.0)
    u_window = rng.uniform(-1, 1, size=20)
    return evolve_window(u_window, U, N_QUBITS)


@pytest.mark.parametrize("group", ["X", "Y", "Z"])
def test_sample_group_unbiased(group):
    rho = _random_physical_rho(seed=10)
    rng = np.random.default_rng(999)
    exact = exact_expectations(rho, group, N_QUBITS)
    sample = sample_group(rho, group, n_shots=200_000, rng=rng, n_qubits=N_QUBITS)
    # CLT: erro padrão ~ 1/sqrt(n) ~ 0.0022; 5-sigma de folga.
    assert np.allclose(sample.means, exact, atol=0.02)


@pytest.mark.parametrize("group", ["X", "Y", "Z"])
def test_variance_of_mean_matches_clt_formula(group):
    rho = _random_physical_rho(seed=11)
    exact = exact_expectations(rho, group, N_QUBITS)
    n_shots = 50
    n_reps = 4000
    rng = np.random.default_rng(42)
    all_means = np.array([sample_group(rho, group, n_shots, rng, N_QUBITS).means for _ in range(n_reps)])
    empirical_var = all_means.var(axis=0, ddof=1)
    expected_var = (1.0 - exact**2) / n_shots
    # tolerância generosa: variância de variância amostral com n_reps=4000
    assert np.allclose(empirical_var, expected_var, atol=3e-3)


@pytest.mark.parametrize("group", ["X", "Y", "Z"])
def test_empirical_covariance_matches_exact_sigma(group):
    rho = _random_physical_rho(seed=12)
    sigma_exact = exact_group_covariance(rho, group, N_QUBITS)
    n_shots = 1
    n_reps = 200_000
    rng = np.random.default_rng(7)
    samples = np.array([sample_group(rho, group, n_shots, rng, N_QUBITS).means for _ in range(n_reps)])
    sigma_empirical = np.cov(samples, rowvar=False, ddof=1)
    assert np.allclose(sigma_empirical, sigma_exact, atol=0.02)


def test_z_basis_no_rotation_on_zero_state():
    rho = ket0_density(N_QUBITS)
    rng = np.random.default_rng(0)
    sample = sample_group(rho, "Z", n_shots=1000, rng=rng, n_qubits=N_QUBITS)
    assert sample.counts[0] == 1000
    assert np.allclose(sample.means, 1.0)


def test_x_basis_rotation_maps_plus_state_to_zero_bitstring():
    """|++++> medido no grupo X deve concentrar toda a massa no bitstring 0000,
    validando a direção da rotação de base X (Hadamard)."""
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    rho_plus = np.outer(plus, plus.conj())
    rho = kron_uniform(rho_plus, N_QUBITS)
    rng = np.random.default_rng(1)
    sample = sample_group(rho, "X", n_shots=1000, rng=rng, n_qubits=N_QUBITS)
    assert sample.counts[0] == 1000
    assert np.allclose(sample.means, 1.0)


def test_y_basis_rotation_maps_plus_i_state_to_zero_bitstring():
    """|+i+i+i+i> medido no grupo Y deve concentrar toda a massa no bitstring 0000,
    validando a direção da rotação de base Y (S dagger seguido de Hadamard)."""
    plus_i = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    rho_plus_i = np.outer(plus_i, plus_i.conj())
    rho = kron_uniform(rho_plus_i, N_QUBITS)
    rng = np.random.default_rng(2)
    sample = sample_group(rho, "Y", n_shots=1000, rng=rng, n_qubits=N_QUBITS)
    assert sample.counts[0] == 1000
    assert np.allclose(sample.means, 1.0)


def test_paired_sampling_prefix_matches_direct_distributional_source():
    """counts_from_indices em um prefixo dos índices sorteados por draw_shot_indices
    dá exatamente o mesmo resultado que usar só os primeiros n_g elementos -- ou seja,
    o pareamento entre orçamentos é um prefixo determinístico da mesma sequência."""
    rho = _random_physical_rho(seed=13)
    rng = np.random.default_rng(5)
    indices = draw_shot_indices(rho, "X", n_shots_max=100, rng=rng, n_qubits=N_QUBITS)
    counts_40 = counts_from_indices(indices, 40, N_QUBITS)
    counts_40_direct = np.bincount(indices[:40], minlength=2**N_QUBITS)
    assert np.array_equal(counts_40, counts_40_direct)
    assert counts_40.sum() == 40
    means_40 = estimates_from_counts(counts_40, N_QUBITS)
    assert means_40.shape == (N_QUBITS,)
