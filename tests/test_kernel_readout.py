import numpy as np

from qrcshots.kernel_readout import (
    build_o_star,
    fit_kernel_ridge,
    kernel_matrix,
    norm_fraction_outside_support,
    predict_kernel_ridge,
    restricted_readout_from_coeffs,
    weight1_pauli_coeffs,
)
from qrcshots.measure import GROUPS
from qrcshots.reservoir import build_hamiltonian, build_propagator, evolve_window, kron_at, pauli_operator, I2

N_QUBITS = 4


def _random_rho_batch(n, seed):
    rng = np.random.default_rng(seed)
    ham = build_hamiltonian(N_QUBITS, 1.0, 0.5, 1.5, rng)
    U = build_propagator(ham.H, 1.0)
    return np.stack([evolve_window(rng.uniform(-1, 1, size=20), U, N_QUBITS) for _ in range(n)])


def test_kernel_matrix_diagonal_is_purity():
    rho_batch = _random_rho_batch(5, seed=0)
    K = kernel_matrix(rho_batch, rho_batch)
    purities = np.array([np.real(np.trace(r @ r)) for r in rho_batch])
    assert np.allclose(np.diag(K), purities)
    assert np.allclose(K, K.T)  # simétrica: tr(AB)=tr(BA)


def test_kernel_ridge_interpolates_with_tiny_lambda():
    rho_batch = _random_rho_batch(15, seed=1)
    rng = np.random.default_rng(2)
    y = rng.normal(size=15)
    K = kernel_matrix(rho_batch, rho_batch)
    fit = fit_kernel_ridge(K, y, lam=1e-10)
    pred = predict_kernel_ridge(fit, K)
    assert np.allclose(pred, y, atol=1e-4)


def test_weight1_coeffs_zero_norm_outside_for_pure_weight1_operator():
    """O* construído só de Paulis de peso 1 deve ter fração fora do suporte ~0."""
    rng = np.random.default_rng(3)
    true_coeffs = {g: rng.normal(size=N_QUBITS) for g in GROUPS}
    o_star = sum(true_coeffs[g][i] * pauli_operator(g, i, N_QUBITS) for g in GROUPS for i in range(N_QUBITS))

    recovered = weight1_pauli_coeffs(o_star, N_QUBITS)
    for g in GROUPS:
        assert np.allclose(recovered[g], true_coeffs[g], atol=1e-10)

    frac = norm_fraction_outside_support(o_star, recovered, N_QUBITS)
    assert np.isclose(frac["fraction_outside_support"], 0.0, atol=1e-8)


def test_weight1_coeffs_nonzero_norm_outside_with_weight2_term():
    """Adicionar um termo de peso 2 (Z0 Z1) deve produzir uma fração > 0 fora do
    suporte, com valor exatamente calculável (Paulis são ortogonais)."""
    rng = np.random.default_rng(4)
    true_coeffs = {g: rng.normal(size=N_QUBITS) for g in GROUPS}
    o_weight1 = sum(true_coeffs[g][i] * pauli_operator(g, i, N_QUBITS) for g in GROUPS for i in range(N_QUBITS))

    z0z1 = kron_at(np.array([[1, 0], [0, -1]], dtype=np.complex128), 0, N_QUBITS) @ kron_at(np.array([[1, 0], [0, -1]], dtype=np.complex128), 1, N_QUBITS)
    c_weight2 = 2.5
    o_star = o_weight1 + c_weight2 * z0z1

    recovered = weight1_pauli_coeffs(o_star, N_QUBITS)
    for g in GROUPS:
        assert np.allclose(recovered[g], true_coeffs[g], atol=1e-10)

    result = norm_fraction_outside_support(o_star, recovered, N_QUBITS)
    dim = 2**N_QUBITS
    weight1_norm_sq = dim * sum(np.sum(c**2) for c in true_coeffs.values())
    weight2_norm_sq = dim * c_weight2**2
    expected_total = weight1_norm_sq + weight2_norm_sq
    expected_fraction = weight2_norm_sq / expected_total
    assert np.isclose(result["total_norm_sq"], expected_total, atol=1e-8)
    assert np.isclose(result["fraction_outside_support"], expected_fraction, atol=1e-8)


def test_restricted_readout_matches_direct_trace_formula():
    rho_batch = _random_rho_batch(3, seed=5)
    rng = np.random.default_rng(6)
    coeffs = {g: rng.normal(size=N_QUBITS) for g in GROUPS}
    o_star_restricted = sum(coeffs[g][i] * pauli_operator(g, i, N_QUBITS) for g in GROUPS for i in range(N_QUBITS))
    y_mean = 0.42

    readout = restricted_readout_from_coeffs(coeffs, y_mean, N_QUBITS)
    from qrcshots.readout import exact_features

    X_exact = exact_features(rho_batch, N_QUBITS)
    pred = readout.predict(X_exact)

    direct = np.array([np.real(np.trace(o_star_restricted @ rho)) + y_mean for rho in rho_batch])
    assert np.allclose(pred, direct, atol=1e-10)


def test_build_o_star_matches_definition():
    rho_batch = _random_rho_batch(6, seed=7)
    rng = np.random.default_rng(8)
    alpha = rng.normal(size=6)
    o_star = build_o_star(rho_batch, alpha)
    manual = sum(alpha[a] * rho_batch[a] for a in range(6))
    assert np.allclose(o_star, manual)
