import numpy as np
import pytest

from qrcshots.reservoir import (
    build_hamiltonian,
    build_propagator,
    evolve_window,
    evolve_window_with_identity,
    kron_uniform,
    ket0_density,
    pauli_operator,
    theta_from_u,
)

N_QUBITS = 4


def expval(rho, P):
    return np.real(np.trace(rho @ P))


def test_injection_isolated_matches_cos_theta():
    """Injeta um único u em |0000> com evolução identidade; <Z_0> deve ser cos(theta)
    exatamente. Isola a injeção/reset da convenção de ordenação de qubits (Hamiltoniano
    e evolução não entram aqui)."""
    rng = np.random.default_rng(0)
    for u in rng.uniform(-1, 1, size=25):
        rho = evolve_window_with_identity(np.array([u]), N_QUBITS)
        z0 = expval(rho, pauli_operator("Z", 0, N_QUBITS))
        theta = theta_from_u(u)
        assert np.isclose(z0, np.cos(theta), atol=1e-10)
        # qubits não injetados permanecem em |0>: <Z_i> = 1 para i>0
        for i in range(1, N_QUBITS):
            assert np.isclose(expval(rho, pauli_operator("Z", i, N_QUBITS)), 1.0, atol=1e-10)


def test_zero_state_reference_expectations():
    rho = ket0_density(N_QUBITS)
    for i in range(N_QUBITS):
        assert np.isclose(expval(rho, pauli_operator("Z", i, N_QUBITS)), 1.0, atol=1e-10)
        assert np.isclose(expval(rho, pauli_operator("X", i, N_QUBITS)), 0.0, atol=1e-10)
        assert np.isclose(expval(rho, pauli_operator("Y", i, N_QUBITS)), 0.0, atol=1e-10)


def test_plus_state_reference_expectations():
    """|++++> obtido injetando u=1 (theta=pi, psi=|1>)? Não -- construímos |+> diretamente
    via theta = pi/2 (u=0 -> theta=pi/2 -> psi=(|0>+|1>)/sqrt2) usando reset em todos os
    qubits manualmente, para checar <X_i>=1 nesse estado (detecta erro de rotação de base
    se usado junto do sample_group em test_measure.py)."""
    H_single = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    rho_plus_1q = np.outer(plus, plus.conj())
    rho = kron_uniform(rho_plus_1q, N_QUBITS)
    # sanity: H|0><0|H = |+><+|
    zero_1q = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    assert np.allclose(H_single @ zero_1q @ H_single.conj().T, rho_plus_1q)
    for i in range(N_QUBITS):
        assert np.isclose(expval(rho, pauli_operator("X", i, N_QUBITS)), 1.0, atol=1e-10)
        assert np.isclose(expval(rho, pauli_operator("Z", i, N_QUBITS)), 0.0, atol=1e-10)


def test_propagator_is_unitary():
    rng = np.random.default_rng(1)
    ham = build_hamiltonian(N_QUBITS, J=1.0, h_low=0.5, h_high=1.5, rng=rng)
    U = build_propagator(ham.H, tau=0.7)
    dim = 2**N_QUBITS
    assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-9)
    assert np.allclose(U.conj().T @ U, np.eye(dim), atol=1e-9)


def test_hamiltonian_is_hermitian():
    rng = np.random.default_rng(2)
    ham = build_hamiltonian(N_QUBITS, J=1.0, h_low=0.5, h_high=1.5, rng=rng)
    assert np.allclose(ham.H, ham.H.conj().T)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rho_stays_physical_along_window(seed):
    """rho permanece hermitiana, PSD (autovalores >= -eps) e de traço 1 em todo passo
    de uma janela completa, não só no estado final."""
    rng = np.random.default_rng(seed)
    ham = build_hamiltonian(N_QUBITS, J=1.0, h_low=0.5, h_high=1.5, rng=rng)
    U = build_propagator(ham.H, tau=1.0)
    u_window = rng.uniform(-1, 1, size=20)

    from qrcshots.reservoir import ket0_density, reset_and_inject

    rho = ket0_density(N_QUBITS)
    for u in u_window:
        rho = reset_and_inject(rho, u, N_QUBITS)
        _assert_physical(rho)
        rho = U @ rho @ U.conj().T
        _assert_physical(rho)


def _assert_physical(rho, atol=1e-9):
    assert np.allclose(rho, rho.conj().T, atol=atol), "rho não é hermitiana"
    assert np.isclose(np.trace(rho).real, 1.0, atol=atol), "traço de rho != 1"
    eigvals = np.linalg.eigvalsh(rho)
    assert eigvals.min() > -atol, f"rho não é PSD, min eigval={eigvals.min()}"


def test_evolve_window_end_to_end_smoke():
    rng = np.random.default_rng(3)
    ham = build_hamiltonian(N_QUBITS, J=1.0, h_low=0.5, h_high=1.5, rng=rng)
    U = build_propagator(ham.H, tau=1.0)
    u_window = rng.uniform(-1, 1, size=20)
    rho = evolve_window(u_window, U, N_QUBITS)
    _assert_physical(rho)
