"""Núcleo quântico do reservatório: matriz densidade escrita à mão em NumPy.

Convenção de ordenação de qubits (fixa em todo o projeto, não redefinir):
o espaço de Hilbert de n qubits é o produto tensorial qubit_0 ⊗ qubit_1 ⊗ ... ⊗ qubit_{n-1},
isto é, qubit 0 é o fator MAIS significativo do produto de Kronecker. O índice de base
computacional s (0..2^n-1) decompõe em bits como s = bit_0 * 2^(n-1) + ... + bit_{n-1} * 2^0,
com bit_i o valor do qubit i. `np.kron(A, B)` com A atuando no(s) qubit(s) de índice menor já
respeita essa convenção — nunca inverter a ordem dos operandos de kron.

Esquema Fujii–Nakajima com reinicialização por janela (Seção 3 do enunciado):
a cada passo de uma janela, o qubit 0 é resetado (traço parcial + produto tensorial com o novo
estado puro) e em seguida todo o sistema evolui por um unitário fixo U = exp(-i H tau).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
from tqdm import tqdm

I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_SINGLE_PAULI = {"X": X2, "Y": Y2, "Z": Z2, "I": I2}


def kron_at(op2x2: np.ndarray, qubit_idx: int, n_qubits: int) -> np.ndarray:
    """Operador de n qubits com `op2x2` no qubit `qubit_idx` e identidade nos demais."""
    ops = [I2] * n_qubits
    ops[qubit_idx] = op2x2
    return functools.reduce(np.kron, ops)


def kron_uniform(op2x2: np.ndarray, n_qubits: int) -> np.ndarray:
    """Produto tensorial do mesmo operador de 1 qubit em todos os n qubits."""
    return functools.reduce(np.kron, [op2x2] * n_qubits)


def pauli_operator(name: str, qubit_idx: int, n_qubits: int) -> np.ndarray:
    """Operador de Pauli `name` ('X'|'Y'|'Z') atuando no qubit `qubit_idx`, embutido em n_qubits."""
    return kron_at(_SINGLE_PAULI[name], qubit_idx, n_qubits)


def ket0_density(n_qubits: int) -> np.ndarray:
    """rho = |0...0><0...0|."""
    rho = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def theta_from_u(u: float) -> float:
    """Mapeia u em [-1, 1] para theta em [0, pi]: theta = pi*(u+1)/2."""
    return np.pi * (u + 1.0) / 2.0


def injection_state(u: float) -> np.ndarray:
    """rho_psi = |psi(u)><psi(u)|, psi(u) = cos(theta/2)|0> + sin(theta/2)|1>."""
    theta = theta_from_u(u)
    psi = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=np.complex128)
    return np.outer(psi, psi.conj())


def reset_and_inject(rho: np.ndarray, u: float, n_qubits: int, qubit_idx: int = 0) -> np.ndarray:
    """Traço parcial sobre `qubit_idx` seguido de produto tensorial com o novo estado injetado.

    Implementado via reshape/einsum: rho é vista como tensor de 2*n_qubits índices
    (n_qubits índices de linha seguidos de n_qubits índices de coluna, na mesma ordem
    qubit_0..qubit_{n-1} imposta pela convenção de kron). Tracar `qubit_idx` soma sobre o
    par (linha=qubit_idx, coluna=qubit_idx) desse tensor.
    """
    if qubit_idx != 0:
        raise NotImplementedError("Só o reset do qubit 0 é usado neste projeto.")
    dim = 2**n_qubits
    tensor = rho.reshape((2,) * (2 * n_qubits))
    # eixos: (r_0, r_1, ..., r_{n-1}, c_0, c_1, ..., c_{n-1}); traça r_0 com c_0.
    reduced = np.einsum(tensor, [0, *range(1, n_qubits), 0, *range(n_qubits + 1, 2 * n_qubits)])
    reduced = reduced.reshape((dim // 2, dim // 2))
    rho_psi = injection_state(u)
    return np.kron(rho_psi, reduced)


@dataclass(frozen=True)
class ReservoirHamiltonian:
    H: np.ndarray
    J_matrix: np.ndarray  # J_ij simétrica, zeros na diagonal
    h_vector: np.ndarray
    n_qubits: int


def build_hamiltonian(n_qubits: int, J: float, h_low: float, h_high: float, rng: np.random.Generator) -> ReservoirHamiltonian:
    """H = sum_{i<j} J_ij X_i X_j + sum_i h_i Z_i, Ising transverso aleatório."""
    J_matrix = np.zeros((n_qubits, n_qubits))
    h_vector = rng.uniform(h_low, h_high, size=n_qubits)
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            j_ij = rng.uniform(-J, J)
            J_matrix[i, j] = j_ij
            J_matrix[j, i] = j_ij
            XiXj = kron_at(X2, i, n_qubits) @ kron_at(X2, j, n_qubits)
            H += j_ij * XiXj
        H += h_vector[i] * pauli_operator("Z", i, n_qubits)
    H = (H + H.conj().T) / 2.0  # remove ruído numérico de assimetria
    return ReservoirHamiltonian(H=H, J_matrix=J_matrix, h_vector=h_vector, n_qubits=n_qubits)


def build_propagator(H: np.ndarray, tau: float) -> np.ndarray:
    """U = exp(-i H tau), calculado uma única vez por scipy.linalg.expm."""
    return scipy.linalg.expm(-1j * H * tau)


def evolve_window(u_window: np.ndarray, U: np.ndarray, n_qubits: int) -> np.ndarray:
    """Aplica reset+injeção e evolução por U para cada u em u_window, a partir de |0...0>.

    Retorna a matriz densidade final da janela (rho_t usada para medição/previsão de y_t).
    """
    rho = ket0_density(n_qubits)
    for u in u_window:
        rho = reset_and_inject(rho, u, n_qubits)
        rho = U @ rho @ U.conj().T
    return rho


def evolve_window_with_identity(u_window: np.ndarray, n_qubits: int) -> np.ndarray:
    """Igual a `evolve_window` mas sem aplicar evolução (U = I) — usado só em testes
    para isolar o efeito da injeção/reset da convenção de ordenação de qubits."""
    dim = 2**n_qubits
    return evolve_window(u_window, np.eye(dim, dtype=np.complex128), n_qubits)


# --- Cache em disco: H, U (uma única vez por config) e rho_t por janela --------------
#
# H e U são idênticos entre todas as estratégias e orçamentos de uma mesma config
# (Seção 3: "o reservatório é idêntico entre estratégias, sempre"). rho_t depende só
# da janela u[t-L+1:t+1] e é determinística, então é calculada uma única vez para
# todas as janelas de treino/val/teste e cacheada em disco (Seção 3, "Otimização
# importante"). O ruído de shot entra depois, na amostragem sobre esses rho_t.

HAMILTONIAN_CACHE_KEYS = [
    "seeds.master",
    "reservoir.n_qubits",
    "reservoir.J",
    "reservoir.h_low",
    "reservoir.h_high",
    "reservoir.tau",
]


def build_or_load_hamiltonian(cfg: dict, seeds: dict, cache_dir: str | Path) -> tuple[ReservoirHamiltonian, np.ndarray]:
    from qrcshots.config import config_hash, rng_for

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_qubits = cfg["reservoir"]["n_qubits"]
    path = cache_dir / f"hamiltonian_{config_hash(cfg, HAMILTONIAN_CACHE_KEYS)}.npz"

    if path.exists():
        data = np.load(path)
        ham = ReservoirHamiltonian(H=data["H"], J_matrix=data["J_matrix"], h_vector=data["h_vector"], n_qubits=n_qubits)
        return ham, data["U"]

    rng = rng_for(seeds, "reservoir_hamiltonian")
    reservoir_cfg = cfg["reservoir"]
    ham = build_hamiltonian(n_qubits, reservoir_cfg["J"], reservoir_cfg["h_low"], reservoir_cfg["h_high"], rng)
    U = build_propagator(ham.H, reservoir_cfg["tau"])
    np.savez(path, H=ham.H, U=U, J_matrix=ham.J_matrix, h_vector=ham.h_vector)
    return ham, U


def compute_or_load_rho_windows(cfg: dict, u: np.ndarray, t_indices: np.ndarray, U: np.ndarray, cache_dir: str | Path) -> np.ndarray:
    """rho_t exato para cada t em `t_indices` (mesma ordem), cacheado em
    `cache/rho_<hash-da-config>.npy`, shape (N, 2**n_qubits, 2**n_qubits) complexo."""
    from qrcshots.config import reservoir_cache_hash

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"rho_{reservoir_cache_hash(cfg)}.npy"
    n_qubits = cfg["reservoir"]["n_qubits"]
    window_L = cfg["reservoir"]["window_L"]

    if path.exists():
        rho_all = np.load(path)
        if rho_all.shape[0] == len(t_indices):
            return rho_all

    dim = 2**n_qubits
    rho_all = np.empty((len(t_indices), dim, dim), dtype=np.complex128)
    for k, t in enumerate(tqdm(t_indices, desc="rho por janela", leave=False)):
        window = u[t - window_L + 1 : t + 1]
        rho_all[k] = evolve_window(window, U, n_qubits)
    np.save(path, rho_all)
    return rho_all
