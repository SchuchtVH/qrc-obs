"""Grupos de medição comutantes (X, Y, Z) e amostragem com shots finitos (Seção 4).

Um shot de um grupo devolve as 4 bitstrings simultâneas (correlacionadas) dos 4 qubits
naquela base. Shots são alocados a *grupos*, não a features individuais.

Pareamento de sementes entre estratégias (Seção 8): `rng.multinomial` amostrado
separadamente para cada `n_shots` NÃO compartilha prefixo entre alocações diferentes
do mesmo grupo (ex. variance aloca 40 shots ao grupo X, uniform aloca 32 — dois
multinomiais independentes não têm shots em comum). Para que a comparação entre
estratégias seja de fato pareada (mesmos números aleatórios "seguram" o que pode),
amostramos uma sequência de índices de bitstring do orçamento MÁXIMO uma única vez
por (seed, janela, grupo) com `draw_shot_indices`, e cada estratégia usa apenas o
prefixo de tamanho `n_g` dessa sequência via `counts_from_indices`. Isso é o análogo
de "common random numbers": a amostra de uma estratégia com menos shots é literalmente
um prefixo da amostra de uma estratégia com mais shots, então a variação entre
estratégias vem só da alocação, não de sorte de amostragem independente.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from qrcshots.reservoir import I2, kron_uniform, pauli_operator

_H1 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_SDAG1 = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
_Y_ROT1 = _H1 @ _SDAG1  # aplica S† depois H (ordem de matriz: H @ Sdag)

_SINGLE_QUBIT_ROTATION = {"X": _H1, "Y": _Y_ROT1, "Z": I2}

GROUPS = ("X", "Y", "Z")


@functools.lru_cache(maxsize=None)
def basis_rotation(group: str, n_qubits: int) -> np.ndarray:
    """R_g: mesma rotação de 1 qubit aplicada a todos os qubits (produto tensorial).

    Não depende de rho, só de (group, n_qubits) -- cacheada porque é chamada uma vez
    por amostragem e o experimento completo faz milhões de amostragens.
    """
    return kron_uniform(_SINGLE_QUBIT_ROTATION[group], n_qubits)


def _bit_signs(n_qubits: int) -> np.ndarray:
    """signs[s, i] = +1 se o bit do qubit i no índice de base s é 0, senão -1.

    Qubit 0 é o bit mais significativo (convenção de reservoir.py)."""
    dim = 2**n_qubits
    signs = np.empty((dim, n_qubits))
    for s in range(dim):
        for i in range(n_qubits):
            bit = (s >> (n_qubits - 1 - i)) & 1
            signs[s, i] = 1.0 - 2.0 * bit
    return signs


_BIT_SIGNS_CACHE: dict[int, np.ndarray] = {}


def bit_signs(n_qubits: int) -> np.ndarray:
    if n_qubits not in _BIT_SIGNS_CACHE:
        _BIT_SIGNS_CACHE[n_qubits] = _bit_signs(n_qubits)
    return _BIT_SIGNS_CACHE[n_qubits]


def diag_probs_after_rotation(rho: np.ndarray, group: str, n_qubits: int) -> np.ndarray:
    R = basis_rotation(group, n_qubits)
    rho_rot = R @ rho @ R.conj().T
    p = np.real(np.diag(rho_rot))
    p = np.clip(p, 0.0, None)
    total = p.sum()
    return p / total


def draw_shot_indices(rho: np.ndarray, group: str, n_shots_max: int, rng: np.random.Generator, n_qubits: int = 4) -> np.ndarray:
    """Sequência de `n_shots_max` índices de bitstring (0..2^n-1) i.i.d. de p.

    Um prefixo `indices[:n_g]` desta sequência é, em distribuição, idêntico a uma
    amostra direta de tamanho `n_g` — é isso que viabiliza o pareamento entre
    estratégias com orçamentos diferentes por grupo.
    """
    p = diag_probs_after_rotation(rho, group, n_qubits)
    dim = 2**n_qubits
    return rng.choice(dim, size=n_shots_max, p=p)


def counts_from_indices(indices: np.ndarray, n_shots: int, n_qubits: int = 4) -> np.ndarray:
    dim = 2**n_qubits
    return np.bincount(indices[:n_shots], minlength=dim)


def estimates_from_counts(counts: np.ndarray, n_qubits: int = 4) -> np.ndarray:
    """4 estimativas <P_i>_hat (médias de ±1) a partir dos counts de 2^n bitstrings."""
    n_shots = counts.sum()
    signs = bit_signs(n_qubits)
    return (counts @ signs) / n_shots


@dataclass(frozen=True)
class GroupSample:
    means: np.ndarray  # shape (n_qubits,), <P_i>_hat
    counts: np.ndarray  # shape (2**n_qubits,), contagens brutas
    n_shots: int


def sample_group(rho: np.ndarray, group: str, n_shots: int, rng: np.random.Generator, n_qubits: int = 4) -> GroupSample:
    """Amostragem direta (sem pareamento) de `n_shots` shots do grupo `group`."""
    p = diag_probs_after_rotation(rho, group, n_qubits)
    dim = 2**n_qubits
    counts = rng.multinomial(n_shots, p)
    means = estimates_from_counts(counts, n_qubits)
    return GroupSample(means=means, counts=counts, n_shots=n_shots)


def exact_expectations(rho: np.ndarray, group: str, n_qubits: int = 4) -> np.ndarray:
    """<P_i> = tr(rho P_i) exato, para os n_qubits observáveis do grupo."""
    return np.array([np.real(np.trace(rho @ pauli_operator(group, i, n_qubits))) for i in range(n_qubits)])


def exact_group_covariance(rho: np.ndarray, group: str, n_qubits: int = 4) -> np.ndarray:
    """Sigma_g[i,j] = tr(rho P_i P_j) - tr(rho P_i) tr(rho P_j) (covariância de shot único).

    Para i != j, P_i P_j (mesmo tipo de Pauli, qubits diferentes) é ele mesmo hermitiano
    com autovalores +-1, então tr(rho P_i P_j) é um valor esperado direto. Para i == j,
    P_i^2 = I, logo Sigma_g[i,i] = 1 - <P_i>^2.
    """
    means = exact_expectations(rho, group, n_qubits)
    sigma = np.empty((n_qubits, n_qubits))
    ops = [pauli_operator(group, i, n_qubits) for i in range(n_qubits)]
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i == j:
                sigma[i, i] = 1.0 - means[i] ** 2
            else:
                pij = np.real(np.trace(rho @ ops[i] @ ops[j]))
                sigma[i, j] = pij - means[i] * means[j]
    return sigma
