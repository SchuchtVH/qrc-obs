"""Baseline de literatura: leitura ótima por kernel de Hilbert-Schmidt (Seção 8).

Ver `docs/baseline_kernel.md` para a leitura do paper (arXiv:2602.14677) e onde esta
implementação difere dele. Resumo da instanciação usada aqui:

    K(rho_a, rho_b) = tr(rho_a rho_b)            (kernel de Hilbert-Schmidt)
    alpha = (K + lambda I)^{-1} (y_train - mean(y_train))   (ridge de kernel)
    O*    = sum_a alpha_a rho_a                  (observável de leitura ótimo)
    y_hat(rho_x) = tr(O* rho_x) + mean(y_train)

`O*` é decomposto na base de Pauli (`c_P = tr(O* P) / 2^n_qubits`) e projetado
ortogonalmente no suporte medível: os 12 Paulis de peso 1 (`X_i, Y_i, Z_i`). Por
Parseval (Paulis são ortogonais sob o produto interno de Hilbert-Schmidt, com
`tr(P^2) = 2^n_qubits`), a norma total de `O*` é `tr(O*^2)` diretamente -- não é
preciso enumerar os 4^n_qubits termos da base completa para calcular a fração de
norma fora do suporte medível.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qrcshots.measure import GROUPS
from qrcshots.readout import FrozenReadout, StandardScalerStats, nmse
from qrcshots.reservoir import pauli_operator


def kernel_matrix(rho_a_batch: np.ndarray, rho_b_batch: np.ndarray) -> np.ndarray:
    """K[i,j] = tr(rho_a_batch[i] @ rho_b_batch[j]), real (rho é hermitiana)."""
    return np.real(np.einsum("ikl,jlk->ij", rho_a_batch, rho_b_batch))


@dataclass(frozen=True)
class KernelRidgeFit:
    alpha: np.ndarray
    y_mean: float
    lam: float


def fit_kernel_ridge(K_train: np.ndarray, y_train: np.ndarray, lam: float) -> KernelRidgeFit:
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    alpha = np.linalg.solve(K_train + lam * np.eye(len(y_train)), y_centered)
    return KernelRidgeFit(alpha=alpha, y_mean=y_mean, lam=lam)


def predict_kernel_ridge(fit: KernelRidgeFit, K_cross: np.ndarray) -> np.ndarray:
    """K_cross[i,j] = K(rho_query_i, rho_train_j)."""
    return K_cross @ fit.alpha + fit.y_mean


def select_kernel_ridge(rho_train: np.ndarray, y_train: np.ndarray, rho_val: np.ndarray, y_val: np.ndarray, lambda_grid: np.ndarray) -> tuple[KernelRidgeFit, list[dict]]:
    K_train = kernel_matrix(rho_train, rho_train)
    K_val = kernel_matrix(rho_val, rho_train)
    results = []
    best = None
    for lam in lambda_grid:
        fit = fit_kernel_ridge(K_train, y_train, lam)
        val_nmse = nmse(y_val, predict_kernel_ridge(fit, K_val))
        results.append({"lam": float(lam), "val_nmse": val_nmse})
        if best is None or val_nmse < best[0]:
            best = (val_nmse, fit)
    return best[1], results


def build_o_star(rho_train: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """O* = sum_a alpha_a rho_a."""
    return np.einsum("a,aij->ij", alpha, rho_train)


def weight1_pauli_coeffs(o_star: np.ndarray, n_qubits: int) -> dict[str, np.ndarray]:
    """c_{g,i} = tr(O* P_{g,i}) / 2^n_qubits, para os 12 Paulis de peso 1."""
    dim = 2**n_qubits
    return {g: np.array([np.real(np.trace(o_star @ pauli_operator(g, i, n_qubits))) / dim for i in range(n_qubits)]) for g in GROUPS}


def norm_fraction_outside_support(o_star: np.ndarray, coeffs: dict[str, np.ndarray], n_qubits: int) -> dict[str, float]:
    dim = 2**n_qubits
    total_norm_sq = float(np.real(np.trace(o_star @ o_star)))
    weight1_norm_sq = float(dim * sum(np.sum(c**2) for c in coeffs.values()))
    return {
        "total_norm_sq": total_norm_sq,
        "weight1_norm_sq": weight1_norm_sq,
        "fraction_outside_support": 1.0 - weight1_norm_sq / total_norm_sq,
    }


def restricted_readout_from_coeffs(coeffs: dict[str, np.ndarray], y_mean: float, n_qubits: int) -> FrozenReadout:
    """Readout linear equivalente à projeção de O* no suporte medível -- reusa toda
    a maquinaria de `allocate.py`/`experiment.py` (scaler identidade: sem
    padronização, pesos já na escala das features raw)."""
    weights = np.concatenate([coeffs[g] for g in GROUPS])
    scaler = StandardScalerStats(mean=np.zeros_like(weights), scale=np.ones_like(weights))
    return FrozenReadout(weights=weights, intercept=y_mean, scaler=scaler, alpha=0.0, n_qubits=n_qubits)
