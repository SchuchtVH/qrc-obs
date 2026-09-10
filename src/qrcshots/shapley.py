"""SHAP linear e agregação por grupo (Seção 6.1).

Identidade analítica para modelo linear com máscara independente (interventional):

    mean_j |SHAP_j| = |w_j| * mean_i |x_ij - E[x_j]|

ou seja, SHAP aqui é essencialmente |peso| * dispersão do sinal na feature -- mede
variabilidade *de sinal*, não incerteza *de medição* (essa distinção é o cerne da
Seção 6: SHAP não sabe nada sobre `Sigma_g`). A igualdade só é exata com
`shap.maskers.Independent` (perturbação interventional, features tratadas como
independentes do background) -- com um masker que respeita a dependência entre
features ("correlation_dependent"/Partition) a igualdade se quebra, porque a
contribuição marginal de cada feature passa a depender das outras. Usamos
`max_samples=len(background)` para não sub-amostrar o background (sub-amostragem
introduz ruído de Monte Carlo que também quebra a igualdade exata; verificado
empiricamente antes de escrever este módulo).
"""

from __future__ import annotations

import numpy as np
import shap

from qrcshots.measure import GROUPS
from qrcshots.readout import FrozenReadout


def compute_shap_phi(readout: FrozenReadout, X_train_std: np.ndarray) -> np.ndarray:
    """phi_j = mean_over_train(|SHAP_j|), shape (n_features,)."""
    masker = shap.maskers.Independent(X_train_std, max_samples=X_train_std.shape[0])
    explainer = shap.LinearExplainer((readout.weights, readout.intercept), masker)
    shap_values = explainer.shap_values(X_train_std)
    return np.mean(np.abs(shap_values), axis=0)


def analytic_phi_formula(readout: FrozenReadout, X_train_std: np.ndarray) -> np.ndarray:
    """|w_j| * mean|x_j - E[x_j]| -- referência fechada para verificação numérica."""
    dispersion = np.mean(np.abs(X_train_std - X_train_std.mean(axis=0)), axis=0)
    return np.abs(readout.weights) * dispersion


def aggregate_phi_by_group(phi: np.ndarray, n_qubits: int) -> dict[str, float]:
    """I_g = sum_{j in g} phi_j."""
    out = {}
    for gi, g in enumerate(GROUPS):
        out[g] = float(np.sum(phi[gi * n_qubits : (gi + 1) * n_qubits]))
    return out


def shap_group_importance(readout: FrozenReadout, X_train_std: np.ndarray, n_qubits: int) -> dict[str, float]:
    phi = compute_shap_phi(readout, X_train_std)
    return aggregate_phi_by_group(phi, n_qubits)
