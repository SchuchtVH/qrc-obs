"""Coleta de features, padronização e readout ridge congelado (Seção 5).

Ordem de features fixa (não redefinir): para cada grupo em `measure.GROUPS` = ('X','Y','Z'),
os `n_qubits` observáveis na ordem de qubit 0..n-1. Com n_qubits=4: X0,X1,X2,X3,Y0,Y1,Y2,Y3,
Z0,Z1,Z2,Z3. Todo módulo a jusante (shapley, allocate) assume essa ordem para mapear
feature -> grupo.

Decisão de design (não especificada explicitamente no enunciado, documentada aqui): o
readout final é ajustado só no split de treino; a validação é usada apenas para escolher
`alpha` por busca em grade, nunca para ajustar os pesos. Isso mantém os papéis de
treino/validação separados e é a leitura mais direta da Seção 5 ("Modelo... com alpha
escolhido por busca em grade na validação" + "Congele os pesos após o treino").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from qrcshots.measure import GROUPS, exact_expectations, sample_group


def feature_names(n_qubits: int = 4) -> list[str]:
    return [f"{g}{i}" for g in GROUPS for i in range(n_qubits)]


def feature_group_index(n_qubits: int = 4) -> np.ndarray:
    """Array de tamanho 3*n_qubits com o índice do grupo (0=X,1=Y,2=Z) de cada feature."""
    return np.array([gi for gi in range(len(GROUPS)) for _ in range(n_qubits)])


def collect_features(rho_batch: np.ndarray, n_shots_per_group: int, rng: np.random.Generator, n_qubits: int = 4) -> np.ndarray:
    """Features ruidosas: `n_shots_per_group` shots por grupo, por janela."""
    allocation = {g: n_shots_per_group for g in GROUPS}
    return collect_features_with_allocation(rho_batch, allocation, rng, n_qubits)


def collect_features_with_allocation(rho_batch: np.ndarray, allocation: dict[str, int], rng: np.random.Generator, n_qubits: int = 4) -> np.ndarray:
    """Features ruidosas com um número de shots POR GRUPO diferente (`allocation`),
    por janela -- usado pela fase 2 (Seção 10), onde a coleta de treino/val passa a
    seguir a alocação aprendida, não mais uniforme."""
    n = rho_batch.shape[0]
    X = np.empty((n, len(GROUPS) * n_qubits))
    for k in range(n):
        cols = []
        for g in GROUPS:
            cols.append(sample_group(rho_batch[k], g, allocation[g], rng, n_qubits).means)
        X[k] = np.concatenate(cols)
    return X


def exact_features(rho_batch: np.ndarray, n_qubits: int = 4) -> np.ndarray:
    """Features exatas (shots -> infinito): tr(rho P_i) sem ruído de amostragem."""
    n = rho_batch.shape[0]
    X = np.empty((n, len(GROUPS) * n_qubits))
    for k in range(n):
        cols = [exact_expectations(rho_batch[k], g, n_qubits) for g in GROUPS]
        X[k] = np.concatenate(cols)
    return X


@dataclass(frozen=True)
class StandardScalerStats:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale

    @staticmethod
    def fit(X: np.ndarray) -> "StandardScalerStats":
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return StandardScalerStats(mean=mean, scale=scale)


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def select_alpha(X_train_std: np.ndarray, y_train: np.ndarray, X_val_std: np.ndarray, y_val: np.ndarray, alpha_grid: np.ndarray) -> tuple[float, list[dict]]:
    results = []
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        model.fit(X_train_std, y_train)
        val_nmse = nmse(y_val, model.predict(X_val_std))
        results.append({"alpha": float(alpha), "val_nmse": val_nmse})
    best = min(results, key=lambda r: r["val_nmse"])
    return best["alpha"], results


@dataclass(frozen=True)
class FrozenReadout:
    weights: np.ndarray  # espaço padronizado, shape (n_features,)
    intercept: float
    scaler: StandardScalerStats
    alpha: float
    n_qubits: int

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        X_std = self.scaler.transform(X_raw)
        return X_std @ self.weights + self.intercept

    def group_weights(self, group: str) -> np.ndarray:
        """Subvetor de `weights` (na escala padronizada) restrito ao grupo, na
        ordem de qubit 0..n-1 -- é o `w_g` usado em allocate.py."""
        gi = GROUPS.index(group)
        start = gi * self.n_qubits
        return self.weights[start : start + self.n_qubits]


def fit_frozen_readout(X_train_raw: np.ndarray, y_train: np.ndarray, X_val_raw: np.ndarray, y_val: np.ndarray, alpha_grid: np.ndarray, n_qubits: int = 4) -> tuple[FrozenReadout, list[dict]]:
    scaler = StandardScalerStats.fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)
    X_val_std = scaler.transform(X_val_raw)

    best_alpha, grid_results = select_alpha(X_train_std, y_train, X_val_std, y_val, alpha_grid)

    model = Ridge(alpha=best_alpha)
    model.fit(X_train_std, y_train)
    readout = FrozenReadout(weights=model.coef_.copy(), intercept=float(model.intercept_), scaler=scaler, alpha=best_alpha, n_qubits=n_qubits)
    return readout, grid_results


def alpha_grid_from_cfg(cfg: dict) -> np.ndarray:
    r = cfg["ridge"]
    return np.logspace(r["alpha_grid_log_min"], r["alpha_grid_log_max"], r["alpha_grid_n"])
