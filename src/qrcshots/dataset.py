"""Construção de janelas, split temporal e guarda de test-set (Seção 2).

Regra dura do enunciado: toda decisão (hiperparâmetros, alocação, escolha de
estimador) usa só treino e validação. `Dataset.test` levanta `RuntimeError` se
acessado antes de `Dataset.freeze()` ter sido chamado -- é uma guarda de verdade,
não um comentário.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qrcshots.tasks import Task, build_task


@dataclass(frozen=True)
class Split:
    t_indices: np.ndarray  # índices absolutos em u/y (mesmos entre estratégias)
    windows: np.ndarray  # shape (n, window_L), u[t-L+1 : t+1] por linha
    y: np.ndarray  # shape (n,)


class TestSetAccessedBeforeFreeze(RuntimeError):
    pass


class Dataset:
    def __init__(self, u: np.ndarray, y: np.ndarray, window_L: int, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, task_name: str):
        self._u = u
        self._y = y
        self.window_L = window_L
        self.task_name = task_name
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def _make_split(self, idx: np.ndarray) -> Split:
        windows = np.stack([self._u[t - self.window_L + 1 : t + 1] for t in idx])
        return Split(t_indices=idx, windows=windows, y=self._y[idx])

    @property
    def train(self) -> Split:
        return self._make_split(self._train_idx)

    @property
    def val(self) -> Split:
        return self._make_split(self._val_idx)

    @property
    def test(self) -> Split:
        if not self._frozen:
            raise TestSetAccessedBeforeFreeze(
                "Split de teste acessado antes de Dataset.freeze(). Toda decisão de "
                "hiperparâmetro, alocação ou estimador deve usar só treino/validação."
            )
        return self._make_split(self._test_idx)

    @property
    def all_valid_t(self) -> np.ndarray:
        """Índices t válidos das três partições, união -- usado para cachear rho_t
        uma única vez para todas as janelas, independente do split (Seção 3)."""
        return np.concatenate([self._train_idx, self._val_idx, self._test_idx])

    @property
    def u(self) -> np.ndarray:
        """Sinal de entrada bruto. Exposto sem guarda de freeze: rho_t é física
        determinística, não uma decisão, então precomputar rho para janelas de teste
        (Seção 3) não viola a regra de isolamento do split de teste."""
        return self._u


def select_rho(rho_all: np.ndarray, cached_t_indices: np.ndarray, query_t_indices: np.ndarray) -> np.ndarray:
    """Extrai de `rho_all` (alinhado com `cached_t_indices`, ordenado) as linhas
    correspondentes a `query_t_indices`, na ordem de `query_t_indices`."""
    pos = np.searchsorted(cached_t_indices, query_t_indices)
    if not np.array_equal(cached_t_indices[pos], query_t_indices):
        raise ValueError("query_t_indices contém t ausente de cached_t_indices.")
    return rho_all[pos]


def _valid_t_indices(y: np.ndarray, window_L: int) -> np.ndarray:
    T = y.shape[0]
    t = np.arange(window_L - 1, T)
    valid = ~np.isnan(y[t])
    return t[valid]


def build_dataset(cfg: dict, seeds: dict) -> Dataset:
    from qrcshots.config import rng_for

    task: Task = build_task(cfg)
    rng = rng_for(seeds, "signal")
    u_raw = task.generate_input(cfg["signal"]["T"], rng)
    y = task.target(u_raw)
    u_injection = task.encode_for_injection(u_raw)

    window_L = cfg["reservoir"]["window_L"]
    valid_t = _valid_t_indices(y, window_L)

    n = valid_t.shape[0]
    n_train = int(round(n * cfg["split"]["train_frac"]))
    n_val = int(round(n * cfg["split"]["val_frac"]))
    train_idx = valid_t[:n_train]
    val_idx = valid_t[n_train : n_train + n_val]
    test_idx = valid_t[n_train + n_val :]

    return Dataset(u=u_injection, y=y, window_L=window_L, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, task_name=task.name)
