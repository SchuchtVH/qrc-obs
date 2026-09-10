"""Sinal de entrada e tarefas temporais (Seção 2).

Cada tarefa expõe `generate_input(T, rng)` e `target(u)`; `target` devolve um array do
mesmo tamanho de `u` com `NaN` nas posições em que o alvo não está definido (antes do
maior lag exigido pela tarefa).

`encode_for_injection(u)` é uma transformação separada, aplicada só ao sinal que entra
no reservatório (`theta = pi*(u'+1)/2`), nunca ao `u` usado para calcular `y`. Para
`primary`/`product3`, u já vive em [-1,1] e a codificação é a identidade. Para
`narma5`, o enunciado especifica u ~ Uniform(0, 0.2) -- uma faixa estreita que, mapeada
por theta = pi*(u+1)/2, cobre só um arco de ~10 graus na esfera de Bloch. Isso faz a
dispersão das features exatas (~0.003-0.04) ficar bem abaixo do ruído de 64 shots
(~0.125), então o readout treinado com shots reais aprende essencialmente nada (NMSE
piso ~0.99) mesmo com um reservatório que decodifica bem em features exatas (NMSE
~0.16). Reescalonar SÓ a injeção para o range completo (u' = (u-centro)/semi_faixa,
cobrindo theta em [0,pi]) amplifica a dispersão das features sem alterar a tarefa (a
recorrência de y e o valor de y continuam exatamente sobre o u original). Ver
discussão em RESULTS.md/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Task:
    name: str
    generate_input: Callable[[int, np.random.Generator], np.ndarray]
    target: Callable[[np.ndarray], np.ndarray]
    max_lag: int
    encode_for_injection: Callable[[np.ndarray], np.ndarray] = lambda u: u


def _generate_uniform(low: float, high: float) -> Callable[[int, np.random.Generator], np.ndarray]:
    def gen(T: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low, high, size=T)

    return gen


def primary_target(u: np.ndarray) -> np.ndarray:
    """y_t = u_{t-2} * u_{t-5}."""
    y = np.full(u.shape[0], np.nan)
    y[5:] = u[3:-2] * u[:-5]
    return y


def product3_target(u: np.ndarray, lags: tuple[int, int, int] = (1, 3, 7)) -> np.ndarray:
    """y_t = u_{t-lag0} * u_{t-lag1} * u_{t-lag2}."""
    l0, l1, l2 = lags
    max_lag = max(lags)
    T = u.shape[0]
    y = np.full(T, np.nan)
    idx = np.arange(max_lag, T)
    y[idx] = u[idx - l0] * u[idx - l1] * u[idx - l2]
    return y


def narma5_target(u: np.ndarray, n: int = 5, alpha: float = 0.3, beta: float = 0.05, gamma: float = 1.5, delta: float = 0.1) -> np.ndarray:
    """NARMA de ordem n=5, generalização de NARMA10 (Rodan & Tino) para n menor:

    y[t] = alpha*y[t-1] + beta*y[t-1]*sum(y[t-n:t]) + gamma*u[t-n]*u[t-1] + delta

    para t >= n; y[0:n] = 0. Recorrência assumida estável para u ~ Uniform(0, 0.2)
    (faixa reduzida especificada na Seção 2 exatamente para evitar a instabilidade
    conhecida de NARMA de ordem baixa com entradas maiores).
    """
    T = u.shape[0]
    y = np.zeros(T)
    for t in range(n, T):
        y[t] = alpha * y[t - 1] + beta * y[t - 1] * np.sum(y[t - n : t]) + gamma * u[t - n] * u[t - 1] + delta
    y_out = y.copy()
    y_out[:n] = np.nan
    return y_out


def _rescale_to_full_range(low: float, high: float) -> Callable[[np.ndarray], np.ndarray]:
    center = (low + high) / 2.0
    half_range = (high - low) / 2.0

    def encode(u: np.ndarray) -> np.ndarray:
        return (u - center) / half_range

    return encode


def build_task(cfg: dict) -> Task:
    name = cfg["task"]["name"]
    if name == "primary":
        return Task("primary", _generate_uniform(cfg["signal"]["u_low"], cfg["signal"]["u_high"]), primary_target, max_lag=5)
    if name == "narma5":
        p = cfg["task"]["narma5"]
        return Task(
            "narma5",
            _generate_uniform(p["u_low"], p["u_high"]),
            narma5_target,
            max_lag=5,
            encode_for_injection=_rescale_to_full_range(p["u_low"], p["u_high"]),
        )
    if name == "product3":
        lags = tuple(cfg["task"]["product3"]["lags"])
        return Task(
            "product3",
            _generate_uniform(cfg["signal"]["u_low"], cfg["signal"]["u_high"]),
            lambda u: product3_target(u, lags),
            max_lag=max(lags),
        )
    raise ValueError(f"Tarefa desconhecida: {name!r}")
