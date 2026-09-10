"""Fase 2 (Seção 10): aprendizado conjunto de pesos do readout e alocação de shots.

Loop, a um orçamento POR JANELA fixo `B` (independente dos orçamentos de teste da
fase 1 -- aqui é o custo de *treinar*, não de prever):

    repita:
      1. colete features de treino/val sob a alocação atual (shots frescos)
      2. treine w (ridge, alpha por busca em grade na validação)
      3. recalcule s_g (passe único) com o w novo; realoque para o ótimo de `variance`
      4. avalie NMSE de validação sob a alocação atual; pare se não melhorar por
         `patience` iterações consecutivas (mantendo o melhor w/alocação já visto)

Tudo isso só toca treino/validação -- a mesma guarda de `Dataset.freeze()` do resto
do projeto vale aqui; o teste só é avaliado uma vez, no final, com o melhor
readout+alocação encontrados.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qrcshots.allocate import GROUPS, allocate_uniform, allocate_variance, s_g_single_pass_empirical
from qrcshots.config import rng_for
from qrcshots.experiment import PipelineContext
from qrcshots.readout import FrozenReadout, collect_features_with_allocation, fit_frozen_readout, nmse


@dataclass(frozen=True)
class JointIterationLog:
    iteration: int
    allocation: dict[str, int]
    val_nmse: float
    alpha: float
    shots_spent_this_iteration: int


@dataclass(frozen=True)
class JointResult:
    history: list[JointIterationLog]
    best_iteration: int
    best_readout: FrozenReadout
    best_allocation: dict[str, int]
    total_shots_spent: int
    converged: bool


def run_joint_loop(ctx: PipelineContext, budget_per_window: int, n_min: int, alpha_grid: np.ndarray, n_cal_shots: int, max_iterations: int, patience: int) -> JointResult:
    seeds = ctx.seeds
    n_ridge_train = ctx.rho_ridge_train.shape[0]
    n_val = ctx.rho_val.shape[0]
    n_calib = ctx.rho_calib.shape[0]

    allocation = allocate_uniform(GROUPS, budget_per_window, n_min)
    history: list[JointIterationLog] = []
    best: tuple[float, FrozenReadout, dict[str, int], int] | None = None
    no_improve = 0
    total_shots = 0

    for it in range(max_iterations):
        rng_train = rng_for(seeds, "phase2_joint", index=4 * it)
        rng_val = rng_for(seeds, "phase2_joint", index=4 * it + 1)
        X_train = collect_features_with_allocation(ctx.rho_ridge_train, allocation, rng_train, ctx.n_qubits)
        X_val = collect_features_with_allocation(ctx.rho_val, allocation, rng_val, ctx.n_qubits)
        shots_this_iter = sum(allocation.values()) * (n_ridge_train + n_val)
        total_shots += shots_this_iter

        readout, _ = fit_frozen_readout(X_train, ctx.y_ridge_train, X_val, ctx.val_split.y, alpha_grid, ctx.n_qubits)
        val_nmse = nmse(ctx.val_split.y, readout.predict(X_val))
        history.append(JointIterationLog(iteration=it, allocation=dict(allocation), val_nmse=val_nmse, alpha=readout.alpha, shots_spent_this_iteration=shots_this_iter))

        if best is None or val_nmse < best[0] - 1e-9:
            best = (val_nmse, readout, dict(allocation), it)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience or it == max_iterations - 1:
            break

        rng_calib = rng_for(seeds, "phase2_joint", index=4 * it + 2)
        s_g = {g: s_g_single_pass_empirical(readout, g, ctx.rho_calib, n_cal_shots, rng_calib) for g in GROUPS}
        total_shots += n_cal_shots * len(GROUPS) * n_calib
        allocation = allocate_variance(s_g, budget_per_window, n_min)

    assert best is not None
    return JointResult(
        history=history,
        best_iteration=best[3],
        best_readout=best[1],
        best_allocation=best[2],
        total_shots_spent=total_shots,
        converged=no_improve >= patience,
    )
