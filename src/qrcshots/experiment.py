"""Orquestração do experimento fase 1 (Seções 5-9): prepara o pipeline (reservatório
fixo, readout congelado, SHAP, calibração), calcula as alocações por estratégia e
orçamento, e avalia todas no teste com amostragem pareada entre estratégias e
orçamentos.

Fases, na ordem em que tocam dados (a guarda de `Dataset.freeze()` protege a
fronteira entre elas):

1. `build_pipeline_context`: monta reservatório, cache de rho, e separa
   treino -> (ridge-train, calibração) + validação. **Não toca teste.**
2. `fit_readout_and_diagnostics`, `compute_shap`, `compute_calibration_s_g`,
   `compute_budget_allocations`: todas as decisões (pesos, SHAP, alocação de
   uniform/shap/variance/random/magnitude). **Ainda não tocam teste.**
3. `unlock_test_data`: chama `dataset.freeze()` e devolve rho/y de teste -- só a
   partir daqui `oracle` (que precisa de estatística exata do teste) e a avaliação
   final podem ser calculadas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qrcshots.allocate import (
    allocate_magnitude,
    allocate_oracle,
    allocate_random,
    allocate_shap,
    allocate_uniform,
    allocate_variance,
    s_g_exact,
    s_g_single_pass_empirical,
)
from qrcshots.allocate import s_g_empirical as _s_g_empirical
from qrcshots.config import derive_seeds, rng_for
from qrcshots.dataset import Dataset, Split, build_dataset, select_rho
from qrcshots.measure import GROUPS, bit_signs, draw_shot_indices
from qrcshots.readout import (
    FrozenReadout,
    alpha_grid_from_cfg,
    collect_features,
    exact_features,
    fit_frozen_readout,
    nmse,
    rmse,
)
from qrcshots.reservoir import build_or_load_hamiltonian, compute_or_load_rho_windows

CORE_STRATEGIES = ("uniform", "shap", "variance", "random", "magnitude")
ALL_STRATEGIES_WITH_ORACLE = CORE_STRATEGIES + ("oracle",)

# estratégias sem custo de calibração adicional além da coleta inicial comum
# (Seção 9: "uniform tem overhead zero" -- generalizado aqui para toda estratégia
# que não exige uma medição de calibração dedicada; só `variance` exige).
ZERO_CALIBRATION_COST_STRATEGIES = ("uniform", "shap", "random", "magnitude", "oracle")


@dataclass
class PipelineContext:
    cfg: dict
    seeds: dict
    dataset: Dataset
    n_qubits: int
    U: np.ndarray
    rho_all: np.ndarray
    all_valid_t: np.ndarray
    rho_ridge_train: np.ndarray
    rho_calib: np.ndarray
    rho_val: np.ndarray
    y_ridge_train: np.ndarray
    y_calib: np.ndarray
    val_split: Split


def build_pipeline_context(cfg: dict) -> PipelineContext:
    seeds = derive_seeds(cfg)
    dataset = build_dataset(cfg, seeds)
    n_qubits = cfg["reservoir"]["n_qubits"]
    cache_dir = cfg["paths"]["cache_dir"]

    _, U = build_or_load_hamiltonian(cfg, seeds, cache_dir)
    rho_all = compute_or_load_rho_windows(cfg, dataset.u, dataset.all_valid_t, U, cache_dir)

    train_split = dataset.train
    val_split = dataset.val

    holdout_frac = cfg["calibration"]["holdout_frac"]
    perm_rng = rng_for(seeds, "calibration_holdout_split")
    n_train = len(train_split.t_indices)
    perm = perm_rng.permutation(n_train)
    n_calib = int(round(n_train * holdout_frac))
    calib_pos, ridge_train_pos = perm[:n_calib], perm[n_calib:]

    ridge_train_t = np.sort(train_split.t_indices[ridge_train_pos])
    calib_t = np.sort(train_split.t_indices[calib_pos])

    y_by_t = dict(zip(train_split.t_indices.tolist(), train_split.y.tolist()))
    y_ridge_train = np.array([y_by_t[t] for t in ridge_train_t])
    y_calib = np.array([y_by_t[t] for t in calib_t])

    rho_ridge_train = select_rho(rho_all, dataset.all_valid_t, ridge_train_t)
    rho_calib = select_rho(rho_all, dataset.all_valid_t, calib_t)
    rho_val = select_rho(rho_all, dataset.all_valid_t, val_split.t_indices)

    return PipelineContext(
        cfg=cfg,
        seeds=seeds,
        dataset=dataset,
        n_qubits=n_qubits,
        U=U,
        rho_all=rho_all,
        all_valid_t=dataset.all_valid_t,
        rho_ridge_train=rho_ridge_train,
        rho_calib=rho_calib,
        rho_val=rho_val,
        y_ridge_train=y_ridge_train,
        y_calib=y_calib,
        val_split=val_split,
    )


@dataclass
class ReadoutResult:
    readout: FrozenReadout
    X_ridge_train_raw: np.ndarray
    X_val_raw: np.ndarray
    alpha_grid_results: list[dict]
    exact_readout: FrozenReadout
    exact_readout_val_nmse: float


def fit_readout_and_diagnostics(ctx: PipelineContext) -> ReadoutResult:
    cfg = ctx.cfg
    n_shots = cfg["collection"]["train_shots_per_group"]
    rng = rng_for(ctx.seeds, "initial_feature_collection")

    X_ridge_train_raw = collect_features(ctx.rho_ridge_train, n_shots, rng, ctx.n_qubits)
    X_val_raw = collect_features(ctx.rho_val, n_shots, rng, ctx.n_qubits)

    alpha_grid = alpha_grid_from_cfg(cfg)
    readout, grid_results = fit_frozen_readout(X_ridge_train_raw, ctx.y_ridge_train, X_val_raw, ctx.val_split.y, alpha_grid, ctx.n_qubits)

    # diagnóstico de errors-in-variables (apêndice, Seção 5): readout treinado com
    # features EXATAS (shots -> infinito), não entra na comparação principal.
    X_ridge_train_exact = exact_features(ctx.rho_ridge_train, ctx.n_qubits)
    X_val_exact = exact_features(ctx.rho_val, ctx.n_qubits)
    exact_readout, _ = fit_frozen_readout(X_ridge_train_exact, ctx.y_ridge_train, X_val_exact, ctx.val_split.y, alpha_grid, ctx.n_qubits)
    exact_readout_val_nmse = nmse(ctx.val_split.y, exact_readout.predict(X_val_exact))

    return ReadoutResult(
        readout=readout,
        X_ridge_train_raw=X_ridge_train_raw,
        X_val_raw=X_val_raw,
        alpha_grid_results=grid_results,
        exact_readout=exact_readout,
        exact_readout_val_nmse=exact_readout_val_nmse,
    )


def compute_shap_importance(ctx: PipelineContext, readout: FrozenReadout, X_ridge_train_raw: np.ndarray) -> dict[str, float]:
    from qrcshots.shapley import shap_group_importance

    X_std = readout.scaler.transform(X_ridge_train_raw)
    return shap_group_importance(readout, X_std, ctx.n_qubits)


@dataclass
class CalibrationResult:
    s_g_single_pass: dict[str, float]  # custo real de calibração (deployável); alimenta a estratégia `variance`
    s_g_empirical_r_repeat: dict[str, float]  # protocolo de checagem de sanidade da Seção 6.2 (R repetições)
    s_g_exact: dict[str, float]  # referência exata, para validar os dois empíricos


def compute_calibration_s_g(ctx: PipelineContext, readout: FrozenReadout) -> CalibrationResult:
    """Duas estimativas empíricas de s_g, com papéis diferentes:

    - `s_g_single_pass`: UM lote de `n_cal_shots` por janela de calibração -- é o que
      um deployment real pagaria, e é o que decide a alocação da estratégia `variance`
      e entra na contabilidade de custo (Seção 9).
    - `s_g_empirical_r_repeat`: repete cada janela `n_repeats` vezes (Seção 6.2) --
      só para o teste de sanidade "a concordância entre as duas é obrigatória" contra
      `s_g_exact`; não é usado para decidir nenhuma alocação nem entra no custo (custaria
      `n_repeats`x mais shots do que se pagaria de fato).
    """
    cal_cfg = ctx.cfg["calibration"]
    rng_single = rng_for(ctx.seeds, "calibration_empirical_repeats", index=0)
    rng_repeat = rng_for(ctx.seeds, "calibration_empirical_repeats", index=1)
    s_single = {g: s_g_single_pass_empirical(readout, g, ctx.rho_calib, cal_cfg["n_cal_shots"], rng_single) for g in GROUPS}
    s_repeat = {g: _s_g_empirical(readout, g, ctx.rho_calib, cal_cfg["n_cal_shots"], cal_cfg["n_repeats"], rng_repeat) for g in GROUPS}
    s_ex = {g: s_g_exact(readout, g, ctx.rho_calib) for g in GROUPS}
    return CalibrationResult(s_g_single_pass=s_single, s_g_empirical_r_repeat=s_repeat, s_g_exact=s_ex)


def compute_budget_allocations(ctx: PipelineContext, readout: FrozenReadout, shap_importance: dict[str, float], s_g_for_allocation: dict[str, float], budgets: list[int]) -> dict[int, dict[str, dict[str, int]]]:
    """Alocação de cada estratégia CORE (sem oracle) para cada orçamento -- global e
    fixa, calculada só a partir de treino+calibração (Seção 7). `s_g_for_allocation`
    deve ser a estimativa de passe único (`CalibrationResult.s_g_single_pass`), não a
    de R repetições."""
    n_min = ctx.cfg["allocation"]["n_min"]
    out: dict[int, dict[str, dict[str, int]]] = {}
    for i, budget in enumerate(budgets):
        rng = rng_for(ctx.seeds, "baseline_random_dirichlet", index=i)
        out[budget] = {
            "uniform": allocate_uniform(GROUPS, budget, n_min),
            "shap": allocate_shap(shap_importance, budget, n_min),
            "variance": allocate_variance(s_g_for_allocation, budget, n_min),
            "random": allocate_random(GROUPS, budget, n_min, rng),
            "magnitude": allocate_magnitude(readout, budget, n_min),
        }
    return out


@dataclass
class TestData:
    rho_test: np.ndarray
    y_test: np.ndarray
    t_indices: np.ndarray


def unlock_test_data(ctx: PipelineContext) -> TestData:
    """Única porta de entrada para o split de teste em todo o pipeline. Chama
    `Dataset.freeze()`; qualquer acesso anterior a isto já teria levantado
    `TestSetAccessedBeforeFreeze`."""
    ctx.dataset.freeze()
    test_split = ctx.dataset.test
    rho_test = select_rho(ctx.rho_all, ctx.all_valid_t, test_split.t_indices)
    return TestData(rho_test=rho_test, y_test=test_split.y, t_indices=test_split.t_indices)


def compute_oracle_allocations(readout: FrozenReadout, test_data: TestData, budgets: list[int], n_min: int, resolution: int) -> dict[int, dict[str, int]]:
    """`oracle`: minimiza a variância analítica de shot noise usando Sigma_g EXATO do
    próprio teste -- só possível porque estamos em simulação; rotulado
    não-implementável (usa conhecimento do teste). O termo de "piso" do reservatório
    (erro do readout com features exatas) não depende da alocação, então minimizar
    sum_g s_g_teste^2/n_g é equivalente, no orçamento de shot, a minimizar o NMSE de
    teste -- sem precisar re-simular o pipeline inteiro por ponto de grade."""
    s_g_test = {g: s_g_exact(readout, g, test_data.rho_test) for g in GROUPS}

    out = {}
    for budget in budgets:
        def eval_fn(alloc, s_g_test=s_g_test):
            return sum(s_g_test[g] ** 2 / alloc[g] for g in GROUPS)

        alloc, _ = allocate_oracle(GROUPS, budget, n_min, resolution, eval_fn)
        out[budget] = alloc
    return out


def _batched_counts(idx2d: np.ndarray, dim: int) -> np.ndarray:
    """counts[k, s] = número de shots iguais a s na janela k, para idx2d (n_windows, n_shots)."""
    n_windows = idx2d.shape[0]
    offsets = (np.arange(n_windows) * dim)[:, None]
    flat = (idx2d.astype(np.int64) + offsets).ravel()
    return np.bincount(flat, minlength=n_windows * dim).reshape(n_windows, dim)


def evaluate_all_seeds(ctx: PipelineContext, readout: FrozenReadout, test_data: TestData, allocations_per_budget: dict[int, dict[str, dict[str, int]]], n_seeds: int) -> pd.DataFrame:
    """Avalia todas as (estratégia, orçamento, seed) no teste. Pareamento (Seção 8):
    para cada seed, sorteamos uma sequência-mestre de índices de bitstring de tamanho
    `max(budgets)` por (janela, grupo) UMA vez; toda estratégia e todo orçamento usa um
    prefixo dessa mesma sequência -- pareamento mais forte que o exigido (o enunciado
    pede pareamento só entre estratégias de um mesmo orçamento; aqui também é pareado
    entre orçamentos, o que só reduz mais a variância da comparação)."""
    n_qubits = ctx.n_qubits
    dim = 2**n_qubits
    signs = bit_signs(n_qubits)
    n_windows = test_data.rho_test.shape[0]
    max_budget = max(allocations_per_budget.keys())

    rows = []
    for seed_idx in range(n_seeds):
        rng = rng_for(ctx.seeds, "experiment_seeds_base", index=seed_idx)
        master = {
            g: np.stack([draw_shot_indices(test_data.rho_test[k], g, max_budget, rng, n_qubits) for k in range(n_windows)])
            for g in GROUPS
        }
        for budget, strategies in allocations_per_budget.items():
            for name, alloc in strategies.items():
                feats = []
                for g in GROUPS:
                    n_g = alloc[g]
                    counts = _batched_counts(master[g][:, :n_g], dim)
                    feats.append(counts @ signs / n_g)
                X_raw = np.concatenate(feats, axis=1)
                y_hat = readout.predict(X_raw)
                rows.append(
                    {
                        "seed": seed_idx,
                        "budget": budget,
                        "strategy": name,
                        "nmse": nmse(test_data.y_test, y_hat),
                        "rmse": rmse(test_data.y_test, y_hat),
                    }
                )
    return pd.DataFrame(rows)


def compute_floor_nmse(readout: FrozenReadout, test_data: TestData, n_qubits: int) -> float:
    """NMSE com shots infinitos (features exatas) -- piso de erro do reservatório
    para ESTE readout (Seção 9)."""
    X_exact = exact_features(test_data.rho_test, n_qubits)
    return nmse(test_data.y_test, readout.predict(X_exact))


def cost_accounting(ctx: PipelineContext, n_test_windows: int) -> dict:
    cfg = ctx.cfg
    train_shots = cfg["collection"]["train_shots_per_group"]
    n_groups = len(GROUPS)
    n_ridge_train = ctx.rho_ridge_train.shape[0]
    n_val = ctx.rho_val.shape[0]
    n_calib = ctx.rho_calib.shape[0]

    shots_coleta_inicial = train_shots * n_groups * (n_ridge_train + n_val)

    cal_cfg = cfg["calibration"]
    # custo real de calibração de `variance`: um único lote de n_cal_shots por janela
    # (s_g_single_pass_empirical), não o protocolo de R repetições (que é só a
    # checagem de sanidade da Seção 6.2 contra s_g_exact, nunca pago de fato).
    shots_calibracao_variance = cal_cfg["n_cal_shots"] * n_groups * n_calib
    shots_calibracao_variance_sanity_check_r_repeat = cal_cfg["n_cal_shots"] * cal_cfg["n_repeats"] * n_groups * n_calib

    shots_calibracao = {s: 0 for s in ZERO_CALIBRATION_COST_STRATEGIES}
    shots_calibracao["variance"] = shots_calibracao_variance

    return {
        "shots_coleta_inicial": shots_coleta_inicial,
        "shots_calibracao_por_estrategia": shots_calibracao,
        "shots_calibracao_variance_sanity_check_r_repeat": shots_calibracao_variance_sanity_check_r_repeat,
        "window_L": cfg["reservoir"]["window_L"],
        "n_test_windows": n_test_windows,
        "n_ridge_train_windows": n_ridge_train,
        "n_val_windows": n_val,
        "n_calib_windows": n_calib,
    }


def save_phase1_outputs(cfg: dict, results_df: pd.DataFrame, metadata: dict, results_dir: str | Path) -> Path:
    from qrcshots.config import config_hash

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    h = config_hash(cfg)
    parquet_path = results_dir / f"phase1_{h}.parquet"
    meta_path = results_dir / f"phase1_{h}_meta.json"
    results_df.to_parquet(parquet_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=float)
    return parquet_path
