#!/usr/bin/env python
"""Script fino: roda o experimento fase 1 completo (Seções 5-9) e salva resultados.

Uso:
    python scripts/run_phase1.py [--config configs/base.yaml] [a.b.c=valor ...]

Exemplo de smoke run (Seção 13):
    python scripts/run_phase1.py signal.T=600 experiment.n_seeds=3 \
        allocation.budgets='[96,300]'
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from qrcshots.allocate import GROUPS
from qrcshots.config import apply_overrides, load_config
from qrcshots.experiment import (
    build_pipeline_context,
    compute_budget_allocations,
    compute_calibration_s_g,
    compute_floor_nmse,
    compute_oracle_allocations,
    compute_shap_importance,
    cost_accounting,
    evaluate_all_seeds,
    fit_readout_and_diagnostics,
    save_phase1_outputs,
    unlock_test_data,
)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*", help="overrides a.b.c=valor")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    t0 = time.time()
    print(f"[run_phase1] tarefa={cfg['task']['name']} T={cfg['signal']['T']} n_seeds={cfg['experiment']['n_seeds']} budgets={cfg['allocation']['budgets']}")

    ctx = build_pipeline_context(cfg)
    print(f"[run_phase1] janelas: ridge_train={ctx.rho_ridge_train.shape[0]} calib={ctx.rho_calib.shape[0]} val={ctx.rho_val.shape[0]}")

    readout_result = fit_readout_and_diagnostics(ctx)
    print(f"[run_phase1] alpha selecionado={readout_result.readout.alpha:.3g}")

    shap_importance = compute_shap_importance(ctx, readout_result.readout, readout_result.X_ridge_train_raw)
    print(f"[run_phase1] importância SHAP por grupo: {shap_importance}")

    calib = compute_calibration_s_g(ctx, readout_result.readout)
    print(f"[run_phase1] s_g passe único (usado na alocação): {calib.s_g_single_pass}")
    print(f"[run_phase1] s_g empírico R-repetições (checagem): {calib.s_g_empirical_r_repeat}")
    print(f"[run_phase1] s_g exato:                            {calib.s_g_exact}")

    budgets = list(cfg["allocation"]["budgets"])
    allocations = compute_budget_allocations(ctx, readout_result.readout, shap_importance, calib.s_g_single_pass, budgets)

    test_data = unlock_test_data(ctx)
    print(f"[run_phase1] janelas de teste: {test_data.rho_test.shape[0]}")

    oracle_allocs = compute_oracle_allocations(
        readout_result.readout, test_data, budgets, cfg["allocation"]["n_min"], cfg["allocation"]["oracle_grid_resolution"]
    )
    for budget in budgets:
        allocations[budget]["oracle"] = oracle_allocs[budget]

    n_seeds = cfg["experiment"]["n_seeds"]
    results_df = evaluate_all_seeds(ctx, readout_result.readout, test_data, allocations, n_seeds)

    floor_nmse = compute_floor_nmse(readout_result.readout, test_data, ctx.n_qubits)
    exact_val_nmse = readout_result.exact_readout_val_nmse
    weight_norm_noisy = float(np.linalg.norm(readout_result.readout.weights))
    weight_norm_exact = float(np.linalg.norm(readout_result.exact_readout.weights))

    cost = cost_accounting(ctx, n_test_windows=test_data.rho_test.shape[0])

    metadata = {
        "config": cfg,
        "task_name": ctx.dataset.task_name,
        "alpha_selected": readout_result.readout.alpha,
        "alpha_grid_results": readout_result.alpha_grid_results,
        "shap_importance": shap_importance,
        "s_g_single_pass": calib.s_g_single_pass,
        "s_g_empirical_r_repeat": calib.s_g_empirical_r_repeat,
        "s_g_exact": calib.s_g_exact,
        "allocations_per_budget": allocations,
        "floor_nmse": floor_nmse,
        "errors_in_variables": {
            "weight_norm_noisy_readout": weight_norm_noisy,
            "weight_norm_exact_readout": weight_norm_exact,
            "val_nmse_exact_readout": exact_val_nmse,
        },
        "cost_accounting": cost,
        "elapsed_seconds": time.time() - t0,
    }

    path = save_phase1_outputs(cfg, results_df, metadata, cfg["paths"]["results_dir"])
    print(f"[run_phase1] resultados salvos em {path}")

    summary = results_df.groupby(["budget", "strategy"])["nmse"].mean().unstack("strategy")
    print("[run_phase1] NMSE médio por (orçamento, estratégia):")
    print(summary.to_string())
    print(f"[run_phase1] piso de erro do reservatório (NMSE, shots->inf): {floor_nmse:.5g}")
    print(f"[run_phase1] concluído em {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
