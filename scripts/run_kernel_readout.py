#!/usr/bin/env python
"""Script fino: baseline `kernel_readout` (Seção 8) -- leitura ótima por kernel de
Hilbert-Schmidt, restrita ao suporte medível, comparada ao ridge do marco 1.

Uso:
    python scripts/run_kernel_readout.py [--config configs/base.yaml] [a.b.c=valor ...]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from qrcshots.allocate import GROUPS, s_g_exact, s_g_single_pass_empirical, allocate_variance
from qrcshots.config import apply_overrides, load_config, rng_for
from qrcshots.experiment import (
    build_pipeline_context,
    fit_readout_and_diagnostics,
    unlock_test_data,
    evaluate_all_seeds,
)
from qrcshots.kernel_readout import (
    build_o_star,
    norm_fraction_outside_support,
    restricted_readout_from_coeffs,
    select_kernel_ridge,
    weight1_pauli_coeffs,
)
from qrcshots.readout import exact_features, nmse


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    ctx = build_pipeline_context(cfg)
    print(f"[kernel_readout] tarefa={cfg['task']['name']} T={cfg['signal']['T']}")
    print(f"[kernel_readout] janelas: ridge_train={ctx.rho_ridge_train.shape[0]} calib={ctx.rho_calib.shape[0]} val={ctx.rho_val.shape[0]}")

    lambda_grid = np.logspace(-8, 2, 21)
    fit, lambda_results = select_kernel_ridge(ctx.rho_ridge_train, ctx.y_ridge_train, ctx.rho_val, ctx.val_split.y, lambda_grid)
    print(f"[kernel_readout] lambda selecionado={fit.lam:.3g}")

    o_star = build_o_star(ctx.rho_ridge_train, fit.alpha)
    coeffs = weight1_pauli_coeffs(o_star, ctx.n_qubits)
    norm_info = norm_fraction_outside_support(o_star, coeffs, ctx.n_qubits)
    print(f"[kernel_readout] fração da norma de O* fora do suporte medível: {norm_info['fraction_outside_support']:.4f}")

    restricted_readout = restricted_readout_from_coeffs(coeffs, fit.y_mean, ctx.n_qubits)

    # (a) NMSE com shots infinitos: kernel_readout restrito vs ridge exato do marco 1.
    ridge_result = fit_readout_and_diagnostics(ctx)
    test_data = unlock_test_data(ctx)
    X_test_exact = exact_features(test_data.rho_test, ctx.n_qubits)

    kernel_exact_nmse = nmse(test_data.y_test, restricted_readout.predict(X_test_exact))
    ridge_exact_nmse = nmse(test_data.y_test, ridge_result.exact_readout.predict(X_test_exact))
    print(f"[kernel_readout] NMSE exato (shots->inf) kernel_readout restrito: {kernel_exact_nmse:.4f}")
    print(f"[kernel_readout] NMSE exato (shots->inf) ridge (marco 1, treinado c/ features exatas): {ridge_exact_nmse:.4f}")

    # (c) NMSE com shots finitos, alocação `variance` para os pesos de O* restrito.
    n_min = cfg["allocation"]["n_min"]
    budgets = list(cfg["allocation"]["budgets"])
    rng_calib = rng_for(ctx.seeds, "calibration_empirical_repeats", index=2)
    s_g_kernel = {g: s_g_single_pass_empirical(restricted_readout, g, ctx.rho_calib, cfg["calibration"]["n_cal_shots"], rng_calib) for g in GROUPS}
    print(f"[kernel_readout] s_g (passe único) do readout restrito: {s_g_kernel}")

    allocations = {b: {"kernel_variance": allocate_variance(s_g_kernel, b, n_min)} for b in budgets}
    n_seeds = cfg["experiment"]["n_seeds"]
    results_df = evaluate_all_seeds(ctx, restricted_readout, test_data, allocations, n_seeds)
    summary = results_df.groupby("budget")["nmse"].agg(["mean", "std"])
    print("[kernel_readout] NMSE com shots finitos (alocação variance sobre O* restrito):")
    print(summary.to_string())

    out = {
        "lambda_selected": fit.lam,
        "lambda_grid_results": lambda_results,
        "norm_fraction_outside_support": norm_info,
        "kernel_readout_exact_nmse": kernel_exact_nmse,
        "ridge_exact_nmse": ridge_exact_nmse,
        "s_g_kernel_readout": s_g_kernel,
        "finite_shot_nmse_by_budget": {int(b): {"mean": float(summary.loc[b, "mean"]), "std": float(summary.loc[b, "std"])} for b in budgets},
        "weight1_coeffs": {g: coeffs[g].tolist() for g in GROUPS},
    }
    out_path = "results/raw/kernel_readout_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[kernel_readout] resumo salvo em {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
