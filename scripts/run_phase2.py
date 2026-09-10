#!/usr/bin/env python
"""Script fino: fase 2 (Seção 10) -- aprendizado conjunto de pesos e alocação.

Uso:
    python scripts/run_phase2.py [--config configs/base.yaml] [a.b.c=valor ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from qrcshots.allocate import GROUPS, allocate_variance
from qrcshots.config import apply_overrides, load_config, rng_for
from qrcshots.experiment import build_pipeline_context, unlock_test_data
from qrcshots.joint import run_joint_loop
from qrcshots.readout import alpha_grid_from_cfg, exact_features, nmse


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--budget-per-window", type=int, default=300)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    t0 = time.time()
    ctx = build_pipeline_context(cfg)
    alpha_grid = alpha_grid_from_cfg(cfg)
    n_min = cfg["allocation"]["n_min"]
    n_cal_shots = cfg["calibration"]["n_cal_shots"]
    max_it = cfg["phase2"]["max_iterations"]
    patience = cfg["phase2"]["patience"]

    print(f"[run_phase2] budget_per_window={args.budget_per_window} max_iterations={max_it} patience={patience}")
    result = run_joint_loop(ctx, args.budget_per_window, n_min, alpha_grid, n_cal_shots, max_it, patience)

    print(f"[run_phase2] convergiu={result.converged} em {len(result.history)} iterações (melhor: it={result.best_iteration})")
    for h in result.history:
        print(f"  it={h.iteration:2d} val_nmse={h.val_nmse:.5f} alpha={h.alpha:.3g} alloc={h.allocation} shots_iter={h.shots_spent_this_iteration}")
    print(f"[run_phase2] custo total de treino do loop: {result.total_shots_spent} shots")

    # avaliação final no teste -- única vez, com o melhor readout+alocação achados.
    test_data = unlock_test_data(ctx)
    rng_test = rng_for(ctx.seeds, "phase2_joint", index=99999)
    from qrcshots.readout import collect_features_with_allocation

    X_test = collect_features_with_allocation(test_data.rho_test, result.best_allocation, rng_test, ctx.n_qubits)
    test_nmse_joint = nmse(test_data.y_test, result.best_readout.predict(X_test))
    print(f"[run_phase2] NMSE de teste (fase 2, alocação {result.best_allocation}): {test_nmse_joint:.5f}")

    # comparação: fase-1 'variance' fixa no MESMO orçamento por janela, sem o loop
    # conjunto (readout treinado uma vez com coleta uniforme, como no marco 1).
    from qrcshots.experiment import fit_readout_and_diagnostics, compute_calibration_s_g

    rr = fit_readout_and_diagnostics(ctx)
    calib = compute_calibration_s_g(ctx, rr.readout)
    phase1_alloc = allocate_variance(calib.s_g_single_pass, args.budget_per_window, n_min)
    rng_test2 = rng_for(ctx.seeds, "phase2_joint", index=99998)
    X_test_phase1 = collect_features_with_allocation(test_data.rho_test, phase1_alloc, rng_test2, ctx.n_qubits)
    test_nmse_phase1 = nmse(test_data.y_test, rr.readout.predict(X_test_phase1))
    print(f"[run_phase2] NMSE de teste (fase 1, 'variance' no mesmo orçamento, sem loop): {test_nmse_phase1:.5f}")

    gain = test_nmse_phase1 - test_nmse_joint
    print(f"[run_phase2] ganho absoluto de NMSE da fase 2 sobre a fase 1: {gain:.5f}")
    print(f"[run_phase2] custo adicional da fase 2 (loop): {result.total_shots_spent} shots")
    if gain <= 0:
        print("[run_phase2] CONCLUSÃO: a fase 2 NÃO melhorou sobre a fase 1 neste orçamento -- o custo do loop não se paga.")
    else:
        print(f"[run_phase2] CONCLUSÃO: a fase 2 melhorou o NMSE em {gain:.5f} ao custo de {result.total_shots_spent} shots de treino extra.")

    out = {
        "budget_per_window": args.budget_per_window,
        "converged": result.converged,
        "n_iterations": len(result.history),
        "best_iteration": result.best_iteration,
        "total_shots_spent": result.total_shots_spent,
        "history": [
            {"iteration": h.iteration, "val_nmse": h.val_nmse, "alpha": h.alpha, "allocation": h.allocation, "shots_spent_this_iteration": h.shots_spent_this_iteration}
            for h in result.history
        ],
        "best_allocation": result.best_allocation,
        "test_nmse_phase2_joint": test_nmse_joint,
        "test_nmse_phase1_variance_no_loop": test_nmse_phase1,
        "gain": gain,
        "elapsed_seconds": time.time() - t0,
    }
    with open("results/raw/phase2_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("[run_phase2] resumo salvo em results/raw/phase2_summary.json")


if __name__ == "__main__":
    main(sys.argv[1:])
