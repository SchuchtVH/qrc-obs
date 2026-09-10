#!/usr/bin/env python
"""Script fino: lê um resultado de fase 1 salvo e gera as figuras + tabelas derivadas.

Uso:
    python scripts/make_figures.py results/raw/phase1_<hash>.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qrcshots.plots import (
    add_amortized_shots,
    compute_breakeven_predictions,
    nmse_decomposition,
    plot_allocation_simplex,
    plot_error_vs_shots,
    plot_ranking_divergence,
    summarize_nmse,
    wilcoxon_shap_vs_variance,
)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_path")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)

    parquet_path = Path(args.parquet_path)
    meta_path = parquet_path.with_name(parquet_path.stem + "_meta.json")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    with open(meta_path) as f:
        meta = json.load(f)

    rng = np.random.default_rng(args.bootstrap_seed)
    summary = summarize_nmse(df, rng)

    floor_nmse = meta["floor_nmse"]
    plot_error_vs_shots(
        summary,
        figures_dir / "error_vs_shots_marginal.png",
        x_col="budget",
        title="NMSE x shots (custo marginal)",
        floor_nmse=floor_nmse,
        x_label="shots por previsão (B)",
    )
    print(f"[make_figures] salvo {figures_dir / 'error_vs_shots_marginal.png'}")

    cost_meta = meta["cost_accounting"]
    df_amortized = add_amortized_shots(df, cost_meta)
    # amortized_shots é 1-a-1 com (budget, strategy) -- junta ao resumo de NMSE (que
    # já tem média/CI por (budget, strategy)) e plota no eixo amortizado.
    amortized_summary = summarize_nmse(df, rng).copy()
    amortized_x = df_amortized.groupby(["budget", "strategy"])["amortized_shots"].mean().reset_index()
    amortized_summary = amortized_summary.merge(amortized_x, on=["budget", "strategy"])
    plot_error_vs_shots(
        amortized_summary,
        figures_dir / "error_vs_shots_amortized.png",
        x_col="amortized_shots",
        title="NMSE x shots (custo total amortizado, calibração inclusa)",
        floor_nmse=floor_nmse,
        x_label="shots amortizados por previsão",
    )
    print(f"[make_figures] salvo {figures_dir / 'error_vs_shots_amortized.png'}")

    allocations = meta["allocations_per_budget"]
    allocations = {int(k): v for k, v in allocations.items()}
    plot_allocation_simplex(allocations, figures_dir / "allocation_simplex.png")
    print(f"[make_figures] salvo {figures_dir / 'allocation_simplex.png'}")

    plot_ranking_divergence(meta["shap_importance"], meta["s_g_single_pass"], figures_dir / "ranking_divergence.png")
    print(f"[make_figures] salvo {figures_dir / 'ranking_divergence.png'}")

    wilcoxon_df = wilcoxon_shap_vs_variance(df)
    breakeven_df = compute_breakeven_predictions(summary, cost_meta, "variance", "uniform")
    decomposition_df = nmse_decomposition(summary, floor_nmse)

    tables_dir = figures_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    summary.to_csv(tables_dir / "nmse_summary.csv", index=False)
    wilcoxon_df.to_csv(tables_dir / "wilcoxon_shap_vs_variance.csv", index=False)
    breakeven_df.to_csv(tables_dir / "breakeven_variance_vs_uniform.csv", index=False)
    decomposition_df.to_csv(tables_dir / "nmse_decomposition.csv", index=False)
    print(f"[make_figures] tabelas salvas em {tables_dir}/")

    print("\n[make_figures] resumo NMSE:")
    print(summary.to_string(index=False))
    print("\n[make_figures] Wilcoxon shap vs variance:")
    print(wilcoxon_df.to_string(index=False))
    print("\n[make_figures] breakeven variance vs uniform (N* previsões):")
    print(breakeven_df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
