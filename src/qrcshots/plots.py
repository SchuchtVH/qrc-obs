"""Estatísticas de comparação e figuras (Seções 8-9)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

GROUPS_ORDER = ("X", "Y", "Z")


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))


def summarize_nmse(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Média, desvio-padrão e IC bootstrap 95% do NMSE por (orçamento, estratégia)."""
    rows = []
    for (budget, strategy), sub in df.groupby(["budget", "strategy"]):
        vals = sub["nmse"].to_numpy()
        lo, hi = bootstrap_ci(vals, rng)
        rows.append(
            {
                "budget": budget,
                "strategy": strategy,
                "mean_nmse": vals.mean(),
                "std_nmse": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_seeds": len(vals),
            }
        )
    return pd.DataFrame(rows).sort_values(["budget", "strategy"]).reset_index(drop=True)


def wilcoxon_shap_vs_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Teste pareado (Wilcoxon signed-rank sobre as seeds), shap vs variance, por
    orçamento, com tamanho de efeito (correlação rank-biserial pareada)."""
    from scipy.stats import wilcoxon

    rows = []
    for budget, sub in df.groupby("budget"):
        shap_v = sub[sub.strategy == "shap"].sort_values("seed")["nmse"].to_numpy()
        var_v = sub[sub.strategy == "variance"].sort_values("seed")["nmse"].to_numpy()
        assert len(shap_v) == len(var_v), "shap e variance precisam ter o mesmo número de seeds"
        diff = shap_v - var_v
        nonzero = diff[diff != 0]
        if len(nonzero) == 0:
            stat, p, effect = np.nan, 1.0, 0.0
        else:
            stat, p = wilcoxon(shap_v, var_v)
            ranks = stats.rankdata(np.abs(nonzero))
            w_pos = ranks[nonzero > 0].sum()
            w_neg = ranks[nonzero < 0].sum()
            effect = float((w_pos - w_neg) / (w_pos + w_neg))
        rows.append(
            {
                "budget": budget,
                "wilcoxon_stat": stat,
                "p_value": p,
                "effect_size_rank_biserial": effect,
                "median_diff_shap_minus_variance": float(np.median(diff)),
                "n_seeds": len(diff),
            }
        )
    return pd.DataFrame(rows)


def add_amortized_shots(df: pd.DataFrame, cost_meta: dict) -> pd.DataFrame:
    """Custo total amortizado (Seção 9): (coleta_inicial + calibração_da_estratégia +
    N_test*B) / N_test = coleta_inicial/N_test + calibração/N_test + B."""
    n_test = cost_meta["n_test_windows"]
    shared = cost_meta["shots_coleta_inicial"] / n_test
    calib = cost_meta["shots_calibracao_por_estrategia"]
    df = df.copy()
    df["amortized_shots"] = shared + df["strategy"].map(lambda s: calib.get(s, 0.0)) / n_test + df["budget"]
    return df


def compute_breakeven_predictions(summary_df: pd.DataFrame, cost_meta: dict, strategy_a: str = "variance", strategy_b: str = "uniform") -> pd.DataFrame:
    """N* de previsões para que `strategy_a` compense seu custo extra de calibração
    frente a `strategy_b` (Seção 9): interpola a curva NMSE(budget) de `strategy_b`
    para achar, a cada NMSE atingido por `strategy_a` num orçamento B, o orçamento
    equivalente que `strategy_b` precisaria para igualar essa NMSE. A diferença de
    shots é a economia marginal por previsão; N* = custo_extra_calibração / economia.
    Se `strategy_a` não é melhor que `strategy_b` nesse ponto, não há economia e o
    breakeven é infinito (reportado como tal, não escondido).
    """
    a = summary_df[summary_df.strategy == strategy_a].sort_values("budget")
    b = summary_df[summary_df.strategy == strategy_b].sort_values("budget")
    calib = cost_meta["shots_calibracao_por_estrategia"]
    extra_calib_cost = calib.get(strategy_a, 0.0) - calib.get(strategy_b, 0.0)

    b_sorted = b.sort_values("mean_nmse")
    rows = []
    for _, row in a.iterrows():
        target_nmse = row["mean_nmse"]
        if target_nmse < b_sorted["mean_nmse"].min() or target_nmse > b_sorted["mean_nmse"].max():
            b_equiv = float("nan")
        else:
            b_equiv = float(np.interp(target_nmse, b_sorted["mean_nmse"], b_sorted["budget"]))
        savings = b_equiv - row["budget"] if not np.isnan(b_equiv) else float("nan")
        if extra_calib_cost <= 0:
            n_star = 0.0
        elif np.isnan(savings) or savings <= 0:
            n_star = float("inf")
        else:
            n_star = extra_calib_cost / savings
        rows.append(
            {
                "budget": row["budget"],
                "nmse": target_nmse,
                f"{strategy_b}_equivalent_budget": b_equiv,
                "marginal_savings_per_prediction": savings,
                "extra_calibration_cost": extra_calib_cost,
                "breakeven_n_predictions": n_star,
            }
        )
    return pd.DataFrame(rows)


def nmse_decomposition(summary_df: pd.DataFrame, floor_nmse: float) -> pd.DataFrame:
    """Verifica NMSE_total ~= NMSE_piso + contribuição de shot noise, por (orçamento,
    estratégia), e reporta a fração atribuível a cada parte."""
    out = summary_df.copy()
    out["floor_nmse"] = floor_nmse
    out["shot_noise_contribution"] = out["mean_nmse"] - floor_nmse
    out["shot_noise_fraction"] = out["shot_noise_contribution"] / out["mean_nmse"]
    out["floor_fraction"] = out["floor_nmse"] / out["mean_nmse"]
    return out


def plot_error_vs_shots(summary_df: pd.DataFrame, out_path: str, x_col: str = "budget", title: str = "", floor_nmse: float | None = None, x_label: str = "shots (orçamento por previsão)") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, sub in summary_df.groupby("strategy"):
        sub = sub.sort_values(x_col)
        yerr_lo = sub["mean_nmse"] - sub["ci_lo"]
        yerr_hi = sub["ci_hi"] - sub["mean_nmse"]
        ax.errorbar(sub[x_col], sub["mean_nmse"], yerr=[yerr_lo, yerr_hi], marker="o", capsize=3, label=strategy)
    if floor_nmse is not None:
        ax.axhline(floor_nmse, color="black", linestyle="--", linewidth=1, label="piso do reservatório (shots→∞)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("NMSE (teste)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _simplex_xy(px: float, py: float, pz: float) -> tuple[float, float]:
    x = py + pz / 2.0
    y = pz * np.sqrt(3) / 2.0
    return x, y


def plot_allocation_simplex(allocations_per_budget: dict[int, dict[str, dict[str, int]]], out_path: str, budgets_to_show: list[int] | None = None) -> None:
    budgets_to_show = budgets_to_show or sorted(allocations_per_budget.keys())
    n = len(budgets_to_show)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3) / 2, 0]

    for i, budget in enumerate(budgets_to_show):
        ax = axes[i // ncols][i % ncols]
        ax.plot(triangle_x, triangle_y, color="black", linewidth=1)
        ax.text(-0.02, -0.03, "X", ha="right")
        ax.text(1.02, -0.03, "Y", ha="left")
        ax.text(0.5, np.sqrt(3) / 2 + 0.02, "Z", ha="center")
        for strategy, alloc in allocations_per_budget[budget].items():
            total = sum(alloc.values())
            x, y = _simplex_xy(alloc["X"] / total, alloc["Y"] / total, alloc["Z"] / total)
            ax.scatter([x], [y], label=strategy, s=60)
        ax.set_title(f"B={budget}")
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.1, 1.0)
        ax.axis("off")
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(-0.3, 1.1))

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ranking_divergence(shap_importance: dict[str, float], s_g: dict[str, float], out_path: str) -> None:
    groups = list(GROUPS_ORDER)
    shap_vals = np.array([shap_importance[g] for g in groups])
    s_vals = np.array([s_g[g] for g in groups])
    shap_rank = stats.rankdata(-shap_vals)
    s_rank = stats.rankdata(-s_vals)
    tau = stats.kendalltau(shap_rank, s_rank).correlation

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(groups))

    axes[0].bar(x - 0.2, shap_vals / shap_vals.sum(), width=0.4, label="importância SHAP (normalizada)")
    axes[0].bar(x + 0.2, s_vals / s_vals.sum(), width=0.4, label="s_g (normalizado)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Variabilidade de sinal (SHAP) vs incerteza de medição (s_g)")

    axes[1].bar(x - 0.2, shap_rank, width=0.4, label="ranking SHAP")
    axes[1].bar(x + 0.2, s_rank, width=0.4, label="ranking s_g")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].set_ylabel("posição no ranking (1 = maior)")
    axes[1].legend(fontsize=8)
    tau_str = f"{tau:.2f}" if tau is not None and not np.isnan(tau) else "n/a"
    axes[1].set_title(f"divergência de ranking (Kendall tau={tau_str})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
