"""Incerteza da contribuição de grupo e alocação de shots (Seções 6.2 e 7).

Nota de escala (não explicitada de forma óbvia no enunciado, resolvida aqui): os pesos
congelados em `FrozenReadout.weights` atuam no espaço PADRONIZADO das features
(`x_std = (x_raw - mean)/scale`), mas `Sigma_g` (measure.exact_group_covariance) é a
covariância das estimativas <P_i>_hat no espaço RAW (valores em [-1,1]). Combinar
`w_std` diretamente com `Sigma_g` raw daria unidades erradas. A contribuição do grupo
e sua variância devem ser calculadas no mesmo espaço; usamos o espaço raw, com o peso
"efetivo" `w_g = w_std_g / scale_g` (pois y = w_std . x_std + b = (w_std/scale) . x_raw
+ const). É isso que o enunciado quer dizer com "w_g já na escala padronizada
correta" -- correta para combinar com Sigma_g raw.
"""

from __future__ import annotations

import numpy as np

from qrcshots.measure import GROUPS, exact_group_covariance
from qrcshots.readout import FrozenReadout


def group_weight_raw_scale(readout: FrozenReadout, group: str) -> np.ndarray:
    """w_g na escala das features raw: w_std_g / scale_g."""
    n_qubits = readout.n_qubits
    gi = GROUPS.index(group)
    w_std_g = readout.group_weights(group)
    scale_g = readout.scaler.scale[gi * n_qubits : (gi + 1) * n_qubits]
    return w_std_g / scale_g


def s_g_exact(readout: FrozenReadout, group: str, rho_calib_batch: np.ndarray) -> float:
    """s_g = sqrt(E_janelas[w_g^T Sigma_g w_g]), Sigma_g exato de cada rho de calibração."""
    n_qubits = readout.n_qubits
    w = group_weight_raw_scale(readout, group)
    quad_forms = np.array([w @ exact_group_covariance(rho, group, n_qubits) @ w for rho in rho_calib_batch])
    return float(np.sqrt(quad_forms.mean()))


def s_g_empirical(readout: FrozenReadout, group: str, rho_calib_batch: np.ndarray, n_cal_shots: int, n_repeats: int, rng: np.random.Generator) -> float:
    """Repete a medição de cada janela de calibração `n_repeats` vezes com `n_cal_shots`
    shots; desvio-padrão de c_g entre repetições, escalado por sqrt(n_cal_shots), média
    sobre janelas.

    Vetorizado: `rng.multinomial(n_cal_shots, p, size=n_repeats)` desenha todas as
    repetições de uma janela numa única chamada (o experimento completo tem milhares
    de janelas de calibração; um laço Python por repetição não escala).
    """
    from qrcshots.measure import bit_signs, diag_probs_after_rotation

    n_qubits = readout.n_qubits
    w = group_weight_raw_scale(readout, group)
    signs = bit_signs(n_qubits)
    per_window = np.empty(len(rho_calib_batch))
    for k, rho in enumerate(rho_calib_batch):
        p = diag_probs_after_rotation(rho, group, n_qubits)
        counts = rng.multinomial(n_cal_shots, p, size=n_repeats)  # (n_repeats, 2**n_qubits)
        means = (counts @ signs) / n_cal_shots  # (n_repeats, n_qubits)
        c_samples = means @ w  # (n_repeats,)
        per_window[k] = c_samples.std(ddof=1) * np.sqrt(n_cal_shots)
    return float(per_window.mean())


def s_g_single_pass_empirical(readout: FrozenReadout, group: str, rho_calib_batch: np.ndarray, n_cal_shots: int, rng: np.random.Generator) -> float:
    """Estimativa de s_g a partir de UM ÚNICO lote de `n_cal_shots` shots por janela
    (não repetido) -- o custo de calibração realmente implantável.

    Distinção do `s_g_empirical` (Seção 6.2): aquele repete a MESMA janela `n_repeats`
    vezes (ex. R=200) para servir de checagem de sanidade contra `s_g_exact` -- é um
    protocolo de VALIDAÇÃO, não algo que se pagaria em um deployment real (custaria
    R vezes mais shots). Aqui, em vez de repetir a medição, usamos os `n_cal_shots`
    shots INDIVIDUAIS de um único lote como amostra: cada shot i dá um valor de
    contribuição c_i = w_g . sinais(bitstring_i), e Sigma_g é por definição a
    covariância de UM shot, então Var(c_i sobre os shots) já é um estimador não-viesado
    de w_g^T Sigma_g w_g -- sem precisar repetir a janela. Isso estima a mesma
    quantidade que `s_g_empirical`, com uma fração (1/n_repeats) do custo em shots.
    """
    from qrcshots.measure import bit_signs, diag_probs_after_rotation

    n_qubits = readout.n_qubits
    w = group_weight_raw_scale(readout, group)
    signs = bit_signs(n_qubits)
    per_shot_contribution = signs @ w  # (2**n_qubits,): valor de c para cada bitstring possível

    per_window = np.empty(len(rho_calib_batch))
    for k, rho in enumerate(rho_calib_batch):
        p = diag_probs_after_rotation(rho, group, n_qubits)
        shot_indices = rng.choice(2**n_qubits, size=n_cal_shots, p=p)
        c_per_shot = per_shot_contribution[shot_indices]
        per_window[k] = c_per_shot.std(ddof=1)
    return float(per_window.mean())


def compute_all_s_g(readout: FrozenReadout, rho_calib_batch: np.ndarray, method: str = "exact", n_cal_shots: int | None = None, n_repeats: int | None = None, rng: np.random.Generator | None = None) -> dict[str, float]:
    if method == "exact":
        return {g: s_g_exact(readout, g, rho_calib_batch) for g in GROUPS}
    if method == "empirical":
        assert n_cal_shots is not None and n_repeats is not None and rng is not None
        return {g: s_g_empirical(readout, g, rho_calib_batch, n_cal_shots, n_repeats, rng) for g in GROUPS}
    raise ValueError(f"método desconhecido: {method!r}")


# --- Alocação: de scores por grupo para n_g inteiros que somam exatamente B --------
#
# Derivação (Seção 7): para custo igual por shot entre grupos, minimizar
#   Var(sum_g c_g) = sum_g (w_g^T Sigma_g w_g) / n_g = sum_g s_g^2 / n_g
# sujeito a sum_g n_g = B. Lagrangiano L = sum_g s_g^2/n_g + lambda*(sum_g n_g - B):
#   dL/dn_g = -s_g^2/n_g^2 + lambda = 0  =>  n_g = s_g / sqrt(lambda)
# Substituindo na restrição: sum_g s_g/sqrt(lambda) = B  =>  sqrt(lambda) = sum_g s_g / B
#   =>  n_g = B * s_g / sum_h s_h


def largest_remainder_round(proportions: dict[str, float], total: int) -> dict[str, int]:
    """Arredondamento pelo método dos maiores restos: soma exatamente `total`."""
    keys = list(proportions.keys())
    raw = np.array([proportions[k] for k in keys], dtype=float)
    if raw.sum() <= 0:
        raw = np.ones_like(raw)
    scaled = raw / raw.sum() * total
    floor_vals = np.floor(scaled).astype(int)
    remainder = int(total - floor_vals.sum())
    fractional = scaled - floor_vals
    order = np.argsort(-fractional, kind="stable")
    alloc = floor_vals.copy()
    for idx in order[:remainder]:
        alloc[idx] += 1
    return dict(zip(keys, alloc.tolist()))


def proportional_allocation_with_floor(scores: dict[str, float], budget: int, n_min: int) -> dict[str, int]:
    """n_g = n_min + arredondamento-por-maior-resto do restante, proporcional a `scores`.

    Soma exatamente `budget`, respeita o piso `n_min` por grupo, e é invariante a
    reescala de `scores` (só as razões entre scores importam, via largest_remainder_round).
    """
    groups = list(scores.keys())
    n_groups = len(groups)
    if n_min * n_groups > budget:
        raise ValueError(f"n_min={n_min} * {n_groups} grupos > budget={budget}: piso não cabe.")
    remaining = budget - n_min * n_groups
    extra = largest_remainder_round(scores, remaining)
    return {g: n_min + extra[g] for g in groups}


def allocate_uniform(groups: tuple[str, ...], budget: int, n_min: int) -> dict[str, int]:
    return proportional_allocation_with_floor({g: 1.0 for g in groups}, budget, n_min)


def allocate_variance(s_g: dict[str, float], budget: int, n_min: int) -> dict[str, int]:
    return proportional_allocation_with_floor(s_g, budget, n_min)


def allocate_shap(importance_g: dict[str, float], budget: int, n_min: int) -> dict[str, int]:
    return proportional_allocation_with_floor(importance_g, budget, n_min)


def allocate_magnitude(readout: FrozenReadout, budget: int, n_min: int) -> dict[str, int]:
    scores = {g: float(np.sum(np.abs(readout.group_weights(g)))) for g in GROUPS}
    return proportional_allocation_with_floor(scores, budget, n_min)


def allocate_random(groups: tuple[str, ...], budget: int, n_min: int, rng: np.random.Generator) -> dict[str, int]:
    """Alocação sorteada do simplex via Dirichlet(1,1,1) -- controle contra "qualquer
    desbalanceamento ajuda"."""
    draw = rng.dirichlet(np.ones(len(groups)))
    scores = dict(zip(groups, draw.tolist()))
    return proportional_allocation_with_floor(scores, budget, n_min)


def simplex_grid(groups: tuple[str, ...], resolution: int, n_min: int, budget: int) -> list[dict[str, int]]:
    """Todas as alocações inteiras que somam `budget`, respeitam `n_min`, e caem
    numa grade de `resolution` pontos por eixo no simplex de proporções."""
    n_groups = len(groups)
    remaining = budget - n_min * n_groups
    if remaining < 0:
        raise ValueError("n_min não cabe no budget.")
    seen = set()
    allocs = []
    grid_points = np.linspace(0.0, 1.0, resolution)
    if n_groups != 3:
        raise NotImplementedError("simplex_grid só implementado para 3 grupos.")
    for a in grid_points:
        for b in grid_points:
            if a + b > 1.0 + 1e-9:
                continue
            c = 1.0 - a - b
            props = {groups[0]: a, groups[1]: b, groups[2]: max(c, 0.0)}
            extra = largest_remainder_round(props, remaining)
            alloc = {g: n_min + extra[g] for g in groups}
            key = tuple(alloc[g] for g in groups)
            if key not in seen:
                seen.add(key)
                allocs.append(alloc)
    return allocs


def allocate_oracle(groups: tuple[str, ...], budget: int, n_min: int, resolution: int, eval_fn) -> tuple[dict[str, int], float]:
    """Busca em grade fina no simplex minimizando `eval_fn(alloc) -> erro`. Limite
    superior de desempenho, não-implementável na prática (usa o split de teste)."""
    best_alloc = None
    best_score = np.inf
    for alloc in simplex_grid(groups, resolution, n_min, budget):
        score = eval_fn(alloc)
        if score < best_score:
            best_score = score
            best_alloc = alloc
    return best_alloc, best_score
