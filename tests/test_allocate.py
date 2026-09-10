import numpy as np
import pytest

from qrcshots.allocate import (
    allocate_oracle,
    allocate_random,
    allocate_uniform,
    largest_remainder_round,
    proportional_allocation_with_floor,
    s_g_empirical,
    s_g_exact,
    s_g_single_pass_empirical,
    simplex_grid,
)
from qrcshots.measure import GROUPS, exact_group_covariance, sample_group
from qrcshots.readout import FrozenReadout, StandardScalerStats
from qrcshots.reservoir import build_hamiltonian, build_propagator, evolve_window

N_QUBITS = 4


def test_largest_remainder_sums_exactly():
    rng = np.random.default_rng(0)
    for _ in range(20):
        budget = int(rng.integers(3, 500))
        scores = {"X": rng.random(), "Y": rng.random(), "Z": rng.random()}
        alloc = largest_remainder_round(scores, budget)
        assert sum(alloc.values()) == budget
        assert all(v >= 0 for v in alloc.values())


def test_proportional_allocation_respects_floor_and_sums_to_budget():
    scores = {"X": 10.0, "Y": 1.0, "Z": 1.0}
    alloc = proportional_allocation_with_floor(scores, budget=96, n_min=8)
    assert sum(alloc.values()) == 96
    assert all(v >= 8 for v in alloc.values())
    assert alloc["X"] > alloc["Y"]


def test_proportional_allocation_invariant_to_rescale():
    scores1 = {"X": 3.0, "Y": 1.0, "Z": 6.0}
    scores2 = {k: v * 100.0 for k, v in scores1.items()}
    alloc1 = proportional_allocation_with_floor(scores1, budget=600, n_min=8)
    alloc2 = proportional_allocation_with_floor(scores2, budget=600, n_min=8)
    assert alloc1 == alloc2


def test_proportional_allocation_raises_when_floor_does_not_fit():
    with pytest.raises(ValueError):
        proportional_allocation_with_floor({"X": 1.0, "Y": 1.0, "Z": 1.0}, budget=20, n_min=8)


def test_uniform_allocation_is_balanced():
    alloc = allocate_uniform(GROUPS, budget=300, n_min=8)
    values = list(alloc.values())
    assert max(values) - min(values) <= 1
    assert sum(values) == 300


def test_random_allocation_sums_to_budget_and_respects_floor():
    rng = np.random.default_rng(1)
    for _ in range(10):
        alloc = allocate_random(GROUPS, budget=150, n_min=8, rng=rng)
        assert sum(alloc.values()) == 150
        assert all(v >= 8 for v in alloc.values())


def test_simplex_grid_all_allocations_sum_to_budget_and_respect_floor():
    grid = simplex_grid(GROUPS, resolution=11, n_min=8, budget=300)
    assert len(grid) > 10
    for alloc in grid:
        assert sum(alloc.values()) == 300
        assert all(v >= 8 for v in alloc.values())


def test_allocate_oracle_finds_grid_minimum():
    def eval_fn(alloc):
        # mínimo artificial em X grande, Y/Z pequenos
        return (alloc["X"] - 200) ** 2 + (alloc["Y"] - 8) ** 2 + (alloc["Z"] - 8) ** 2

    best_alloc, best_score = allocate_oracle(GROUPS, budget=216, n_min=8, resolution=21, eval_fn=eval_fn)
    assert sum(best_alloc.values()) == 216
    assert best_alloc["X"] > best_alloc["Y"]
    assert best_alloc["X"] > best_alloc["Z"]


def _fixed_readout_and_rho(seed):
    rng = np.random.default_rng(seed)
    ham = build_hamiltonian(N_QUBITS, J=1.0, h_low=0.5, h_high=1.5, rng=rng)
    U = build_propagator(ham.H, tau=1.0)
    rho = evolve_window(rng.uniform(-1, 1, size=20), U, N_QUBITS)
    weights = rng.normal(size=3 * N_QUBITS)
    # escala não-trivial para exercitar a conversão w_std -> w_raw
    scale = rng.uniform(0.5, 2.0, size=3 * N_QUBITS)
    scaler = StandardScalerStats(mean=np.zeros(3 * N_QUBITS), scale=scale)
    readout = FrozenReadout(weights=weights, intercept=0.0, scaler=scaler, alpha=1.0, n_qubits=N_QUBITS)
    return readout, rho


def test_s_g_empirical_matches_s_g_exact():
    readout, rho = _fixed_readout_and_rho(seed=5)
    rho_batch = np.stack([rho])
    rng = np.random.default_rng(123)
    for group in GROUPS:
        exact = s_g_exact(readout, group, rho_batch)
        empirical = s_g_empirical(readout, group, rho_batch, n_cal_shots=64, n_repeats=3000, rng=rng)
        assert np.isclose(exact, empirical, rtol=0.1), f"grupo {group}: exact={exact}, empirical={empirical}"


def test_s_g_single_pass_empirical_converges_to_exact_with_many_calibration_windows():
    """O estimador de passe único usa 1 lote de n_cal_shots por janela (sem repetir);
    precisa de várias janelas de calibração para ter poder estatístico equivalente ao
    método de R repetições -- aqui usamos ~400 janelas do MESMO reservatório (H, U
    fixos), variando só a sequência de entrada injetada, como na Seção 6.2."""
    readout, _ = _fixed_readout_and_rho(seed=5)
    rng_windows = np.random.default_rng(21)
    ham = build_hamiltonian(N_QUBITS, J=1.0, h_low=0.5, h_high=1.5, rng=rng_windows)
    U = build_propagator(ham.H, tau=1.0)
    rho_batch = np.stack([evolve_window(rng_windows.uniform(-1, 1, size=20), U, N_QUBITS) for _ in range(400)])

    rng = np.random.default_rng(321)
    for group in GROUPS:
        exact = s_g_exact(readout, group, rho_batch)
        single_pass = s_g_single_pass_empirical(readout, group, rho_batch, n_cal_shots=64, rng=rng)
        assert np.isclose(exact, single_pass, rtol=0.1), f"grupo {group}: exact={exact}, single_pass={single_pass}"


def test_allocation_formula_minimizes_empirical_prediction_variance():
    """Teste central da Seção 7: sweep no simplex de alocações, variância REAL
    (Monte Carlo, via sample_group de verdade) de sum_g c_g por alocação, e checa que
    o mínimo empírico cai perto de onde a fórmula n_g = B*s_g/sum(s_h) prevê."""
    readout, rho = _fixed_readout_and_rho(seed=9)
    rho_batch = np.stack([rho])
    s_g = {g: s_g_exact(readout, g, rho_batch) for g in GROUPS}

    budget = 300
    n_min = 8
    grid = simplex_grid(GROUPS, resolution=13, n_min=n_min, budget=budget)

    from qrcshots.allocate import group_weight_raw_scale

    w = {g: group_weight_raw_scale(readout, g) for g in GROUPS}
    rng = np.random.default_rng(2024)
    n_reps = 1500

    def empirical_variance(alloc):
        totals = np.empty(n_reps)
        for r in range(n_reps):
            total = 0.0
            for g in GROUPS:
                means = sample_group(rho, g, alloc[g], rng, N_QUBITS).means
                total += w[g] @ means
            totals[r] = total
        return totals.var(ddof=1)

    scored = [(alloc, empirical_variance(alloc)) for alloc in grid]
    best_alloc, best_var = min(scored, key=lambda x: x[1])

    formula_alloc = proportional_allocation_with_floor(s_g, budget, n_min)
    # variância analítica exata (sem MC) sob a alocação prevista pela fórmula
    analytic_formula_var = sum(s_g[g] ** 2 / formula_alloc[g] for g in GROUPS)

    # a variância analítica da alocação da fórmula deve estar entre as menores do grid
    # (checagem determinística, sem ruído de MC: valida a fórmula em si)
    all_analytic = sorted(sum(s_g[g] ** 2 / a[g] for g in GROUPS) for a in grid)
    assert analytic_formula_var <= all_analytic[min(2, len(all_analytic) - 1)] + 1e-9, (
        f"alocação da fórmula não está entre as melhores analiticamente: "
        f"formula_var={analytic_formula_var}, top3={all_analytic[:3]}"
    )

    # Checagem com ruído de Monte Carlo real (sample_group de verdade): perto do
    # ótimo a variância é localmente plana (mínimo de uma função convexa em 1/n_g),
    # então váaias alocações próximas do ótimo têm variância quase idêntica -- exigir
    # que o argmin do grid EM SHOTS coincida exatamente com a fórmula é frágil demais.
    # Em vez disso, confirmamos que a variância empírica sob a alocação da fórmula é
    # ela mesma próxima do mínimo empírico encontrado no grid (dentro do erro de MC:
    # com n_reps=1500, erro relativo de uma estimativa de variância ~ sqrt(2/n_reps) ~ 3.6%).
    emp_formula_var = empirical_variance(formula_alloc)
    assert emp_formula_var <= best_var * 1.25, (
        f"variância empírica da alocação da fórmula ({emp_formula_var:.4g}) muito acima do "
        f"mínimo empírico do grid ({best_var:.4g}); formula_alloc={formula_alloc}, best_alloc={best_alloc}"
    )
