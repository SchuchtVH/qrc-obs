import numpy as np

from qrcshots.config import apply_overrides, derive_seeds, load_config
from qrcshots.experiment import build_pipeline_context
from qrcshots.joint import run_joint_loop
from qrcshots.readout import alpha_grid_from_cfg


def _small_ctx():
    cfg = load_config("configs/base.yaml")
    cfg = apply_overrides(cfg, ["signal.T=800"])
    seeds = derive_seeds(cfg)
    return build_pipeline_context(cfg), cfg


def test_joint_loop_runs_and_tracks_history():
    ctx, cfg = _small_ctx()
    alpha_grid = alpha_grid_from_cfg(cfg)
    result = run_joint_loop(ctx, budget_per_window=192, n_min=8, alpha_grid=alpha_grid, n_cal_shots=32, max_iterations=6, patience=2)

    assert len(result.history) >= 1
    assert result.total_shots_spent > 0
    assert sum(result.best_allocation.values()) == 192
    # o val_nmse do melhor iteração é o mínimo entre todas as registradas
    best_val_nmse = min(h.val_nmse for h in result.history)
    assert np.isclose(result.history[result.best_iteration].val_nmse, best_val_nmse)


def test_joint_loop_never_touches_test_split():
    """O loop só recebe rho_ridge_train/rho_val/rho_calib no contexto -- garantido
    estruturalmente, já que PipelineContext não expõe rho_test antes de
    unlock_test_data. Este teste apenas confirma que o dataset não foi congelado
    pelo loop (freeze só deve acontecer explicitamente, fora de run_joint_loop)."""
    ctx, cfg = _small_ctx()
    alpha_grid = alpha_grid_from_cfg(cfg)
    assert not ctx.dataset.is_frozen
    run_joint_loop(ctx, budget_per_window=96, n_min=8, alpha_grid=alpha_grid, n_cal_shots=32, max_iterations=3, patience=1)
    assert not ctx.dataset.is_frozen


def test_joint_loop_stops_within_patience_of_best():
    ctx, cfg = _small_ctx()
    alpha_grid = alpha_grid_from_cfg(cfg)
    patience = 2
    result = run_joint_loop(ctx, budget_per_window=192, n_min=8, alpha_grid=alpha_grid, n_cal_shots=32, max_iterations=10, patience=patience)
    if result.converged:
        assert len(result.history) - 1 - result.best_iteration == patience
