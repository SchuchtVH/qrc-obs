"""Carregamento de config, derivação determinística de seeds e hashing para cache.

Convenção de aleatoriedade do projeto: existe um único ``seeds.master`` na config.
Toda semente usada em qualquer lugar do código é derivada dele via
``numpy.random.SeedSequence(master).spawn(n)``, indexada por posição. Isso dá
sementes filhas estatisticamente independentes e, por serem determinadas
unicamente por ``master`` e pela posição, reprodutíveis bit-a-bit entre rodadas.
Nenhuma função usa ``np.random.seed`` global; toda função aleatória recebe um
``numpy.random.Generator`` explícito por argumento.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Índices fixos dos "slots" de seed derivados do master. A ordem nunca muda;
# novos usos de aleatoriedade recebem um novo índice ao final da lista.
SEED_SLOTS = [
    "signal",
    "reservoir_hamiltonian",
    "calibration_holdout_split",
    "calibration_empirical_repeats",
    "experiment_seeds_base",
    "baseline_random_dirichlet",
    "initial_feature_collection",
    "phase2_joint",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Aplica overrides tipo CLI ``a.b.c=valor`` sobre uma cópia da config."""
    import copy

    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        key, _, raw_value = ov.partition("=")
        if not _:
            raise ValueError(f"Override mal formado (esperado a.b.c=valor): {ov!r}")
        value = yaml.safe_load(raw_value)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return cfg


def derive_seeds(cfg: dict[str, Any]) -> dict[str, np.random.SeedSequence]:
    """Deriva uma SeedSequence filha por slot nomeado a partir de seeds.master."""
    master = cfg["seeds"]["master"]
    root = np.random.SeedSequence(master)
    children = root.spawn(len(SEED_SLOTS))
    return dict(zip(SEED_SLOTS, children))


def rng_for(seeds: dict[str, np.random.SeedSequence], slot: str, index: int | None = None) -> np.random.Generator:
    """Constrói um Generator para um slot; se index for dado, deriva mais uma
    camada (ex.: um rng por seed de experimento ou por janela de calibração)."""
    seq = seeds[slot]
    if index is None:
        return np.random.default_rng(seq)
    child = seq.spawn(index + 1)[index]
    return np.random.default_rng(child)


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list | tuple):
        return [_canonicalize(v) for v in obj]
    return obj


def config_hash(cfg: dict[str, Any], keys: list[str] | None = None, length: int = 16) -> str:
    """Hash estável (sha256, truncado) de um subconjunto canonicalizado da config.

    ``keys`` são caminhos ``a.b.c`` dentro de ``cfg``; se None, usa a config inteira.
    Usado para nomear artefatos de cache (ex. rho por janela) que só precisam ser
    recalculados quando os campos relevantes mudam.
    """
    if keys is None:
        subset = cfg
    else:
        subset = {}
        for key in keys:
            parts = key.split(".")
            src = cfg
            for p in parts:
                src = src[p]
            node = subset
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = src
    payload = json.dumps(_canonicalize(subset), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]


# Campos da config que determinam univocamente as janelas de entrada e os rho_t
# resultantes (Seção 3: "o reservatório é idêntico entre estratégias, sempre").
RESERVOIR_CACHE_KEYS = [
    "seeds.master",
    "signal.T",
    "signal.u_low",
    "signal.u_high",
    "task",
    "split.train_frac",
    "split.val_frac",
    "split.test_frac",
    "reservoir.n_qubits",
    "reservoir.window_L",
    "reservoir.J",
    "reservoir.h_low",
    "reservoir.h_high",
    "reservoir.tau",
]


def reservoir_cache_hash(cfg: dict[str, Any]) -> str:
    return config_hash(cfg, RESERVOIR_CACHE_KEYS)
