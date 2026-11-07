"""Pure-Python helpers for the closed-loop DVFS study (no Isaac Sim / torch)."""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from simulator.evaluation.metrics import summarize


def rotated_order(configs: list[Any], episode: int) -> list[Any]:
    """Cyclic Latin-square order: each config occupies each position equally often.

    Rotating per episode spreads thermal drift and simulator warm-up evenly across
    configurations, while every configuration sees the same initial state.
    """
    if not configs:
        return []
    k = episode % len(configs)
    return list(configs[k:]) + list(configs[:k])


def episode_noise_seed(seed: int, task_index: int, episode: int) -> int:
    """Per-(task, episode) seed shared by all DVFS configs of that episode."""
    return int(seed) * 10_000 + int(task_index) * 1_000 + int(episode)


def _mean_std(values: Iterable[Any]) -> tuple[float | None, float | None, int]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a binomial proportion, in percent."""
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize_closed_loop(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Task metrics (SR, CT, collision) plus AJJ, energy, latency and pacing."""
    out = summarize(rows)
    n, k = out['episodes'], out['successes']
    out['task_success_rate_ci95_low_pct'], out['task_success_rate_ci95_high_pct'] = wilson_ci(k, n)
    collisions = sum(bool(r['collision']) for r in rows)
    out['collision_rate_ci95_low_pct'], out['collision_rate_ci95_high_pct'] = wilson_ci(collisions, n)
    for key, name in [('ajj', 'ajj'), ('energy_j', 'energy_j'), ('mean_power_w', 'mean_power_w'),
                      ('inference_energy_j', 'inference_energy_j'), ('mean_inference_ms', 'inference_ms'),
                      ('mean_action_delay_ticks', 'action_delay_ticks'), ('holding_duration_s', 'holding_s'),
                      ('dropped_action_slots', 'dropped_action_slots'), ('real_time_factor', 'real_time_factor')]:
        mean, std, count = _mean_std(r.get(key) for r in rows)
        out[f'{name}_mean'] = mean
        out[f'{name}_std'] = std
        if name == 'energy_j':
            out['energy_n'] = count
    total_energy = [r.get('energy_j') for r in rows]
    if k and all(e is not None for e in total_energy):
        out['energy_per_success_j'] = sum(total_energy) / k
    else:
        out['energy_per_success_j'] = None
    out['frequency_valid_all'] = all(r.get('dvfs_frequency_valid', True) for r in rows)
    out['min_real_time_factor'] = min((r['real_time_factor'] for r in rows
                                       if r.get('real_time_factor') is not None), default=None)
    return out


def group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary per (dvfs_config, task), plus an 'all' task row per config."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        cfg = r.get('dvfs_config') or 'default'
        groups.setdefault((cfg, r['task']), []).append(r)
        groups.setdefault((cfg, 'all'), []).append(r)
    order = list(dict.fromkeys(r.get('dvfs_config') or 'default' for r in rows))
    tasks = list(dict.fromkeys(r['task'] for r in rows))
    if len(tasks) > 1:
        tasks.append('all')
    return [dict(dvfs_config=cfg, task=t, **summarize_closed_loop(groups[(cfg, t)]))
            for cfg in order for t in tasks if (cfg, t) in groups]


def paired_success_tests(rows: list[dict[str, Any]], reference: str) -> list[dict[str, Any]]:
    """Exact McNemar test of each config's success against ``reference`` on matched episodes."""
    by_key: dict[tuple[str, int], dict[str, bool]] = {}
    for r in rows:
        by_key.setdefault((r['task'], r.get('seed'), int(r['episode'])), {})[r.get('dvfs_config') or 'default'] = bool(r['success'])
    configs = list(dict.fromkeys(r.get('dvfs_config') or 'default' for r in rows))
    out = []
    for cfg in configs:
        if cfg == reference:
            continue
        pairs = [v for v in by_key.values() if reference in v and cfg in v]
        b = sum(v[reference] and not v[cfg] for v in pairs)
        c = sum(v[cfg] and not v[reference] for v in pairs)
        out.append(dict(config=cfg, reference=reference, matched_episodes=len(pairs),
                        ref_only_success=b, config_only_success=c, mcnemar_p=mcnemar_exact(b, c)))
    return out
