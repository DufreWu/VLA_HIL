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
    """Exact McNemar test of each config's success against ``reference`` on matched episodes.

    One test per task (effects can have opposite signs across tasks) plus a pooled 'all' test.
    """
    by_key: dict[tuple[str, Any, int], dict[str, bool]] = {}
    for r in rows:
        by_key.setdefault((r['task'], r.get('seed'), int(r['episode'])), {})[r.get('dvfs_config') or 'default'] = bool(r['success'])
    configs = list(dict.fromkeys(r.get('dvfs_config') or 'default' for r in rows))
    tasks = list(dict.fromkeys(r['task'] for r in rows))
    out = []
    for task in tasks + (['all'] if len(tasks) > 1 else []):
        for cfg in configs:
            if cfg == reference:
                continue
            pairs = [v for k, v in by_key.items() if (task == 'all' or k[0] == task)
                     and reference in v and cfg in v]
            b = sum(v[reference] and not v[cfg] for v in pairs)
            c = sum(v[cfg] and not v[reference] for v in pairs)
            out.append(dict(task=task, config=cfg, reference=reference, matched_episodes=len(pairs),
                            ref_only_success=b, config_only_success=c, mcnemar_p=mcnemar_exact(b, c)))
    return out


# ---------------------------------------------------------------- result tables

def _mean(rows, key, scale=1.0):
    m, _, n = _mean_std(r.get(key) for r in rows)
    return None if m is None else m * scale


def _requested(cfg, key):
    """Requested frequency; symbolic min/max are reported as resolved on the device."""
    value = cfg.get(f'resolved_{key}', cfg[key])
    return float('nan') if isinstance(value, str) else value


def dvfs_table(rows: list[dict[str, Any]], configs: list[dict[str, Any]], reference: str) -> list[dict[str, Any]]:
    """One row per (config, task): frequencies, task metrics, latency, power, paired p-value."""
    tests = {(t['task'], t['config']): t['mcnemar_p'] for t in paired_success_tests(rows, reference)}
    tasks = list(dict.fromkeys(r['task'] for r in rows))
    table = []
    for cfg in configs:
        name = cfg['name']
        governed = cfg.get('cpu_khz') is None
        for task in tasks:
            g = [r for r in rows if (r.get('dvfs_config') or 'default') == name and r['task'] == task]
            if not g:
                continue
            n, k = len(g), sum(bool(r['success']) for r in g)
            low, high = wilson_ci(k, n)
            ct = [r['completion_time_s'] for r in g if r['success'] and r.get('completion_time_s') is not None]
            energy = [r.get('energy_j') for r in g]
            table.append(dict(
                config=name,
                cpu_mode=cfg.get('cpu_governor', 'fixed') if governed else 'fixed',
                gpu_mode=cfg.get('gpu_governor', 'fixed') if governed else 'fixed',
                cpu_mhz=_mean(g, 'mean_cpu_khz', 1e-3) if governed else _requested(cfg, 'cpu_khz') / 1e3,
                gpu_mhz=_mean(g, 'mean_gpu_hz', 1e-6) if governed else _requested(cfg, 'gpu_hz') / 1e6,
                measured_cpu_mhz=_mean(g, 'mean_cpu_khz', 1e-3),
                measured_gpu_mhz=_mean(g, 'mean_gpu_hz', 1e-6),
                task=task, episodes=n, successes=k,
                success_rate_pct=100 * k / n, success_ci95_low_pct=low, success_ci95_high_pct=high,
                mcnemar_p_vs_reference=None if name == reference else tests.get((task, name)),
                task_progress=_mean(g, 'task_progress'),
                completion_time_s=statistics.mean(ct) if ct else None,
                collision_rate_pct=100 * sum(bool(r['collision']) for r in g) / n,
                tip_over_rate_pct=_mean(g, 'tip_over', 100.0) if task == 'pushing' else None,
                ajj_rad_s3=_mean(g, 'ajj'),
                inference_ms=_mean(g, 'mean_inference_ms'),
                action_delay_ticks=_mean(g, 'mean_action_delay_ticks'),
                holding_s=_mean(g, 'holding_duration_s'),
                mean_power_w=_mean(g, 'mean_power_w'),
                energy_per_success_j=(sum(energy) / k) if k and all(e is not None for e in energy) else None,
                frequency_valid_all=all(r.get('dvfs_frequency_valid', True) for r in g),
                min_real_time_factor=min((r['real_time_factor'] for r in g
                                          if r.get('real_time_factor') is not None), default=None),
            ))
    return table


# (header, key, format) for the human-readable tables.
_DISPLAY = [
    ('Config', 'config', 's'), ('CPU (MHz)', 'cpu_mhz', 'freq'), ('GPU (MHz)', 'gpu_mhz', 'freq'),
    ('Task', 'task', 's'), ('N', 'episodes', 'd'), ('SR (%) [95% CI]', 'success_rate_pct', 'sr'),
    ('p', 'mcnemar_p_vs_reference', 'p'), ('Progress', 'task_progress', '.2f'),
    ('CT (s)', 'completion_time_s', '.2f'), ('Coll. (%)', 'collision_rate_pct', '.0f'),
    ('Tip-over (%)', 'tip_over_rate_pct', '.0f'), ('AJJ', 'ajj_rad_s3', '.1f'),
    ('Inference (ms)', 'inference_ms', '.0f'), ('Delay (ticks)', 'action_delay_ticks', '.1f'),
    ('Power (W)', 'mean_power_w', '.2f'), ('J/success', 'energy_per_success_j', '.0f'),
]


def _cell(row, key, fmt, latex=False):
    v = row.get(key)
    if fmt == 's':
        return str(v).replace('_', r'\_') if latex else str(v)
    if fmt == 'freq':
        mode = row['cpu_mode' if key == 'cpu_mhz' else 'gpu_mode']
        if mode != 'fixed':
            return f'{mode} ({v:.0f})' if v is not None else mode
        return f'{v:.0f}'
    if v is None:
        return '--' if latex else '–'
    if fmt == 'sr':
        return f"{v:.0f} [{row['success_ci95_low_pct']:.0f}, {row['success_ci95_high_pct']:.0f}]"
    if fmt == 'p':
        return '<0.001' if v < 1e-3 else f'{v:.3f}'
    if fmt == 'd':
        return str(int(v))
    return format(v, fmt)


def table_markdown(table: list[dict[str, Any]], reference: str) -> str:
    head = '| ' + ' | '.join(h for h, _, _ in _DISPLAY) + ' |'
    sep = '|' + '|'.join('---' if f == 's' else '---:' for _, _, f in _DISPLAY) + '|'
    body = ['| ' + ' | '.join(_cell(r, k, f) for _, k, f in _DISPLAY) + ' |' for r in table]
    note = (f'\n\np: exact McNemar test of success vs. `{reference}` on the same episodes, per task. '
            'Governor rows show the measured mean frequency. J/success = total episode energy / successes.')
    return '\n'.join([head, sep, *body]) + note + '\n'


def table_latex(table: list[dict[str, Any]], reference: str) -> str:
    cols = 'l' + 'r' * (len(_DISPLAY) - 1)
    lines = [r'\begin{tabular}{' + cols + '}', r'\toprule',
             ' & '.join(h.replace('%', r'\%') for h, _, _ in _DISPLAY) + r' \\', r'\midrule']
    prev = None
    for r in table:
        if prev is not None and r['config'] != prev:
            lines.append(r'\midrule')
        prev = r['config']
        lines.append(' & '.join(_cell(r, k, f, latex=True).replace('<', '$<$') for _, k, f in _DISPLAY) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              f'% p: exact McNemar vs. {reference} per task; governor rows: measured mean frequency.']
    return '\n'.join(lines) + '\n'
