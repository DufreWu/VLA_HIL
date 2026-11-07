"""Offline metrics and paired comparison for finished evaluation runs (no re-simulation).

Reads ``episodes.jsonl``, ``run_config.json`` and per-episode traces written by
``simulator/evaluate.py`` and derives:

* normalized task progress (0..1, 1 == success criterion reached), for both tasks;
* peak cube linear speed;
* pushing tip-over (cube centre rises >= 1.0 cm, i.e. the 5 cm cube rolls onto an edge);
* communication overhead (RTT - server inference time) and real-time factor.

Runs are compared episode-by-episode, paired on (task, seed, episode); evaluate.py seeds
initial cube poses, so equal keys mean equal initial states (checked). Binary metrics use
an exact McNemar test, continuous metrics a paired sign-flip permutation test, and mean
differences get a paired bootstrap 95% CI. Only NumPy is required.

Example:
    python simulator/posthoc_metrics.py \
        --run server=outputs/eval_rtx5000_continuous \
        --run jetson=outputs/eval_jetson_continuous \
        --output outputs/posthoc_server_vs_jetson
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

TASKS = ('grasping', 'pushing')

# (key, label, kind) -- kind: 'binary' or 'continuous'.
METRICS = [
    ('success', 'Success rate (%)', 'binary'),
    ('collision', 'Collision rate (%)', 'binary'),
    ('progress', 'Task progress (0-1)', 'continuous'),
    ('progress_final', 'Final task progress (0-1)', 'continuous'),
    ('completion_time_s', 'Completion time (s)', 'continuous'),
    ('ajj', 'AJJ (rad/s^3)', 'continuous'),
    ('cube_peak_speed', 'Cube peak speed (m/s)', 'continuous'),
    ('tip_over', 'Tip-over rate, pushing (%)', 'binary'),
    ('mean_inference_ms', 'Inference time (ms)', 'continuous'),
    ('mean_rtt_ms', 'RTT (ms)', 'continuous'),
    ('p95_rtt_ms', 'P95 RTT (ms)', 'continuous'),
    ('comm_overhead_ms', 'RTT - inference (ms)', 'continuous'),
    ('dropped_action_slots', 'Dropped action slots', 'continuous'),
    ('queue_starvation_events', 'Queue starvation steps', 'continuous'),
    ('real_time_factor', 'Real-time factor', 'continuous'),
]
PERCENT = {'success', 'collision', 'tip_over'}
# Metrics in the LaTeX table.
TABLE_METRICS = ['success', 'collision', 'progress', 'completion_time_s', 'ajj',
                 'cube_peak_speed', 'tip_over', 'mean_inference_ms', 'mean_rtt_ms']


# ---------------------------------------------------------------- per-episode metrics

def task_progress(task, positions, target, lift_m, target_tol_m):
    """Per-frame progress in [0, 1]; reaching the task's success threshold gives 1."""
    p = np.asarray(positions, dtype=float)
    if task == 'grasping':
        return np.clip((p[:, 2] - p[0, 2]) / lift_m, 0.0, 1.0)
    dist = np.linalg.norm(p[:, :2] - np.asarray(target[:2], dtype=float), axis=1)
    span = dist[0] - target_tol_m
    if span <= 0:  # started inside the tolerance
        return np.ones(len(p))
    return np.clip((dist[0] - dist) / span, 0.0, 1.0)


def tip_over(positions, rise_m):
    """True if the cube centre ever rises >= rise_m above its initial height."""
    z = np.asarray(positions, dtype=float)[:, 2]
    return bool((z - z[0] >= rise_m).any())


def trace_path(run_dir, task, seed, episode, multi_seed):
    prefix = f'{task}_seed{seed}' if multi_seed else task
    return run_dir / f'{prefix}_{episode:04d}_trace.jsonl'


def load_run(run_dir, tip_rise_m):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'run_config.json').read_text())
    metric_cfg = config.get('metrics', {})
    lift_m = float(metric_cfg.get('lift_m', 0.10))
    tol_m = float(metric_cfg.get('target_tolerance_m', 0.03))
    seeds = config['arguments'].get('seeds') or [config['arguments']['seed']]
    multi_seed = len(seeds) > 1
    episodes = [json.loads(line) for line in (run_dir / 'episodes.jsonl').open() if line.strip()]
    rows = []
    for e in episodes:
        path = trace_path(run_dir, e['task'], e['seed'], e['episode'], multi_seed)
        positions, speeds = [], []
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                positions.append(r['position'])
                speeds.append(float(np.linalg.norm(r['velocity'][:3])))
        progress = task_progress(e['task'], positions, e['target'], lift_m, tol_m)
        rtt, inf = e.get('mean_rtt_ms'), e.get('mean_inference_ms')
        wall = e.get('wall_time_s') or e.get('tct_wall_s')
        rows.append(dict(
            task=e['task'], seed=int(e['seed']), episode=int(e['episode']),
            initial=e['initial'],
            success=float(bool(e['success'])), collision=float(bool(e['collision'])),
            progress=float(progress.max()), progress_final=float(progress[-1]),
            completion_time_s=e.get('completion_time_s'),
            ajj=e.get('ajj'),
            cube_peak_speed=max(speeds) if speeds else None,
            tip_over=float(tip_over(positions, tip_rise_m)) if e['task'] == 'pushing' else None,
            mean_inference_ms=inf, mean_rtt_ms=rtt, p95_rtt_ms=e.get('p95_rtt_ms'),
            comm_overhead_ms=(rtt - inf) if rtt is not None and inf is not None else None,
            dropped_action_slots=e.get('dropped_action_slots'),
            queue_starvation_events=e.get('queue_starvation_events'),
            real_time_factor=(e['duration_s'] / wall) if wall else None,
        ))
    return rows, config


# ---------------------------------------------------------------- statistics

def values(rows, key):
    return np.array([np.nan if r[key] is None else float(r[key]) for r in rows], dtype=float)


def describe(rows, key):
    """Episode mean, and mean/sd of the per-seed means."""
    v = values(rows, key)
    ok = ~np.isnan(v)
    if not ok.any():
        return dict(n=0, mean=None, seed_mean=None, seed_sd=None)
    scale = 100.0 if key in PERCENT else 1.0
    seeds = np.array([r['seed'] for r in rows])
    per_seed = [np.nanmean(v[seeds == s]) for s in sorted(set(seeds[ok]))]
    per_seed = [x for x in per_seed if not np.isnan(x)]
    sd = float(np.std(per_seed, ddof=1)) * scale if len(per_seed) > 1 else None
    return dict(n=int(ok.sum()), mean=float(np.nanmean(v)) * scale,
                seed_mean=float(np.mean(per_seed)) * scale, seed_sd=sd)


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def permutation_paired(diff, n_perm, rng):
    """Two-sided paired sign-flip permutation test on the mean difference."""
    diff = np.asarray(diff, float)
    observed = abs(diff.mean())
    if observed == 0:
        return 1.0
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    return float((1 + (null >= observed - 1e-12).sum()) / (n_perm + 1))


def bootstrap_ci(diff, n_boot, rng):
    diff = np.asarray(diff, float)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(ref_rows, other_rows, key, kind, n_resample, rng):
    """Paired comparison on episodes present (with a value) in both runs."""
    ref = {(r['task'], r['seed'], r['episode']): r[key] for r in ref_rows}
    pairs = [(ref[k], r[key]) for r in other_rows
             if (k := (r['task'], r['seed'], r['episode'])) in ref
             and ref[k] is not None and r[key] is not None]
    if len(pairs) < 2:
        return None
    a, b = (np.array(x, float) for x in zip(*pairs))
    scale = 100.0 if key in PERCENT else 1.0
    diff = (b - a) * scale
    lo, hi = bootstrap_ci(diff, n_resample, rng)
    if kind == 'binary':
        p, test = mcnemar_exact(a, b), 'exact McNemar'
    else:
        p, test = permutation_paired(diff, n_resample, rng), 'paired permutation'
    return dict(n_pairs=len(pairs), ref_mean=float(a.mean()) * scale, other_mean=float(b.mean()) * scale,
                diff=float(diff.mean()), ci_low=lo, ci_high=hi, p_value=p, test=test)


# ---------------------------------------------------------------- output

def check_pairing(runs):
    names = list(runs)
    ref = {(r['task'], r['seed'], r['episode']): r['initial'] for r in runs[names[0]]}
    for name in names[1:]:
        keys = {(r['task'], r['seed'], r['episode']) for r in runs[name]}
        common = keys & set(ref)
        mismatched = [k for r in runs[name]
                      if (k := (r['task'], r['seed'], r['episode'])) in ref
                      and not np.allclose(r['initial'], ref[k], atol=1e-6)]
        print(f'pairing {names[0]} vs {name}: {len(common)} common episodes, '
              f'{len(set(ref) - keys)} only in {names[0]}, {len(keys - set(ref))} only in {name}')
        if mismatched:
            raise SystemExit(f'Initial cube states differ for {len(mismatched)} paired episodes '
                             f'(e.g. {mismatched[0]}); runs are not comparable episode-by-episode.')


def subsets(rows):
    yield 'both', rows
    for t in TASKS:
        sub = [r for r in rows if r['task'] == t]
        if sub:
            yield t, sub


def fmt(x, digits=2):
    return '' if x is None else f'{x:.{digits}f}'


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def latex_table(summary, run_names):
    cell = {(s['run'], s['task'], s['metric']): s for s in summary}
    labels = {k: lbl for k, lbl, _ in METRICS}
    tasks = [t for t in ('grasping', 'pushing', 'both') if any(s['task'] == t for s in summary)]
    cols = 'l' + 'c' * (len(run_names) * len(tasks))
    lines = [r'\begin{tabular}{' + cols + '}', r'\toprule',
             'Metric & ' + ' & '.join(rf'\multicolumn{{{len(run_names)}}}{{c}}{{{t.capitalize()}}}' for t in tasks) + r' \\',
             ' & ' + ' & '.join(n for _ in tasks for n in run_names) + r' \\', r'\midrule']
    for key in TABLE_METRICS:
        cells = []
        for t in tasks:
            for n in run_names:
                s = cell.get((n, t, key))
                if not s or s['seed_mean'] is None:
                    cells.append('--')
                elif s['seed_sd'] is None:
                    cells.append(fmt(s['seed_mean']))
                else:
                    cells.append(f"{fmt(s['seed_mean'])} $\\pm$ {fmt(s['seed_sd'])}")
        label = labels[key].replace('%', r'\%').replace('^3', '$^3$')
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run', action='append', required=True, metavar='NAME=DIR',
                   help='Evaluation output directory; repeat. The first run is the comparison reference.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tip-rise-cm', type=float, default=1.0,
                   help='Cube-centre rise counted as tip-over (5 cm cube on an edge: 1.04 cm)')
    p.add_argument('--resamples', type=int, default=10000, help='Bootstrap/permutation resamples')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    runs = {}
    for spec in args.run:
        name, sep, path = spec.partition('=')
        if not sep:
            name, path = Path(spec).name, spec
        print(f'loading {name}: {path}')
        runs[name], _ = load_run(path, args.tip_rise_cm / 100.0)
    if len(runs) > 1:
        check_pairing(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    names = list(runs)

    episode_fields = ['run', 'task', 'seed', 'episode'] + [k for k, _, _ in METRICS]
    write_csv(args.output / 'episodes_posthoc.csv',
              [dict(run=n, **r) for n in names for r in runs[n]], episode_fields)

    summary = []
    for n in names:
        for task, rows in subsets(runs[n]):
            for key, label, _ in METRICS:
                summary.append(dict(run=n, task=task, metric=key, label=label, **describe(rows, key)))
    write_csv(args.output / 'summary.csv', summary,
              ['run', 'task', 'metric', 'label', 'n', 'mean', 'seed_mean', 'seed_sd'])

    comparisons = []
    for other in names[1:]:
        ref_sub, other_sub = dict(subsets(runs[names[0]])), dict(subsets(runs[other]))
        for task in ref_sub:
            if task not in other_sub:
                continue
            for key, label, kind in METRICS:
                c = compare(ref_sub[task], other_sub[task], key, kind, args.resamples, rng)
                if c:
                    comparisons.append(dict(reference=names[0], other=other, task=task,
                                            metric=key, label=label, **c))
    if comparisons:
        write_csv(args.output / 'comparison.csv', comparisons,
                  ['reference', 'other', 'task', 'metric', 'label', 'n_pairs', 'ref_mean',
                   'other_mean', 'diff', 'ci_low', 'ci_high', 'p_value', 'test'])
    (args.output / 'summary_table.tex').write_text(latex_table(summary, names))
    (args.output / 'posthoc_config.json').write_text(json.dumps(dict(
        runs={n: s.partition('=')[2] or s for n, s in zip(names, args.run)},
        tip_rise_cm=args.tip_rise_cm, resamples=args.resamples, seed=args.seed,
        progress_definition='grasping: max cube rise / lift_m; pushing: (d0 - d) / (d0 - target_tolerance_m); '
                            'clipped to [0, 1], max over the episode',
        aggregation='seed_mean/seed_sd: mean and sample SD of per-seed means'), indent=2))

    # Console report.
    for task in ('grasping', 'pushing', 'both'):
        print(f'\n== {task}')
        print(f"{'metric':30s}" + ''.join(f'{n:>20s}' for n in names) +
              ('' if len(names) < 2 else f"{'diff [95% CI]':>28s}{'p':>9s}"))
        for key, label, _ in METRICS:
            cells = [next((s for s in summary if s['run'] == n and s['task'] == task and s['metric'] == key), None)
                     for n in names]
            if all(c is None or c['n'] == 0 for c in cells):
                continue
            line = f'{label:30s}' + ''.join(
                f"{'--':>20s}" if c is None or c['n'] == 0 else
                f"{fmt(c['seed_mean']) + (' ± ' + fmt(c['seed_sd']) if c['seed_sd'] is not None else ''):>20s}"
                for c in cells)
            c = next((c for c in comparisons if c['task'] == task and c['metric'] == key
                      and c['other'] == names[-1]), None) if len(names) > 1 else None
            if c:
                line += f"{fmt(c['diff']) + ' [' + fmt(c['ci_low']) + ', ' + fmt(c['ci_high']) + ']':>28s}{c['p_value']:>9.4f}"
            print(line)
    print(f'\nWrote {args.output}/episodes_posthoc.csv, summary.csv, comparison.csv, summary_table.tex')


if __name__ == '__main__':
    main()
