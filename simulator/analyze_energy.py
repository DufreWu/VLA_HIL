"""Aggregate, validate and plot a DVFS energy evaluation (offline; no Isaac Sim needed).

  python simulator/analyze_energy.py outputs/dvfs_energy_evaluation/<experiment_id> [--reference default]

Writes per_task_summary.csv, per_config_summary.csv, paired_tests.csv,
latency_horizon_analysis.csv, validation_report.json and figures/fig{1..6}_*.{pdf,png}
into the experiment directory. Never modifies the raw logs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.evaluation.energy import (read_csv, to_float, trapezoid, finite, EPISODE_COLUMNS,
                                         INFERENCE_COLUMNS, HORIZON_TICKS, FPS)
from simulator.evaluation.dvfs_study import wilson_ci, mcnemar_exact
from simulator.posthoc_metrics import permutation_paired, bootstrap_ci

BOOL_COLS = {'success', 'collision', 'frequency_valid', 'frequency_stable', 'used', 'drained', 'exceeds_horizon'}
STR_COLS = {'experiment_id', 'episode_uid', 'configuration', 'task', 'failure_reason', 'raw_reason',
            'cpu_governor', 'gpu_governor', 'power_mode', 'error'}
# (column, label) of the continuous metrics summarized per group.
METRICS = [
    ('inference_ms_mean', 'inference latency mean (ms)'), ('inference_ms_p95', 'inference latency p95 (ms)'),
    ('computational_power_w', 'computational power (W)'), ('computational_energy_j', 'E_comp (J)'),
    ('inference_energy_j', 'E_inf per inference (J)'), ('computational_edp_js', 'EDP_comp (J s)'),
    ('mechanical_power_w', 'mechanical power (W)'), ('mechanical_energy_j', 'E_mech (J)'),
    ('actuator_power_w', 'actuator power (W, model)'), ('actuator_energy_j', 'E_act (J, model)'),
    ('total_energy_j', 'E_total (J)'), ('total_power_w', 'total power (W)'),
    ('completion_time_s', 'completion time (s, successes)'), ('episode_duration_s', 'episode duration (s)'),
    ('ajj', 'AJJ (rad/s^3)'), ('action_delay_ms', 'action delay (ms)'), ('action_delay_ticks', 'action delay (ticks)'),
    ('obs_age_ms', 'observation age (ms)'), ('holding_time_s', 'holding time (s)'),
    ('underrun_s', 'underrun duration (s)'), ('buffer_underrun_rate', 'buffer underrun rate'),
    ('dropped_action_slots', 'dropped slots'), ('real_time_factor', 'RTF'),
    ('cpu_mhz_actual', 'CPU MHz (measured)'), ('gpu_mhz_actual', 'GPU MHz (measured)'),
]


def _parse(rows):
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if k in STR_COLS:
                d[k] = v
            elif k in BOOL_COLS:
                d[k] = None if v in ('', 'nan', 'None') else v in ('1', 'True', 'true')
            else:
                d[k] = to_float(v)
        out.append(d)
    return out


def load(exp: Path):
    eps = _parse(read_csv(exp / 'episodes.csv'))
    reqs = _parse(read_csv(exp / 'inference_requests.csv')) if (exp / 'inference_requests.csv').exists() else []
    cfg = json.loads((exp / 'experiment_config.json').read_text())
    return eps, reqs, cfg


# ------------------------------------------------------------------ statistics

def _arr(rows, key):
    return np.array([r[key] if isinstance(r.get(key), (int, float)) else math.nan for r in rows], float)


def group_stats(rows, seeds):
    n = len(rows)
    k = sum(bool(r['success']) for r in rows)
    c = sum(bool(r['collision']) for r in rows)
    lo, hi = wilson_ci(k, n)
    clo, chi = wilson_ci(c, n)
    out = dict(episodes=n, successes=k, success_rate_pct=100 * k / n if n else math.nan,
               success_ci95_low_pct=lo, success_ci95_high_pct=hi,
               collisions=c, collision_rate_pct=100 * c / n if n else math.nan,
               collision_ci95_low_pct=clo, collision_ci95_high_pct=chi)
    # Seed level: success rate per seed, then mean/SD across seeds (not episode-level SD).
    per_seed = [100 * np.mean([bool(r['success']) for r in rows if r['seed'] == s])
                for s in seeds if any(r['seed'] == s for r in rows)]
    out.update(n_seeds=len(per_seed), success_rate_seed_mean_pct=float(np.mean(per_seed)) if per_seed else math.nan,
               success_rate_seed_sd_pct=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else math.nan)
    for key, _ in METRICS:
        v = _arr(rows, key)
        v = v[np.isfinite(v)]
        out[f'{key}_mean'] = float(v.mean()) if v.size else math.nan
        out[f'{key}_sd'] = float(v.std(ddof=1)) if v.size > 1 else math.nan
        out[f'{key}_n'] = int(v.size)
    for key in ('computational_energy_j', 'total_energy_j', 'mechanical_energy_j'):
        v = _arr(rows, key)
        # Includes failed/timed-out episodes; undefined (NaN) with zero successes or missing data.
        out[f'{key.replace("_j", "")}_per_success_j'] = (float(v.sum()) / k if k and np.isfinite(v).all()
                                                       else math.nan)
    per_seed_e = [np.nanmean(_arr([r for r in rows if r['seed'] == s], 'computational_energy_j'))
                  for s in seeds if any(r['seed'] == s for r in rows)]
    per_seed_e = [x for x in per_seed_e if np.isfinite(x)]
    out['computational_energy_seed_sd_j'] = float(np.std(per_seed_e, ddof=1)) if len(per_seed_e) > 1 else math.nan
    fv = [r['frequency_valid'] for r in rows if r['frequency_valid'] is not None]
    fs = [r['frequency_stable'] for r in rows if r['frequency_stable'] is not None]
    out.update(frequency_valid_all=all(fv) if fv else None, frequency_stable_all=all(fs) if fs else None,
               failure_reasons=json.dumps({x: sum(r['failure_reason'] == x for r in rows)
                                           for x in sorted({r['failure_reason'] for r in rows}) if x}))
    return out


def summaries(eps, configs, seeds):
    tasks = list(dict.fromkeys(r['task'] for r in eps))
    per_task, per_config = [], []
    for cfg in configs:
        for t in tasks:
            g = [r for r in eps if r['configuration'] == cfg and r['task'] == t]
            if g:
                per_task.append(dict(configuration=cfg, task=t, **group_stats(g, seeds)))
        g = [r for r in eps if r['configuration'] == cfg]
        if g:
            per_config.append(dict(configuration=cfg, task='all', **group_stats(g, seeds)))
    return per_task, per_config


def paired_tests(eps, reference, configs, n_resample=10000, seed=0):
    """Matched episodes (task, seed, episode): McNemar for success; sign-flip permutation and
    bootstrap 95% CI of the mean paired difference for energies and latency."""
    rng = np.random.default_rng(seed)
    key = lambda r: (r['task'], r['seed'], r['episode'])
    ref = {key(r): r for r in eps if r['configuration'] == reference}
    out = []
    tasks = list(dict.fromkeys(r['task'] for r in eps))
    for cfg in configs:
        if cfg == reference:
            continue
        other = {key(r): r for r in eps if r['configuration'] == cfg}
        for task in tasks + ['all']:
            ks = [k for k in ref if k in other and (task == 'all' or k[0] == task)]
            if not ks:
                continue
            b = sum(bool(ref[k]['success']) and not other[k]['success'] for k in ks)
            c = sum(bool(other[k]['success']) and not ref[k]['success'] for k in ks)
            out.append(dict(configuration=cfg, reference=reference, task=task, metric='success',
                            n_pairs=len(ks), diff_mean=(c - b) / len(ks) * 100, ci95_low=math.nan,
                            ci95_high=math.nan, p_value=mcnemar_exact(b, c), test='exact McNemar',
                            note=f'ref-only {b}, config-only {c}; diff in percentage points'))
            for m in ('computational_energy_j', 'total_energy_j', 'mechanical_energy_j', 'inference_ms_mean',
                      'inference_energy_j'):
                d = np.array([other[k][m] - ref[k][m] for k in ks], float)
                d = d[np.isfinite(d)]
                if d.size < 2:
                    continue
                lo, hi = bootstrap_ci(d, n_resample, rng)
                out.append(dict(configuration=cfg, reference=reference, task=task, metric=m, n_pairs=int(d.size),
                                diff_mean=float(d.mean()), ci95_low=lo, ci95_high=hi,
                                p_value=permutation_paired(d, n_resample, rng),
                                test='paired sign-flip permutation', note='config minus reference'))
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return math.nan
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def latency_horizon(eps, reqs):
    """Descriptive association between end-to-end latency above the 2.5 s horizon, buffer
    underruns and success (no causal claim)."""
    by_uid = {}
    for q in reqs:
        by_uid.setdefault(q['episode_uid'], []).append(q)
    rows = []
    for r in eps:
        qs = [q for q in by_uid.get(r['episode_uid'], []) if finite(q['end_to_end_ms'])]
        r['_frac_over'] = (np.mean([q['end_to_end_ms'] > HORIZON_TICKS / FPS * 1000 for q in qs])
                           if qs else math.nan)
    for task in list(dict.fromkeys(r['task'] for r in eps)) + ['all']:
        g = [r for r in eps if task == 'all' or r['task'] == task]
        for label, sel in (('all requests <= horizon', lambda r: r['_frac_over'] == 0),
                           ('some requests > horizon', lambda r: 0 < r['_frac_over'] < 1),
                           ('all requests > horizon', lambda r: r['_frac_over'] == 1)):
            s = [r for r in g if finite(r['_frac_over']) and sel(r)]
            k = sum(bool(r['success']) for r in s)
            lo, hi = wilson_ci(k, len(s))
            rows.append(dict(task=task, group=label, episodes=len(s), successes=k,
                             success_rate_pct=100 * k / len(s) if s else math.nan,
                             success_ci95_low_pct=lo, success_ci95_high_pct=hi,
                             underrun_rate_mean=float(np.nanmean(_arr(s, 'buffer_underrun_rate'))) if s else math.nan,
                             holding_time_s_mean=float(np.nanmean(_arr(s, 'holding_time_s'))) if s else math.nan))
        rows.append(dict(task=task, group='spearman(inference_ms_mean, buffer_underrun_rate)',
                         episodes=len(g), rho=spearman(_arr(g, 'inference_ms_mean'), _arr(g, 'buffer_underrun_rate'))))
        rows.append(dict(task=task, group='spearman(inference_ms_mean, success)',
                         episodes=len(g), rho=spearman(_arr(g, 'inference_ms_mean'),
                                                       [float(bool(r['success'])) for r in g])))
    return rows


# ------------------------------------------------------------------ validation

def validate(exp: Path, eps, reqs, per_task, per_config, cfg, raw_tol=1e-6):
    checks = []

    def add(name, ok, detail='', severity='error'):
        checks.append(dict(check=name, passed=bool(ok), severity=severity, detail=detail))

    cols = list(read_csv(exp / 'episodes.csv')[0].keys()) if eps else []
    add('episodes.csv has all required columns', set(EPISODE_COLUMNS) <= set(cols),
        f'missing: {sorted(set(EPISODE_COLUMNS) - set(cols))}')
    if reqs:
        rcols = list(read_csv(exp / 'inference_requests.csv')[0].keys())
        add('inference_requests.csv has all columns', set(INFERENCE_COLUMNS) <= set(rcols))
    uids = [r['episode_uid'] for r in eps]
    add('no duplicated episode IDs', len(uids) == len(set(uids)), f'{len(uids) - len(set(uids))} duplicates')
    keys = [(r['configuration'], r['task'], r['seed'], r['episode']) for r in eps]
    add('no duplicated (config, task, seed, episode)', len(keys) == len(set(keys)))
    for col, allow_nan in (('computational_energy_j', True), ('mechanical_energy_j', False),
                           ('actuator_energy_j', True), ('total_energy_j', True)):
        v = _arr(eps, col)
        add(f'{col} >= 0', np.all(v[np.isfinite(v)] >= 0), f'{int((v < 0).sum())} negative')
        if not allow_nan:
            add(f'{col} available for every episode', np.isfinite(v).all(), f'{int((~np.isfinite(v)).sum())} NaN')
    ec, ea, et = (_arr(eps, k) for k in ('computational_energy_j', 'actuator_energy_j', 'total_energy_j'))
    both = np.isfinite(ec) & np.isfinite(ea)
    add('E_total = E_comp + E_act where available', np.allclose(et[both], ec[both] + ea[both], rtol=1e-9, atol=1e-9)
        and not np.isfinite(et[~both]).any(), f'{int(both.sum())} episodes with E_act')
    # Exact zeros in measured quantities usually mean a missing value was replaced by 0.
    for col in ('computational_energy_j', 'computational_power_w', 'inference_ms_mean', 'inference_energy_j',
                'mechanical_energy_j', 'power_sample_rate_hz'):
        v = _arr(eps, col)
        add(f'no exact zeros in {col} (missing values must be NaN)', not np.any(v == 0), f'{int((v == 0).sum())} zeros')
    avail = {c: int(np.isfinite(_arr(eps, c)).sum()) for c in
             ('computational_energy_j', 'inference_energy_j', 'computational_edp_js', 'actuator_energy_j',
              'total_energy_j')}
    add('availability of measured/model quantities (count of non-NaN episodes)', True, json.dumps(avail), 'info')
    # Denominators and zero-success handling.
    ok_den, ok_zero = True, True
    for s in per_task + per_config:
        g = [r for r in eps if r['configuration'] == s['configuration'] and (s['task'] == 'all' or r['task'] == s['task'])]
        k = sum(bool(r['success']) for r in g)
        ok_den &= s['episodes'] == len(g) and s['successes'] == k and math.isclose(
            s['success_rate_pct'], 100 * k / len(g))
        if k == 0:
            ok_zero &= all(math.isnan(s[c]) for c in ('computational_energy_per_success_j', 'total_energy_per_success_j'))
    add('success-rate denominators = attempted episodes', ok_den)
    add('zero-success groups report energy/success as NaN', ok_zero)
    # Frequency validity.
    bad = [r['episode_uid'] for r in eps if r['frequency_valid'] is False]
    add('requested DVFS configuration applied (switch read-back)', not bad, f'{len(bad)} invalid: {bad[:5]}')
    unstable = [r['episode_uid'] for r in eps if r['frequency_stable'] is False]
    add('pinned frequencies stable during episodes (no throttling/unexpected change)', not unstable,
        f'{len(unstable)} episodes: {unstable[:5]}', 'warning')
    # Timestamps.
    bad_t = [r['episode_uid'] for r in eps if not (r['episode_duration_s'] > 0 and (
        not r['success'] or 0 < r['completion_time_s'] <= r['episode_duration_s'] + 1e-9))]
    bad_ct = [r['episode_uid'] for r in eps if not r['success'] and finite(r['completion_time_s'])]
    add('valid episode timestamps (duration > 0, completion <= duration)', not bad_t, str(bad_t[:5]))
    add('completion time only for successful episodes', not bad_ct, str(bad_ct[:5]))
    order_bad = []
    for q in reqs:
        seq = [q[k] for k in ('capture_wall_s', 'encoded_wall_s', 'sent_wall_s', 'received_wall_s',
                              'available_wall_s', 'first_exec_wall_s') if finite(q[k])]
        if any(b < a - 1e-9 for a, b in zip(seq, seq[1:])):
            order_bad.append(f"{q['episode_uid']}#{int(q['request_index'])}")
    add('request timestamps monotone (capture<=sent<=received<=available<=first exec)', not order_bad, str(order_bad[:5]))
    neg_comm = [q for q in reqs if finite(q['communication_ms']) and q['communication_ms'] < 0]
    add('communication latency (RTT - server time) >= 0', not neg_comm, f'{len(neg_comm)} negative')
    # Continuous real-time execution and time alignment.
    rtf = _arr(eps, 'real_time_factor')
    add('real-time execution (0.95 <= RTF <= 1.05 in every episode)',
        np.all(np.abs(rtf[np.isfinite(rtf)] - 1) <= 0.05),
        f'min RTF {np.nanmin(rtf) if np.isfinite(rtf).any() else math.nan:.3f}', 'warning')
    mode = cfg.get('protocol', {}).get('mode')
    add('continuous mode with wall-clock pacing', mode == 'continuous' and cfg['protocol'].get('realtime_pacing'),
        f'mode={mode}')
    err = _arr(eps, 'time_alignment_error_s')
    dur = _arr(eps, 'episode_duration_s')
    rel = np.abs(err) / dur
    add('E_comp (wall) and E_mech/E_act (sim) intervals agree within 5%',
        np.all(rel[np.isfinite(rel)] <= 0.05), f'max |wall - sim| = {np.nanmax(np.abs(err)) if np.isfinite(err).any() else math.nan:.3f} s',
        'warning')
    # Re-integrate raw streams (correct time-window integration, units J = W*s).
    raw = recompute_from_raw(exp, eps)
    add('E_comp reproduced from power_samples.csv.gz', raw['comp_max_rel_err'] <= raw_tol or raw['comp_n'] == 0,
        f"{raw['comp_n']} episodes, max rel. error {raw['comp_max_rel_err']:.2e}")
    add('E_mech reproduced from joint_samples.csv.gz', raw['mech_max_rel_err'] <= raw_tol or raw['mech_n'] == 0,
        f"{raw['mech_n']} episodes, max rel. error {raw['mech_max_rel_err']:.2e}")
    add('joint samples at the physics rate (dt = 1/60 s)', raw['dt_ok'], raw['dt_detail'])
    drive = exp / 'drive_crosscheck.json'
    if drive.exists():
        d = json.loads(drive.read_text())
        rel = [x['relative_residual'] for x in d]
        user = max((abs(v) for x in d for v in x['user_actuation_effort_nm']), default=0.0)
        loaded = [c for x in d for c, r in zip(x['corr_per_joint'], x['rms_effort_nm']) if c is not None and r > 1.0]
        add('joint effort is the simulated joint load (inverse dynamics agreement; user actuation force = 0)',
            rel and max(rel) < 0.25 and user < 1e-6,
            f'relative residual {[round(r, 3) for r in rel]}; corr (joints > 1 N m RMS) '
            f'{[round(c, 3) for c in loaded]}; max |user actuation| {user:.1e} N m', 'warning')
    passed = all(c['passed'] for c in checks if c['severity'] == 'error')
    return dict(experiment_id=cfg.get('experiment_id'), passed=passed, n_episodes=len(eps), checks=checks)


def recompute_from_raw(exp, eps):
    out = dict(comp_n=0, comp_max_rel_err=0.0, mech_n=0, mech_max_rel_err=0.0, dt_ok=True, dt_detail='')
    by_uid = {r['episode_uid']: r for r in eps}
    p = exp / 'power_samples.csv.gz'
    if p.exists():
        samples = {}
        for r in read_csv(p):
            samples.setdefault(r['episode_uid'], []).append((float(r['server_t_s']), float(r['power_w'])))
        for uid, s in samples.items():
            r = by_uid.get(uid)
            if r is None or not finite(r['computational_energy_j']):
                continue
            from jetson_orin_nx.benchmark.dvfs_control import integrate
            e, _ = integrate(s, 0.0, r['comp_window_s'])
            if e is not None:
                out['comp_n'] += 1
                out['comp_max_rel_err'] = max(out['comp_max_rel_err'],
                                              abs(e - r['computational_energy_j']) / max(abs(e), 1e-12))
    j = exp / 'joint_samples.csv.gz'
    if j.exists():
        data = {}
        for r in read_csv(j):
            data.setdefault(r['episode_uid'], []).append([float(r[k]) for k in list(r)[1:]])
        worst_dt = 0.0
        for uid, rows in data.items():
            a = np.asarray(rows)
            t, qd, tau = a[:, 0], a[:, 8:15], a[:, 15:22]
            worst_dt = max(worst_dt, float(np.max(np.abs(np.diff(t) - 1 / 60))) if t.size > 1 else 0.0)
            r = by_uid.get(uid)
            if r is None:
                continue
            e = trapezoid(t, np.maximum(tau * qd, 0).sum(axis=1))
            out['mech_n'] += 1
            out['mech_max_rel_err'] = max(out['mech_max_rel_err'], abs(e - r['mechanical_energy_j']) / max(abs(e), 1e-12))
        out['dt_ok'] = worst_dt < 1e-4
        out['dt_detail'] = f'max |dt - 1/60| = {worst_dt:.2e} s over {len(data)} episodes'
    return out


# ------------------------------------------------------------------ figures

BLUE, ORANGE = '#2a78d6', '#eb6834'      # validated categorical slots 1-2 (tasks; comp vs actuation)
INK, MUTED, GRID = '#1f1f1e', '#5f5e58', '#e4e3df'


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                         'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.axisbelow': True, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return plt


def _save(fig, out, name, layout=True):
    out.mkdir(exist_ok=True)
    if layout:
        fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=300)
    fig.clf()


def _short(c):
    return c.replace('gpu_', 'G').replace('_cpu_', '/C').replace('max', 'max').replace('_', ' ')


SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']  # one-hue ramp


def grid_points(cfg):
    """{config name: (cpu MHz, gpu MHz)} for fixed-frequency configs (resolved on the device)."""
    out = {}
    for c in cfg.get('dvfs_configs') or []:
        if not c or c.get('cpu_khz') is None:
            continue
        cpu, gpu = c.get('resolved_cpu_khz', c['cpu_khz']), c.get('resolved_gpu_hz', c['gpu_hz'])
        if isinstance(cpu, (int, float)) and isinstance(gpu, (int, float)):
            out[c['name']] = (cpu / 1e3, gpu / 1e6)
    return out


def heatmap_figures(exp, eps, reqs, per_task, per_config, points, model_label):
    """Figures 1-6 as CPU x GPU heatmaps (used when the configs form a frequency grid)."""
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    cmap.set_bad('#f0efec')
    out = exp / 'figures'
    cpus = sorted({c for c, _ in points.values()})
    gpus = sorted({g for _, g in points.values()})
    others = [c for c in dict.fromkeys(r['configuration'] for r in eps) if c not in points]
    tasks = list(dict.fromkeys(r['task'] for r in eps))
    made, skipped = [], {}

    def lookup(rows, cfg, task, key):
        for s_ in rows:
            if s_['configuration'] == cfg and s_['task'] == task:
                return s_.get(key, math.nan)
        return math.nan

    def matrix(fn):
        m = np.full((len(cpus), len(gpus)), np.nan)
        for name, (c, g) in points.items():
            m[cpus.index(c), gpus.index(g)] = fn(name)
        return m

    def draw(name, panels):
        fig, axes = plt.subplots(1, len(panels), figsize=(2.45 * len(panels), 0.5 + 0.17 * len(cpus) + 0.9),
                                 squeeze=False)
        for ax, (label, fn, fmt, *rng) in zip(axes[0], panels):
            m = matrix(fn)
            finite_m = m[np.isfinite(m)]
            lo, hi = rng[0] if rng else ((finite_m.min(), finite_m.max()) if finite_m.size else (0, 1))
            im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, aspect='auto', origin='lower', vmin=lo, vmax=hi)
            ax.grid(False)
            ax.set_xticks(range(len(gpus)), [f'{g:g}' for g in gpus], rotation=90)
            ax.set_yticks(range(len(cpus)), [f'{c:g}' for c in cpus])
            ax.set_xlabel('GPU frequency (MHz)')
            if ax is axes[0][0]:
                ax.set_ylabel('CPU frequency (MHz)')
            mid = (lo + hi) / 2
            if len(cpus) * len(gpus) <= 250:
                for i in range(len(cpus)):
                    for j in range(len(gpus)):
                        v = m[i, j]
                        ax.text(j, i, 'n/a' if not np.isfinite(v) else format(v, fmt), ha='center', va='center',
                                fontsize=4.5, color=MUTED if not np.isfinite(v) else ('white' if v > mid else INK))
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
            cb.outline.set_visible(False)
            ref = [f'{o}: {format(fn(o), fmt)}' for o in others if np.isfinite(fn(o))]
            ax.set_title(label + (f'\n({"; ".join(ref)})' if ref else ''), fontsize=7)
        _save(fig, out, name); made.append(name.split('_')[0]); plt.close(fig)

    by_cfg_lat = {}
    for q in reqs:
        if finite(q['inference_ms']):
            by_cfg_lat.setdefault(q['configuration'], []).append(q['inference_ms'] / 1000)
    med = lambda c: float(np.median(by_cfg_lat[c])) if by_cfg_lat.get(c) else math.nan
    allstat = lambda key: (lambda c: lookup(per_config, c, 'all', key))
    taskstat = lambda t, key: (lambda c: lookup(per_task, c, t, key))
    draw('fig1_inference_latency', [('Median inference latency (s)', med, '.2f'),
                                     ('Mean P95 latency per episode (s)',
                                      lambda c: allstat('inference_ms_p95_mean')(c) / 1000, '.2f')])
    draw('fig2_power_inference_energy', [('Computational power (W, measured)', allstat('computational_power_w_mean'), '.1f'),
                                         ('Energy per inference (J, measured)', allstat('inference_energy_j_mean'), '.2f'),
                                         ('E_comp per episode (J, measured)', allstat('computational_energy_j_mean'), '.0f')])
    if np.isfinite(matrix(allstat('actuator_energy_j_mean'))).any():
        draw('fig3_energy_breakdown', [('E_comp (J, measured)', allstat('computational_energy_j_mean'), '.0f'),
                                       (f'E_act (J, {model_label})', allstat('actuator_energy_j_mean'), '.0f'),
                                       ('E_total (J)', allstat('total_energy_j_mean'), '.0f')])
    else:
        skipped['fig3'] = 'actuator energy model unavailable (no total-energy chart)'
    draw('fig4_success_rate', [(f'{t.capitalize()} success rate (%)', taskstat(t, 'success_rate_pct'), '.0f', (0, 100))
                               for t in tasks])
    key = ('total_energy_per_success_j' if 'fig3' not in skipped else 'computational_energy_per_success_j')
    lab = 'Total' if key.startswith('total') else 'Computational'
    draw('fig5_energy_per_success', [(f'{t.capitalize()}: {lab} energy per success (J)', taskstat(t, key), '.0f')
                                     for t in tasks])
    draw('fig6_action_timing', [('Action delay (s)', lambda c: allstat('action_delay_ms_mean')(c) / 1000, '.2f'),
                                ('Holding time (s)', allstat('holding_time_s_mean'), '.1f'),
                                ('Buffer underrun rate (%)', lambda c: allstat('buffer_underrun_rate_mean')(c) * 100, '.0f')])
    return dict(made=made, skipped=skipped, layout='CPU x GPU heatmap', grid=dict(cpu_mhz=cpus, gpu_mhz=gpus))


def figures(exp, eps, reqs, per_task, per_config, configs, model_label):
    plt = _style()
    out = exp / 'figures'
    tasks = list(dict.fromkeys(r['task'] for r in eps))
    colors = dict(zip(tasks, (BLUE, ORANGE)))
    x = np.arange(len(configs))
    w = 0.38
    made, skipped = [], {}

    def stat(rows, cfg, task, key):
        for s in rows:
            if s['configuration'] == cfg and s['task'] == task:
                return s.get(key, math.nan)
        return math.nan

    # Fig 1: per-request forward-pass latency distribution per configuration (both tasks).
    lat = [_arr([q for q in reqs if q['configuration'] == c], 'inference_ms') for c in configs]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    data = [v[np.isfinite(v)] / 1000 for v in lat]
    if any(d.size for d in data):
        ax.boxplot([d if d.size else [math.nan] for d in data], widths=0.5, showfliers=False,
                   medianprops=dict(color=BLUE, linewidth=1.5), boxprops=dict(color=MUTED),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        p95 = [np.nanmean(_arr([r for r in eps if r['configuration'] == c], 'inference_ms_p95')) / 1000 for c in configs]
        ax.plot(x + 1, p95, 'D', color=ORANGE, markersize=4, label='P95 per episode (mean)')
        ax.axhline(HORIZON_TICKS / FPS, color=MUTED, linestyle='--', linewidth=0.8)
        ax.text(len(configs) + 0.45, HORIZON_TICKS / FPS, '2.5 s horizon', va='bottom', ha='right', fontsize=6, color=MUTED)
        ax.set_xticks(x + 1, [_short(c) for c in configs])
        ax.set_ylabel('Inference latency (s)'); ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0))
        _save(fig, out, 'fig1_inference_latency'); made.append('fig1')
    else:
        skipped['fig1'] = 'no inference latency data'
    plt.close(fig)

    # Fig 2: computational power and per-inference energy (two panels, one axis each).
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, key, label in ((axes[0], 'computational_power_w', 'Computational power (W)'),
                           (axes[1], 'inference_energy_j', 'Energy per inference (J)')):
        m = [stat(per_config, c, 'all', f'{key}_mean') for c in configs]
        sd = [stat(per_config, c, 'all', f'{key}_sd') for c in configs]
        ax.bar(x, m, 0.6, yerr=sd, color=BLUE, error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
        ax.set_xticks(x, [_short(c) for c in configs]); ax.set_ylabel(label)
    axes[0].set_title('Measured VDD_IN, episode mean $\\pm$ SD', color=MUTED)
    axes[1].set_title('Measured VDD_IN over each forward pass', color=MUTED)
    _save(fig, out, 'fig2_power_inference_energy'); made.append('fig2'); plt.close(fig)

    # Fig 3: energy breakdown; only with a valid actuator model.
    ea = [stat(per_config, c, 'all', 'actuator_energy_j_mean') for c in configs]
    if all(np.isfinite(ea)):
        ec = [stat(per_config, c, 'all', 'computational_energy_j_mean') for c in configs]
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        ax.bar(x, ec, 0.6, color=BLUE, label='Computation (measured)', edgecolor='white', linewidth=1)
        ax.bar(x, ea, 0.6, bottom=ec, color=ORANGE, hatch='///', edgecolor='white', linewidth=1,
               label=f'Actuation ({model_label})')
        for i, (a, b) in enumerate(zip(ec, ea)):
            ax.text(i, a + b, f'{a + b:.0f}', ha='center', va='bottom', fontsize=6, color=INK)
        ax.set_xticks(x, [_short(c) for c in configs]); ax.set_ylabel('Energy per episode (J)')
        ax.legend(frameon=False, loc='upper left')
        _save(fig, out, 'fig3_energy_breakdown'); made.append('fig3'); plt.close(fig)
    else:
        skipped['fig3'] = 'actuator energy model unavailable (no total-energy chart)'

    # Fig 4: success rate with Wilson 95% CI per task.
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for i, t in enumerate(tasks):
        sr = np.array([stat(per_task, c, t, 'success_rate_pct') for c in configs])
        lo = np.array([stat(per_task, c, t, 'success_ci95_low_pct') for c in configs])
        hi = np.array([stat(per_task, c, t, 'success_ci95_high_pct') for c in configs])
        ax.bar(x + (i - 0.5) * w, sr, w - 0.03, color=colors[t], label=t.capitalize(),
               yerr=[np.maximum(sr - lo, 0), np.maximum(hi - sr, 0)], error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
    ax.set_ylim(0, 105); ax.set_xticks(x, [_short(c) for c in configs])
    ax.set_ylabel('Success rate (%)')
    ax.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0))
    _save(fig, out, 'fig4_success_rate'); made.append('fig4'); plt.close(fig)

    # Fig 5: energy per success (total if the actuator model is valid, else computational only).
    key = 'total_energy_per_success_j' if all(np.isfinite(ea)) else 'computational_energy_per_success_j'
    ylabel = ('Total energy per success (J)' if key.startswith('total')
              else 'Computational energy per success (J)')
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for i, t in enumerate(tasks):
        v = np.array([stat(per_task, c, t, key) for c in configs])
        ax.bar(x + (i - 0.5) * w, np.nan_to_num(v, nan=0.0), w - 0.03, color=colors[t], label=t.capitalize())
        for xi, vi in zip(x + (i - 0.5) * w, v):
            if not np.isfinite(vi):
                ax.text(xi, 0, 'n/a', ha='center', va='bottom', fontsize=6, color=MUTED, rotation=90)
    ax.set_xticks(x, [_short(c) for c in configs]); ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0))
    ax.text(1.0, 1.02, 'n/a: no success', transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color=MUTED)
    _save(fig, out, 'fig5_energy_per_success'); made.append('fig5'); plt.close(fig)

    # Fig 6: action delay, holding time and underrun rate (three panels).
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for ax, k, label, scale in ((axes[0], 'action_delay_ms', 'Action delay (s)', 1e-3),
                                (axes[1], 'holding_time_s', 'Holding time (s)', 1.0),
                                (axes[2], 'buffer_underrun_rate', 'Buffer underrun rate (%)', 100.0)):
        for i, t in enumerate(tasks):
            m = np.array([stat(per_task, c, t, f'{k}_mean') for c in configs]) * scale
            sd = np.array([stat(per_task, c, t, f'{k}_sd') for c in configs]) * scale
            ax.bar(x + (i - 0.5) * w, m, w - 0.03, yerr=sd, color=colors[t], label=t.capitalize(),
                   error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
        ax.set_xticks(x, [_short(c) for c in configs], rotation=30, ha='right'); ax.set_ylabel(label)
    axes[0].axhline(HORIZON_TICKS / FPS, color=MUTED, linestyle='--', linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc='upper center')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out, 'fig6_action_timing', layout=False); made.append('fig6'); plt.close(fig)
    return dict(made=made, skipped=skipped)


# ------------------------------------------------------------------ driver

def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('nan' if isinstance(v, float) and math.isnan(v) else
                            ('' if v is None else v)) for k, v in r.items()})


def analyze(exp, reference=None, plots=True):
    exp = Path(exp)
    eps, reqs, cfg = load(exp)
    if not eps:
        raise ValueError(f'{exp}/episodes.csv has no episodes')
    configs = list(dict.fromkeys([c['name'] for c in cfg.get('dvfs_configs') or [] if c]
                                 + [r['configuration'] for r in eps]))
    configs = [c for c in configs if any(r['configuration'] == c for r in eps)]
    reference = reference or cfg.get('reference_config') or configs[0]
    seeds = list(dict.fromkeys(r['seed'] for r in eps))
    per_task, per_config = summaries(eps, configs, seeds)
    write_csv(exp / 'per_task_summary.csv', per_task)
    write_csv(exp / 'per_config_summary.csv', per_config)
    if len(configs) > 1:
        write_csv(exp / 'paired_tests.csv', paired_tests(eps, reference, configs))
    write_csv(exp / 'latency_horizon_analysis.csv', latency_horizon(eps, reqs))
    report = validate(exp, eps, reqs, per_task, per_config, cfg)
    model = (cfg.get('measurement') or {}).get('actuator_model', 'unavailable')
    report['actuator_model'] = model
    if plots:
        label = 'model-based estimate' if 'estimate' in str(model) else 'model'
        points = {k: v for k, v in grid_points(cfg).items() if k in configs}
        if len(points) > 8 and len({c for c, _ in points.values()}) > 1 and len({g for _, g in points.values()}) > 1:
            report['figures'] = heatmap_figures(exp, eps, reqs, per_task, per_config, points, label)
        else:
            report['figures'] = figures(exp, eps, reqs, per_task, per_config, configs, label)
    (exp / 'validation_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    failed = [c['check'] for c in report['checks'] if not c['passed']]
    print(f"[analysis] {len(eps)} episodes, {len(configs)} configs; validation "
          f"{'PASSED' if report['passed'] else 'FAILED'}" + (f"; not passed: {failed}" if failed else ''), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('experiment', type=Path)
    p.add_argument('--reference', default=None)
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--strict', action='store_true', help='Exit 1 if any error-level validation check fails')
    a = p.parse_args()
    report = analyze(a.experiment, a.reference, plots=not a.no_plots)
    sys.exit(1 if a.strict and not report['passed'] else 0)


if __name__ == '__main__':
    main()
