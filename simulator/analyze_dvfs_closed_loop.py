"""Tabulate a closed-loop DVFS run (no Isaac Sim needed).

    python simulator/analyze_dvfs_closed_loop.py outputs/dvfs_closed_loop

Writes ``table_closed_loop.csv``, ``table_closed_loop.tex`` and
``paired_tests.json`` next to ``episodes.jsonl``.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.evaluation.dvfs_study import group_summaries, paired_success_tests


def fmt(mean, std=None, digits=2):
    if mean is None:
        return '--'
    return f'{mean:.{digits}f}' if std is None else f'{mean:.{digits}f} $\\pm$ {std:.{digits}f}'


def build_table(rows, task='all'):
    summaries = [s for s in group_summaries(rows) if s['task'] == task]
    if not summaries:  # single-task run: group_summaries emits no 'all' row
        summaries = group_summaries(rows)
    first = {}
    for r in rows:
        first.setdefault(r.get('dvfs_config') or 'default', r)
    table = []
    for s in summaries:
        r = first[s['dvfs_config']]
        table.append(dict(
            config=s['dvfs_config'], task=s['task'],
            cpu_mhz=(r.get('requested_cpu_khz') or 0) / 1e3 or None,
            gpu_mhz=(r.get('requested_gpu_hz') or 0) / 1e6 or None,
            episodes=s['episodes'],
            sr_pct=s['task_success_rate_pct'],
            sr_ci_low=s['task_success_rate_ci95_low_pct'], sr_ci_high=s['task_success_rate_ci95_high_pct'],
            ct_s=s['task_completion_time_s'], ct_std_s=s['completion_time_std_s'],
            ajj=s['ajj_mean'], ajj_std=s['ajj_std'],
            collision_pct=s['collision_rate_pct'],
            energy_j=s['energy_j_mean'], energy_std_j=s['energy_j_std'],
            power_w=s['mean_power_w_mean'],
            energy_per_success_j=s['energy_per_success_j'],
            inference_ms=s['inference_ms_mean'],
            action_delay_ticks=s['action_delay_ticks_mean'],
            holding_s=s['holding_s_mean'],
            min_rtf=s['min_real_time_factor'],
            frequency_valid=s['frequency_valid_all']))
    return table


def to_latex(table):
    lines = [r'\begin{tabular}{lcccccccc}', r'\toprule',
             r'Config & CPU/GPU [MHz] & Latency [ms] & SR [\%] & CT [s] & AJJ [rad/s$^3$] & '
             r'Coll. [\%] & $E_\mathrm{ep}$ [J] & $E/\mathrm{succ.}$ [J] \\', r'\midrule']
    for t in table:
        freqs = '--' if t['cpu_mhz'] is None else f"{t['cpu_mhz']:.0f}/{t['gpu_mhz']:.0f}"
        sr = f"{t['sr_pct']:.0f} [{t['sr_ci_low']:.0f}, {t['sr_ci_high']:.0f}]"
        lines.append(' & '.join([t['config'], freqs, fmt(t['inference_ms'], digits=0), sr,
                                 fmt(t['ct_s'], t['ct_std_s'], 1), fmt(t['ajj'], t['ajj_std'], 1),
                                 f"{t['collision_pct']:.0f}", fmt(t['energy_j'], t['energy_std_j'], 0),
                                 r'$\infty$' if t['sr_pct'] == 0 else fmt(t['energy_per_success_j'], digits=0)])
                     + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='evaluate.py output directory')
    p.add_argument('--reference', default=None, help='Config for paired tests (default: first in run)')
    p.add_argument('--task', default='all', help="grasping, pushing or all")
    a = p.parse_args()
    rows = [json.loads(line) for line in (a.run/'episodes.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit('No episodes recorded')
    table = build_table(rows, a.task)
    with (a.run/'table_closed_loop.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (a.run/'table_closed_loop.tex').write_text(to_latex(table))
    reference = a.reference or (rows[0].get('dvfs_config') or 'default')
    tests = paired_success_tests(rows, reference)
    (a.run/'paired_tests.json').write_text(json.dumps(tests, indent=2) + '\n')
    for t in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
    for t in tests:
        print(f"McNemar {t['config']} vs {t['reference']}: p={t['mcnemar_p']:.4f} "
              f"(discordant {t['ref_only_success']}/{t['config_only_success']}, n={t['matched_episodes']})")
    if any(not t['frequency_valid'] for t in table):
        print('[WARN] Some episodes ran with frequencies outside tolerance; see dvfs_switches.jsonl')


if __name__ == '__main__':
    main()
