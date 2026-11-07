"""Tests for the DVFS energy evaluation: energy definitions, attribution, DVFS resolution,
request instrumentation, logging schema, aggregation and validation (synthetic data only)."""
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml

from simulator.evaluation import energy as E
from simulator.evaluation.policies import _ChunkPolicy, InferenceFailure
from jetson_orin_nx.benchmark.dvfs_control import (PowerMonitor, DvfsController, load_dvfs_configs, ROOT as JROOT,
                                                   MIN_UPDATES_PER_INFERENCE)


class Scene:
    def state(self):
        return np.zeros(9, dtype=np.float32)

    def observation(self):
        return np.zeros(9, dtype=np.float32), np.zeros((256, 256, 3), dtype=np.uint8)


class EnergyDefinitionTests(unittest.TestCase):
    def test_mechanical_energy_counts_positive_work_only(self):
        t = np.linspace(0, 1, 61)
        qd = np.zeros((61, 7)); tau = np.zeros((61, 7))
        qd[:, 0], tau[:, 0] = 0.5, 2.0      # +1 W on joint 1
        qd[:, 1], tau[:, 1] = -1.0, 3.0     # -3 W on joint 2 (braking, no regeneration)
        m = E.mechanical_energy(t, qd, tau)
        self.assertAlmostEqual(m['mechanical_energy_j'], 1.0)
        self.assertAlmostEqual(m['mechanical_energy_signed_j'], -2.0)
        self.assertAlmostEqual(m['mechanical_power_w'], 1.0)

    def test_mechanical_energy_nan_on_bad_input(self):
        self.assertTrue(math.isnan(E.mechanical_energy([0.0], np.zeros((1, 7)), np.zeros((1, 7)))['mechanical_energy_j']))

    def _model_file(self, d, **override):
        joints = {j: dict(eta=0.8, alpha_w_per_nm2=0.01, p0_w=1.0) for j in E.ARM_JOINTS}
        data = dict(estimate=True, provenance='unit test', joints=joints) | override
        path = Path(d) / 'm.yaml'
        path.write_text(yaml.safe_dump(data))
        return path

    def test_actuator_model_template_is_unavailable(self):
        with self.assertRaises(E.ActuatorModelUnavailable):
            E.load_actuator_model(ROOT / 'simulator/config/actuator_model.example.yaml')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(E.ActuatorModelUnavailable):
                E.load_actuator_model(self._model_file(d, provenance=''))

    def test_actuator_energy_formula(self):
        with tempfile.TemporaryDirectory() as d:
            model = E.load_actuator_model(self._model_file(d))
        self.assertEqual(model['label'], 'model-based estimate')
        t = np.linspace(0, 2, 121)
        qd = np.zeros((121, 7)); tau = np.zeros((121, 7))
        qd[:, 0], tau[:, 0] = 1.0, 2.0
        a = E.actuator_energy(t, qd, tau, model)
        # Joint 1: 2/0.8 + 0.01*4 = 2.54 W; P0 on all 7 joints: 7 W -> 9.54 W for 2 s.
        self.assertAlmostEqual(a['actuator_energy_j'], 2 * 9.54)
        self.assertTrue(math.isnan(E.actuator_energy(t, qd, tau, None)['actuator_energy_j']))

    def test_inference_energy_requires_every_request_attributable(self):
        s = E.inference_energy_stats([dict(energy_j=2.0, latency_s=0.5), dict(energy_j=4.0, latency_s=1.0)])
        self.assertAlmostEqual(s['inference_energy_j'], 3.0)
        self.assertAlmostEqual(s['computational_edp_js'], (1.0 + 4.0) / 2)   # mean(E_i * L_i), not E_ep * L
        self.assertAlmostEqual(s['inference_power_w'], 4.0)
        s = E.inference_energy_stats([dict(energy_j=2.0, latency_s=0.5), dict(energy_j=None, latency_s=1.0)])
        self.assertTrue(math.isnan(s['inference_energy_j']))
        self.assertTrue(math.isnan(E.inference_energy_stats([])['computational_edp_js']))

    def test_failure_reasons(self):
        self.assertEqual(E.failure_reason('success', True, True), '')
        self.assertEqual(E.failure_reason('timeout', False, False), 'no_valid_action')
        self.assertEqual(E.failure_reason('timeout', False, True), 'timeout_task_not_reached')
        self.assertEqual(E.failure_reason('object_fell', False, True), 'constraint_violation')
        self.assertEqual(E.failure_reason('inference_failure', False, True), 'inference_or_communication_failure')


class PowerAttributionTests(unittest.TestCase):
    def test_active_window_and_per_inference_energy(self):
        now = [0.0]
        mon = PowerMonitor(None, interval_s=0.01, reader=lambda: 5.0 + (now[0] * 100) % 2, clock=lambda: now[0])
        mon.begin_episode('a')
        for _ in range(301):
            mon.sample_once()
            now[0] = round(now[0] + 0.01, 10)
        mon.record_inference(0.2, 0.7, dict(obs_id=0))
        mon.record_inference(0.70, 0.705, dict(obs_id=10))   # shorter than one sample: not attributable
        now[0] = 3.0
        out = mon.end_episode('a', settle_s=0.0, active_s=2.0, include_samples=True)
        ref_active = sum(0.5 * (out['power_samples'][i][1] + out['power_samples'][i + 1][1]) * 0.01
                         for i in range(200))
        self.assertAlmostEqual(out['energy_active_j'], ref_active, places=6)
        self.assertAlmostEqual(out['active_s'], 2.0)
        first, second = out['inferences']
        self.assertEqual(first['obs_id'], 0)
        self.assertGreaterEqual(first['n_sensor_updates'], MIN_UPDATES_PER_INFERENCE)
        self.assertGreater(first['energy_j'], 0)
        self.assertIsNone(second['energy_j'])
        self.assertGreater(out['sample_rate_hz'], 90)

    def test_constant_reading_is_not_attributable(self):
        # A sensor that never refreshes inside the interval cannot attribute energy to it.
        now = [0.0]
        mon = PowerMonitor(None, interval_s=0.01, reader=lambda: 5.0, clock=lambda: now[0])
        mon.begin_episode('a')
        for _ in range(100):
            mon.sample_once(); now[0] += 0.01
        mon.record_inference(0.1, 0.5, dict(obs_id=0))
        out = mon.end_episode('a', settle_s=0.0, active_s=0.9)
        self.assertIsNone(out['inferences'][0]['energy_j'])
        self.assertAlmostEqual(out['energy_active_j'], 4.5, places=6)


def fake_board(d, cpu_list='115200 422400 1984000', gpu_list='306000000 918000000'):
    d = Path(d)
    for i in range(2):
        c = d / 'cpu' / f'cpu{i}' / 'cpufreq'; c.mkdir(parents=True)
        for k, v in dict(scaling_governor='schedutil', scaling_min_freq=115200, scaling_max_freq=1984000,
                         cpuinfo_min_freq=115200, cpuinfo_max_freq=1984000, scaling_cur_freq=1984000,
                         scaling_available_frequencies=cpu_list).items():
            (c / k).write_text(str(v))
    g = d / 'gpu'; g.mkdir()
    for k, v in dict(governor='nvhost_podgov', min_freq=306000000, max_freq=918000000, cur_freq=918000000,
                     available_frequencies=gpu_list).items():
        (g / k).write_text(str(v))
    cfg = d / 'board.yaml'
    cfg.write_text(yaml.safe_dump({'jetson_orin_nx_16g': dict(cpu_path=str(d / 'cpu'), gpu_path=str(g),
                                                              cpu_cores_num=2, ina_path=None)}))
    return DvfsController(cfg)


class DvfsResolutionTests(unittest.TestCase):
    def test_symbolic_frequencies_resolve_on_device(self):
        with tempfile.TemporaryDirectory() as d:
            ctl = fake_board(d)
            caps = ctl.capabilities()
            self.assertEqual(caps['cpu_khz_available'], [115200, 422400, 1984000])
            self.assertEqual(ctl.resolve('min', 'max'), (115200, 918000000))
            self.assertEqual(ctl.resolve('max', 'min'), (1984000, 306000000))
            with self.assertRaises(ValueError):
                ctl.resolve(500000, 'max')       # not listed by the device
            status = ctl.apply('min', 'min', settle_s=0)
            self.assertEqual((status['requested_cpu_khz'], status['requested_gpu_hz']), (115200, 306000000))
            self.assertEqual(status['requested_symbolic'], ['min', 'min'])
            self.assertIn('power_mode', status)

    def test_energy_config_file(self):
        configs = load_dvfs_configs(JROOT / 'jetson_orin_nx/benchmark/closed_loop_dvfs_energy.yaml')
        self.assertEqual([c['name'] for c in configs],
                         ['default', 'gpu_max_cpu_max', 'gpu_min_cpu_max', 'gpu_max_cpu_min', 'gpu_min_cpu_min'])
        self.assertEqual([(c['cpu_khz'], c['gpu_hz']) for c in configs[1:]],
                         [('max', 'max'), ('max', 'min'), ('min', 'max'), ('min', 'min')])
        self.assertIsNone(configs[0]['cpu_khz'])


class InstantPolicy(_ChunkPolicy):
    """Real _submit path; the request returns immediately after ``delay_s`` of wall time."""

    def __init__(self, delay_s=0.0, fail=False):
        self._init_common(25, 'continuous', 20, 0.0)
        self.delay_s, self.fail = delay_s, fail

    def _request(self, payload):
        time.sleep(self.delay_s)
        if self.fail:
            raise OSError('connection refused')
        actions = (payload['obs_id'] + np.arange(50, dtype=np.float32))[:, None].repeat(9, axis=1)
        return payload, dict(episode_id=payload['episode_id'], obs_id=payload['obs_id'], actions=actions.tolist(),
                             inference_ms=1.0, server_timing_ms=dict(total=2.0, inference=1.0)), 3.0


class InstrumentationTests(unittest.TestCase):
    def test_request_records_and_obs_age(self):
        p = InstantPolicy()
        p.reset(Scene(), 'grasping', None, None)
        for _ in range(60):
            p.act(0.0)
            time.sleep(0.002)
        p.finish()
        rec = p.requests[0]
        for a, b in (('capture', 'encoded'), ('encoded', 'sent'), ('sent', 'received'),
                     ('received', 'available'), ('available', 'first_exec')):
            self.assertLessEqual(rec[a], rec[b])
        self.assertEqual(rec['first_exec_tick'] - rec['obs_tick'], rec['release_tick'] - rec['obs_tick'])
        self.assertTrue(rec['used'])
        self.assertTrue(all(age >= 1 for age in p.obs_ages))
        self.assertEqual(len(p.tick_log), 60)
        q = p.queue_stats()
        self.assertAlmostEqual(q['holding_duration_s'], q['initial_wait_s'] + q['underrun_s'])
        rows = E.request_rows(p.requests, p.requests[0]['capture'], [], dict(episode_uid='x'))
        self.assertTrue(all(r['communication_ms'] == 1.0 for r in rows if r['used']))
        self.assertTrue(all(math.isnan(r['inference_energy_j']) for r in rows))   # no server data -> NaN

    def test_inference_failure_is_explicit(self):
        p = InstantPolicy(fail=True)
        p.reset(Scene(), 'grasping', None, None)
        with self.assertRaises(InferenceFailure):
            for _ in range(20):
                p.act(0.0)
                time.sleep(0.002)
        self.assertIn('connection refused', p.requests[0]['error'])


# ---------------------------------------------------------------- synthetic experiment (tests only)

def synthetic_experiment(d, zero_success_config='gpu_min_cpu_min'):
    d = Path(d)
    configs = ['default', 'gpu_max_cpu_max', zero_success_config]
    (d / 'experiment_config.json').write_text(json.dumps(dict(
        experiment_id='test', reference_config='default', dvfs_configs=[dict(name=c) for c in configs],
        protocol=dict(mode='continuous', realtime_pacing=True), measurement=dict(actuator_model='not configured'))))
    w = E.ExperimentWriter(d)
    rng = np.random.default_rng(0)
    for c in configs:
        for task in ('grasping', 'pushing'):
            for seed in (1, 2):
                for ep in range(3):
                    uid = f'{c}|{task}|{seed}|{ep}'
                    ident = dict(experiment_id='test', episode_uid=uid, configuration=c, task=task, seed=seed, episode=ep)
                    success = c != zero_success_config and ep != 2
                    dur = 10.0 if success else 30.0
                    t = np.arange(0, dur + 1e-9, 1 / 60)
                    p = 8.0 + rng.random(t.size)
                    w.write('power', ([uid, float(ti), float(pi)] for ti, pi in zip(t, p)))
                    qd = np.full((t.size, 7), 0.1); tau = np.full((t.size, 7), 2.0)
                    w.write('joints', ([uid, float(ti), *([0.0] * 7), *qd[i], *tau[i]] for i, ti in enumerate(t)))
                    from jetson_orin_nx.benchmark.dvfs_control import integrate
                    e_comp = integrate(list(zip(t, p)), 0.0, dur)[0]
                    req = [dict(ident, request_index=0, obs_tick=0, capture_wall_s=0.0, encoded_wall_s=0.01,
                                sent_wall_s=0.01, received_wall_s=1.0, available_wall_s=1.01, first_exec_wall_s=1.05,
                                inference_ms=900.0, rtt_ms=950.0, server_total_ms=920.0, communication_ms=30.0,
                                end_to_end_ms=1000.0, obs_to_action_ms=1050.0, server_inference_start_s=0.05,
                                server_inference_end_s=0.95, inference_energy_j=9.0, action_delay_ticks=21,
                                used=True, drained=False, error='')]
                    result = dict(success=success, failure_reason='' if success else 'timeout_task_not_reached',
                                  reason='success' if success else 'timeout', collision=False, duration_s=dur,
                                  completion_time_s=dur if success else None, wall_time_s=dur,
                                  real_time_factor=1.0, holding_duration_s=1.0, buffer_underrun_rate=0.1)
                    energy = dict(energy_active_j=e_comp, active_s=dur, energy_j=e_comp + 1, inferences=[{}],
                                  sample_rate_hz=60.0)
                    w.write('episodes', [E.episode_row(ident, result, energy, E.mechanical_energy(t, qd, tau),
                                                       E.actuator_energy(t, qd, tau, None), req, {}, False)])
                    w.write('inference', req)
    w.close()
    return d


class AnalysisTests(unittest.TestCase):
    def test_analysis_and_validation_on_synthetic_run(self):
        from simulator.analyze_energy import analyze, read_csv
        with tempfile.TemporaryDirectory() as d:
            report = analyze(synthetic_experiment(d), plots=True)
            checks = {c['check']: c for c in report['checks']}
            self.assertTrue(report['passed'], [c for c in report['checks'] if not c['passed']])
            self.assertTrue(checks['E_comp reproduced from power_samples.csv.gz']['passed'])
            self.assertTrue(checks['E_mech reproduced from joint_samples.csv.gz']['passed'])
            per_task = {(r['configuration'], r['task']): r for r in read_csv(Path(d) / 'per_task_summary.csv')}
            zero = per_task[('gpu_min_cpu_min', 'grasping')]
            self.assertEqual(zero['successes'], '0')
            self.assertEqual(zero['computational_energy_per_success_j'], 'nan')   # never 0
            ref = per_task[('default', 'grasping')]
            self.assertEqual((ref['episodes'], ref['successes']), ('6', '4'))
            self.assertAlmostEqual(float(ref['success_rate_pct']), 400 / 6)
            # Energy per success includes the failed episodes' energy.
            eps = [r for r in read_csv(Path(d) / 'episodes.csv') if r['configuration'] == 'default'
                   and r['task'] == 'grasping']
            self.assertAlmostEqual(float(ref['computational_energy_per_success_j']),
                                   sum(float(r['computational_energy_j']) for r in eps) / 4)
            self.assertTrue(math.isnan(float(eps[0]['total_energy_j'])))           # no actuator model
            self.assertIn('fig3', report['figures']['skipped'])                    # no total-energy chart
            self.assertTrue((Path(d) / 'figures/fig4_success_rate.pdf').exists())

    def test_validation_flags_duplicates_and_zero_fill(self):
        from simulator.analyze_energy import analyze
        with tempfile.TemporaryDirectory() as d:
            synthetic_experiment(d)
            path = Path(d) / 'episodes.csv'
            lines = path.read_text().splitlines()
            header = lines[0].split(',')
            row = lines[1].split(',')
            row[header.index('inference_energy_j')] = '0'
            path.write_text('\n'.join(lines + [','.join(row)]) + '\n')
            report = analyze(d, plots=False)
            failed = {c['check'] for c in report['checks'] if not c['passed']}
            self.assertFalse(report['passed'])
            self.assertIn('no duplicated episode IDs', failed)
            self.assertIn('no exact zeros in inference_energy_j (missing values must be NaN)', failed)


if __name__ == '__main__':
    unittest.main()
