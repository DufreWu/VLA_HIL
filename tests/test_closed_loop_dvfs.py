from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml

from simulator.evaluation.policies import _ChunkPolicy, release_delay_ticks
from simulator.evaluation.dvfs_study import (rotated_order, wilson_ci, mcnemar_exact,
                                             group_summaries, paired_success_tests, episode_noise_seed,
                                             dvfs_table, table_markdown, table_latex)
from jetson_orin_nx.benchmark.dvfs_control import (PowerMonitor, DvfsController, integrate,
                                                   load_dvfs_configs, CLOSED_LOOP_CONFIG)


class Scene:
    def state(self):
        return np.zeros(9, dtype=np.float32)

    def observation(self):
        return np.zeros(9, dtype=np.float32), np.zeros((256, 256, 3), dtype=np.uint8)


class FakeLatencyPolicy(_ChunkPolicy):
    """Inference completes instantly in wall time but reports a fixed capture-to-arrival delay."""

    def __init__(self, delay_s, **kw):
        self._init_common(25, 'continuous', 20, kw.get('starvation_abort_s', 5.0))
        self.delay_s = delay_s
        self.payloads = []

    def _submit(self):
        payload = self._build_request(self.tick)
        self.payloads.append(payload)
        # Chunk slot i has value source_tick + i in every joint.
        actions = (payload['obs_id'] + np.arange(50, dtype=np.float32))[:, None].repeat(9, axis=1)
        response = dict(episode_id=payload['episode_id'], obs_id=payload['obs_id'],
                        actions=actions.tolist(), inference_ms=1.0)
        f = Future()
        f.set_result((payload, response, 1.0, self.delay_s))
        self.future = f


class ContinuousPolicyTests(unittest.TestCase):
    def test_release_delay_ticks(self):
        self.assertEqual(release_delay_ticks(0.788), 16)
        self.assertEqual(release_delay_ticks(0.05), 1)
        self.assertEqual(release_delay_ticks(0.0), 1)
        self.assertEqual(release_delay_ticks(3.122), 63)

    def test_chunk_released_after_measured_delay_and_aligned_to_current_slot(self):
        policy = FakeLatencyPolicy(0.788)
        policy.reset(Scene(), 'grasping', None, None)
        targets = [policy.act(0.0)[0] for _ in range(20)]
        # Holds the initial pose for 16 ticks, then executes slot 16 of the chunk from tick 0.
        self.assertTrue(all(t == 0.0 for t in targets[:16]))
        self.assertEqual(targets[16:], [16.0, 17.0, 18.0, 19.0])
        self.assertEqual(policy.action_delays, [16])

    def test_queue_executes_while_request_in_flight_and_is_replaced(self):
        policy = FakeLatencyPolicy(0.25)  # 5 ticks
        policy.reset(Scene(), 'grasping', None, None)
        targets = [policy.act(0.0)[0] for _ in range(40)]
        # Never falls back to a hold after the first chunk: values track the tick index.
        self.assertEqual(targets[5:], [float(t) for t in range(5, 40)])
        self.assertGreater(len(policy.payloads), 2)
        self.assertEqual(policy.starvation_events, 0)

    def test_chunk_longer_than_horizon_never_acts(self):
        # A request is always in flight, so this runs to the episode timeout
        # (holding the pose) instead of tripping the starvation abort.
        policy = FakeLatencyPolicy(3.122, starvation_abort_s=5)
        policy.reset(Scene(), 'grasping', None, None)
        targets = [policy.act(0.0)[0] for _ in range(300)]
        self.assertTrue(all(t == 0.0 for t in targets))
        self.assertGreater(policy.dropped_slots, 0)
        self.assertEqual(policy.action_delays[0], 63)
        self.assertGreater(policy.inflight_hold_ticks, 250)

    def test_noise_seed_identical_across_configs(self):
        seeds = []
        for _ in range(2):
            policy = FakeLatencyPolicy(0.5)
            policy.reset(Scene(), 'grasping', None, None, episode_seed=episode_noise_seed(42, 1, 3))
            for _ in range(60):
                policy.act(0.0)
            seeds.append([p['noise_seed'] for p in policy.payloads])
        self.assertEqual(seeds[0], seeds[1])
        self.assertEqual(len(set(seeds[0])), len(seeds[0]))


class PowerTests(unittest.TestCase):
    def test_integrate_clips_linearly(self):
        samples = [(0.0, 4.0), (1.0, 6.0), (2.0, 6.0)]
        energy, _ = integrate(samples, 0.5, 1.5)
        self.assertAlmostEqual(energy, 0.5 * (5.0 + 6.0) * 0.5 + 6.0 * 0.5)
        self.assertEqual(integrate(samples, -1, 1)[0], None)

    def test_episode_energy_with_fake_clock(self):
        now = [0.0]
        mon = PowerMonitor(None, interval_s=0.01, reader=lambda: 5.0, clock=lambda: now[0])
        mon.begin_episode('a')
        for _ in range(101):
            mon.sample_once()
            now[0] += 0.01
        mon.record_inference(0.2, 0.7)
        now[0] = 0.9
        out = mon.end_episode('a', settle_s=0.0)
        self.assertAlmostEqual(out['energy_j'], 4.5, places=6)
        self.assertAlmostEqual(out['mean_power_w'], 5.0, places=6)
        self.assertAlmostEqual(out['inference_energy_j'], 2.5, places=6)
        with self.assertRaises(ValueError):
            mon.end_episode('a')

    def test_episode_reports_mean_frequency(self):
        now = [0.0]
        freqs = iter([(1000.0, 300.0), (2000.0, 900.0), (3000.0, 600.0)])
        mon = PowerMonitor(None, interval_s=0.01, reader=lambda: 5.0, clock=lambda: now[0],
                           freq_reader=lambda: next(freqs), freq_interval_s=0.05)
        mon.begin_episode('a')
        for _ in range(10):  # frequency sampled at samples 0 and 5 ...
            mon.sample_once()
            now[0] += 0.01
        out = mon.end_episode('a', settle_s=0.0)  # ... and 10 (closing sample)
        self.assertEqual(out['n_freq_samples'], 3)
        self.assertEqual((out['mean_cpu_khz'], out['mean_gpu_hz']), (2000.0, 600.0))


class DvfsControllerTests(unittest.TestCase):
    def test_apply_and_restore_on_fake_sysfs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for i in range(2):
                c = d/'cpu'/f'cpu{i}'/'cpufreq'; c.mkdir(parents=True)
                for k, v in dict(scaling_governor='schedutil', scaling_min_freq=115200,
                                 scaling_max_freq=1984000, cpuinfo_min_freq=115200,
                                 cpuinfo_max_freq=1984000, scaling_setspeed=0,
                                 scaling_cur_freq=1984000).items():
                    (c/k).write_text(str(v))
            g = d/'gpu'; g.mkdir()
            for k, v in dict(governor='nvhost_podgov', min_freq=306000000, max_freq=918000000,
                             cur_freq=918000000,
                             available_frequencies='306000000 408000000 918000000').items():
                (g/k).write_text(str(v))
            cfg = d/'board.yaml'
            cfg.write_text(yaml.safe_dump({'jetson_orin_nx_16g': dict(
                cpu_path=str(d/'cpu'), gpu_path=str(g), cpu_cores_num=2, ina_path=None)}))
            ctl = DvfsController(cfg)
            ctl.take_snapshot()
            status = ctl.apply(422400, 408000000, settle_s=0)
            self.assertEqual((d/'cpu/cpu1/cpufreq/scaling_max_freq').read_text(), '422400')
            self.assertEqual((g/'min_freq').read_text(), '408000000')
            self.assertFalse(status['frequency_valid'])  # fake cur_freq does not follow
            ctl.restore()
            self.assertEqual((d/'cpu/cpu0/cpufreq/scaling_governor').read_text(), 'schedutil')
            self.assertEqual((d/'cpu/cpu0/cpufreq/scaling_min_freq').read_text(), '115200')
            self.assertEqual((d/'cpu/cpu0/cpufreq/scaling_max_freq').read_text(), '1984000')
            self.assertEqual((g/'governor').read_text(), 'nvhost_podgov')
            self.assertEqual((g/'min_freq').read_text(), '306000000')

    def test_shipped_closed_loop_configs(self):
        configs = load_dvfs_configs(CLOSED_LOOP_CONFIG)
        self.assertEqual([c['name'] for c in configs],
                         ['default', 'gpu_max_cpu_max', 'gpu_max_cpu_min', 'gpu_min_cpu_max', 'gpu_min_cpu_min'])
        self.assertEqual((configs[0]['cpu_khz'], configs[0]['cpu_governor'], configs[0]['gpu_governor']),
                         (None, 'schedutil', 'nvhost_podgov'))
        # Corners are the endpoints of the board's frequency lists.
        board = yaml.safe_load((ROOT/'jetson_orin_nx/benchmark/jetson_orin_nx.yaml').read_text())['jetson_orin_nx_16g']
        corners = {(c['cpu_khz'], c['gpu_hz']) for c in configs[1:]}
        self.assertEqual(corners, {(cpu, gpu) for cpu in (min(board['cpu_freqs']), max(board['cpu_freqs']))
                                   for gpu in (min(board['gpu_freqs']), max(board['gpu_freqs']))})
        self.assertEqual(load_dvfs_configs(CLOSED_LOOP_CONFIG, ['gpu_min_cpu_min'])[0]['cpu_khz'], 422400)
        with self.assertRaises(ValueError):
            load_dvfs_configs(CLOSED_LOOP_CONFIG, ['turbo'])

    def test_governor_config_rejects_mixed_fields(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'c.yaml'
            path.write_text(yaml.safe_dump({'configs': [dict(name='x', governor='default', cpu_khz=1)]}))
            with self.assertRaises(ValueError):
                load_dvfs_configs(path)

    def test_apply_governor_on_fake_sysfs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for i in range(2):
                c = d/'cpu'/f'cpu{i}'/'cpufreq'; c.mkdir(parents=True)
                for k, v in dict(scaling_governor='userspace', scaling_min_freq=422400,
                                 scaling_max_freq=422400, cpuinfo_min_freq=115200,
                                 cpuinfo_max_freq=1984000, scaling_cur_freq=1000000 + i * 200000).items():
                    (c/k).write_text(str(v))
            g = d/'gpu'; g.mkdir()
            for k, v in dict(governor='userspace', min_freq=306000000, max_freq=306000000,
                             cur_freq=612000000, available_frequencies='306000000 918000000').items():
                (g/k).write_text(str(v))
            cfg = d/'board.yaml'
            cfg.write_text(yaml.safe_dump({'jetson_orin_nx_16g': dict(
                cpu_path=str(d/'cpu'), gpu_path=str(g), cpu_cores_num=2, ina_path=None)}))
            ctl = DvfsController(cfg)
            status = ctl.apply_governor('schedutil', 'nvhost_podgov', settle_s=0)
            self.assertTrue(status['frequency_valid'])
            self.assertEqual((d/'cpu/cpu1/cpufreq/scaling_min_freq').read_text(), '115200')
            self.assertEqual((d/'cpu/cpu1/cpufreq/scaling_max_freq').read_text(), '1984000')
            self.assertEqual(((g/'min_freq').read_text(), (g/'max_freq').read_text()), ('306000000', '918000000'))
            self.assertEqual(ctl.read_freqs(), (1100000.0, 612000000))


class StudyStatisticsTests(unittest.TestCase):
    def test_rotated_order_is_latin_square(self):
        cfgs = ['a', 'b', 'c']
        orders = [rotated_order(cfgs, e) for e in range(3)]
        for pos in range(3):
            self.assertEqual(sorted(o[pos] for o in orders), cfgs)

    def test_wilson_and_mcnemar(self):
        low, high = wilson_ci(10, 20)
        self.assertAlmostEqual(low, 29.93, places=1)
        self.assertAlmostEqual(high, 70.07, places=1)
        self.assertEqual(wilson_ci(0, 10)[0], 0.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 0.03125)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_group_summaries_and_paired_tests(self):
        def row(cfg, ep, ok, energy):
            return dict(task='grasping', episode=ep, dvfs_config=cfg, success=ok, collision=False,
                        completion_time_s=10.0 if ok else None, duration_s=10.0 if ok else 30.0,
                        ajj=1.0, energy_j=energy, mean_power_w=energy / 30, real_time_factor=1.0)
        rows = [row('performance', e, True, 240.0) for e in range(4)]
        rows += [row('powersave', e, e == 0, 170.0) for e in range(4)]
        s = {x['dvfs_config']: x for x in group_summaries(rows)}
        self.assertEqual(s['performance']['task_success_rate_pct'], 100.0)
        self.assertAlmostEqual(s['performance']['energy_per_success_j'], 240.0)
        self.assertAlmostEqual(s['powersave']['energy_per_success_j'], 680.0)
        test = paired_success_tests(rows, 'performance')[0]
        self.assertEqual((test['ref_only_success'], test['config_only_success']), (3, 0))
        json.dumps(group_summaries(rows), allow_nan=False)

    def test_paired_tests_are_per_task(self):
        # Opposite effects per task cancel when pooled.
        rows = []
        for ep in range(8):
            rows += [dict(task='grasping', episode=ep, dvfs_config='ref', success=True),
                     dict(task='grasping', episode=ep, dvfs_config='low', success=False),
                     dict(task='pushing', episode=ep, dvfs_config='ref', success=False),
                     dict(task='pushing', episode=ep, dvfs_config='low', success=True)]
        tests = {t['task']: t for t in paired_success_tests(rows, 'ref')}
        self.assertLess(tests['grasping']['mcnemar_p'], 0.01)
        self.assertLess(tests['pushing']['mcnemar_p'], 0.01)
        self.assertEqual(tests['all']['mcnemar_p'], 1.0)

    def test_dvfs_table(self):
        configs = [dict(name='default', cpu_khz=None, gpu_hz=None, cpu_governor='schedutil',
                        gpu_governor='nvhost_podgov'),
                   dict(name='gpu_max_cpu_max', cpu_khz=1984000, gpu_hz=918000000)]
        def row(cfg, task, ep, ok):
            return dict(task=task, episode=ep, dvfs_config=cfg, success=ok, collision=False,
                        completion_time_s=10.0 if ok else None, energy_j=200.0, mean_power_w=7.0,
                        task_progress=1.0 if ok else 0.3, tip_over=0.0 if task == 'pushing' else None,
                        ajj=50.0, mean_inference_ms=800.0, mean_action_delay_ticks=20.0,
                        mean_cpu_khz=1500000.0, mean_gpu_hz=600000000.0, real_time_factor=1.0,
                        dvfs_frequency_valid=True)
        rows = [row(c, t, e, e < 2) for c in ('default', 'gpu_max_cpu_max')
                for t in ('grasping', 'pushing') for e in range(4)]
        table = dvfs_table(rows, configs, 'default')
        self.assertEqual([(r['config'], r['task']) for r in table],
                         [('default', 'grasping'), ('default', 'pushing'),
                          ('gpu_max_cpu_max', 'grasping'), ('gpu_max_cpu_max', 'pushing')])
        ref, fixed = table[0], table[2]
        self.assertEqual((ref['cpu_mode'], ref['cpu_mhz'], ref['gpu_mhz']), ('schedutil', 1500.0, 600.0))
        self.assertEqual((fixed['cpu_mhz'], fixed['gpu_mhz']), (1984.0, 918.0))
        self.assertIsNone(ref['mcnemar_p_vs_reference'])
        self.assertEqual(fixed['mcnemar_p_vs_reference'], 1.0)
        self.assertEqual((fixed['success_rate_pct'], fixed['energy_per_success_j']), (50.0, 400.0))
        self.assertIsNone(table[0]['tip_over_rate_pct'])
        md, tex = table_markdown(table, 'default'), table_latex(table, 'default')
        self.assertIn('schedutil (1500)', md)
        self.assertEqual(sum(l.startswith('| ') for l in md.splitlines()), 5)  # header + 4 rows
        self.assertIn(r'gpu\_max\_cpu\_max', tex)
        json.dumps(table, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
