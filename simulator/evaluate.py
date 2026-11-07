r"""Run with Isaac Sim 6.0 Python: python simulator/evaluate.py --task both.

Closed-loop DVFS study (Jetson server started with --dvfs-control):
  python simulator/evaluate.py --policy remote --server http://NX_IP:8080 --mode continuous \
      --task both --episodes 20 --execute-steps 25 --prefetch 20 \
      --dvfs-configs jetson_orin_nx/benchmark/closed_loop_dvfs.yaml --energy --headless
  Tables: <output>/table_dvfs.{csv,md,tex} (one row per config and task).

DVFS energy evaluation (computational, mechanical and actuator energy; see simulator/DVFS_ENERGY.md):
  python simulator/evaluate.py --policy remote --server http://NX_IP:8080 --mode continuous \
      --task both --seeds 41 42 43 44 45 --episodes 10 --execute-steps 25 --prefetch 20 \
      --dvfs-configs jetson_orin_nx/benchmark/closed_loop_dvfs_energy.yaml --energy --headless \
      [--actuator-model simulator/config/actuator_model.yaml]
  Every run writes episodes.csv, inference_requests.csv, power_samples.csv.gz, joint_samples.csv.gz,
  action_ticks.csv.gz and experiment_config.json; analysis: python simulator/analyze_energy.py <output>.
"""
import argparse
import csv
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.evaluation.metrics import TaskConfig, EpisodeMetrics, average_joint_jerk, summarize
from simulator.evaluation.dvfs_study import (rotated_order, episode_noise_seed, group_summaries,
                                             paired_success_tests, dvfs_table, table_markdown, table_latex)
from simulator.posthoc_metrics import task_progress, tip_over


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except TypeError:
            pass
    raise TypeError(f'Object of type {value.__class__.__name__} is not JSON serializable')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default)+'\n')

def compute_average_joint_jerk(joint_positions, dt_s):
    """Mean L2 norm of 7-DOF joint jerk, in rad/s^3.

    Positions are sampled uniformly at physics frequency. The forward
    third finite difference approximates d^3q/dt^3 and is averaged over time.
    Gripper joints must not be included.
    """
    import numpy as np

    q = np.asarray(joint_positions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError('joint_positions must have shape (N, 7)')
    if not np.all(np.isfinite(q)):
        raise ValueError('Joint positions contain non-finite values')
    if dt_s <= 0:
        raise ValueError('dt_s must be positive')
    if q.shape[0] < 4:
        return None

    jerk = np.diff(q, n=3, axis=0) / (dt_s ** 3)
    return float(np.linalg.norm(jerk, axis=1).mean())


def add_ajj_to_summary(summary, episode_rows):
    """Add per-task mean and sample std, each episode weighted equally."""
    values = [float(r['ajj']) for r in episode_rows
              if r.get('ajj') is not None]
    summary = dict(summary)
    summary['ajj_mean_rad_s3'] = statistics.mean(values) if values else None
    summary['ajj_std_rad_s3'] = (statistics.stdev(values) if len(values) > 1
                                else (0.0 if values else None))
    return summary



def write_dvfs_tables(output, rows, dvfs_configs, reference):
    """Per-(config, task) result table as CSV, Markdown and LaTeX; rewritten after every episode."""
    table = dvfs_table(rows, dvfs_configs, reference)
    with (output/'table_dvfs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    (output/'table_dvfs.md').write_text(table_markdown(table, reference))
    (output/'table_dvfs.tex').write_text(table_latex(table, reference))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=['grasping', 'pushing', 'both'], default='both')
    p.add_argument('--episodes', type=int, default=10, help='Episodes per task')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='Run --episodes episodes for each of these seeds (overrides --seed)')
    p.add_argument('--timeout', type=float, default=30.)
    p.add_argument('--hold', type=float, default=1.)
    p.add_argument('--collision-force', type=float, default=1.)
    p.add_argument('--output', type=Path, default=Path('outputs/evaluation'))
    p.add_argument('--headless', action='store_true')
    p.add_argument('--save-preview', action='store_true', help='Save camera GIFs at 2 Hz for motion inspection')
    p.add_argument('--policy', default='scripted', help='scripted, remote, local, or module:Class with reset/act methods')
    p.add_argument('--server', default='http://127.0.0.1:8080')
    p.add_argument('--model', type=Path, default=None, help='Local model directory for --policy local')
    p.add_argument('--device', default='cpu', help='Model device for --policy local (cpu/cuda)')
    p.add_argument('--mode', choices=['sync', 'continuous'], default='sync')
    p.add_argument('--execute-steps', type=int, default=10)
    p.add_argument('--prefetch', type=int, default=5)
    p.add_argument('--allow-mock', action='store_true')
    p.add_argument('--no-realtime', action='store_true',
                   help='Continuous mode only: do not pace simulation to wall-clock time. Without pacing, '
                        'inference latency in simulation ticks is distorted by the real-time factor.')
    p.add_argument('--starvation-abort-s', type=float, default=None,
                   help='Fail an episode after this many seconds with no usable action and no request in '
                        'flight; 0 disables. Default 5, or 0 with --dvfs-configs.')
    p.add_argument('--dvfs-configs', type=Path, default=None,
                   help='YAML with named Jetson DVFS configs; every episode is run once per config from '
                        'the same initial state. Needs --policy remote and a server with --dvfs-control.')
    p.add_argument('--dvfs-names', nargs='+', default=None, help='Subset/order of configs (first = reference)')
    p.add_argument('--dvfs-settle-s', type=float, default=2.0)
    p.add_argument('--dvfs-warmup', type=int, default=3, help='Server-side warm-up inferences after a switch')
    p.add_argument('--energy', action='store_true',
                   help='Record per-episode Jetson VDD_IN energy (server needs --energy or --dvfs-control)')
    p.add_argument('--actuator-model', type=Path, default=None,
                   help='YAML with per-joint eta, alpha and P0 plus provenance; without it E_act and '
                        'E_total are reported as unavailable (NaN)')
    p.add_argument('--experiment-id', default=None, help='Default: UTC timestamp + random suffix')
    p.add_argument('--drive-check-episodes', type=int, default=2,
                   help='Episodes in which the joint-effort signal is cross-checked against the PD drive law')
    a = p.parse_args()
    if a.episodes < 1 or not 1 <= a.execute_steps <= 50 or not 0 <= a.prefetch < a.execute_steps:
        p.error('episodes must be positive; execute-steps must be between 1 and 50; prefetch must be in [0, execute_steps)')
    if (a.dvfs_configs or a.energy) and a.policy != 'remote':
        p.error('--dvfs-configs/--energy require --policy remote (the Jetson policy server)')
    if a.dvfs_configs and (a.mode != 'continuous' or a.no_realtime):
        p.error('--dvfs-configs requires --mode continuous with wall-clock pacing (no --no-realtime): '
                'physics must not wait for inference and energy windows must match simulation time')
    if a.starvation_abort_s is None:
        a.starvation_abort_s = 0.0 if a.dvfs_configs else 5.0
    try:
        config = TaskConfig(timeout_s=a.timeout, hold_s=a.hold, collision_force_n=a.collision_force)
    except ValueError as exc:
        p.error(str(exc))
    a.seeds = list(dict.fromkeys(a.seeds)) if a.seeds else [a.seed]
    if a.experiment_id is None:
        import datetime
        import uuid
        a.experiment_id = (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                           + '_' + uuid.uuid4().hex[:6])
    return a, config


def summarize_runs(rows, tasks, seeds):
    """Overall per-task summary plus per-(task, seed) summaries."""
    overall, by_seed = {}, []
    for t in tasks:
        task_rows = [r for r in rows if r['task'] == t]
        if not task_rows:
            continue
        s = add_ajj_to_summary(summarize(task_rows), task_rows)
        rates = []
        for seed in seeds:
            seed_rows = [r for r in task_rows if r['seed'] == seed]
            if seed_rows:
                seed_summary = add_ajj_to_summary(summarize(seed_rows), seed_rows)
                rates.append(seed_summary['task_success_rate_pct'])
                by_seed.append(dict(task=t, seed=seed, **seed_summary))
        s['num_seeds'] = len(rates)
        s['success_rate_seed_mean_pct'] = statistics.mean(rates)
        s['success_rate_seed_std_pct'] = statistics.stdev(rates) if len(rates) > 1 else 0.0
        overall[t] = s
    return overall, by_seed


def main():
    args, config = parse_args()
    import numpy as np
    from simulator.evaluation.contacts import ContactMonitor, FINGERS
    from simulator.evaluation.policies import ActionStarvation, InferenceFailure
    from simulator.evaluation import energy as E
    dvfs_configs = [None]
    if args.dvfs_configs is not None:
        from jetson_orin_nx.benchmark.dvfs_control import load_dvfs_configs
        dvfs_configs = load_dvfs_configs(args.dvfs_configs, args.dvfs_names)
    reference = dvfs_configs[0]['name'] if dvfs_configs[0] is not None else None
    device_caps = None
    actuator_model, actuator_status = None, 'not configured (--actuator-model not given)'
    if args.actuator_model is not None:
        from simulator.evaluation.energy import load_actuator_model, ActuatorModelUnavailable
        try:
            actuator_model = load_actuator_model(args.actuator_model)
            actuator_status = actuator_model['label']
        except ActuatorModelUnavailable as exc:
            actuator_status = f'unavailable: {exc}'
            print(f'[WARN] Actuator model {actuator_status}; E_act and E_total will be NaN.', flush=True)
    if args.policy == 'remote':
        from simulator.evaluation.policies import RemotePolicy
        from simulator.franka.run_hil import request_json
        server = args.server.rstrip('/')
        policy = RemotePolicy(args.server, args.execute_steps, args.allow_mock,
                              mode=args.mode, prefetch=args.prefetch,
                              starvation_abort_s=args.starvation_abort_s)
        if args.dvfs_configs and not policy.health.get('dvfs_control'):
            raise RuntimeError('Policy server was not started with --dvfs-control')
        if any(c is not None and c.get('cpu_khz') is None for c in dvfs_configs) \
                and 'governor' not in policy.health.get('dvfs_modes', []):
            raise RuntimeError('The policy server does not support governor DVFS configs (e.g. "default"); '
                               'update jetson_orin_nx/ on the Jetson to this version and restart the server.')
        if args.energy and not policy.health.get('energy'):
            raise RuntimeError('Policy server was not started with --energy/--dvfs-control')
        features = set(policy.health.get('features', []))
        if args.dvfs_configs:
            symbolic = [c['name'] for c in dvfs_configs if isinstance(c.get('cpu_khz'), str)
                        or isinstance(c.get('gpu_hz'), str)]
            if symbolic and 'symbolic_frequencies' not in features:
                raise RuntimeError(f'Configs {symbolic} use min/max frequencies, which this policy server cannot '
                                   'resolve; update jetson_orin_nx/ on the Jetson and restart the server.')
            if 'dvfs_capabilities' in features:
                device_caps, _ = request_json(server+'/dvfs/capabilities')
                for c in dvfs_configs:  # Resolve min/max on the device; unsupported values fail here.
                    if c['cpu_khz'] is not None:
                        resolved, _ = request_json(server+'/dvfs', dict(resolve_only=True, cpu_khz=c['cpu_khz'],
                                                                         gpu_hz=c['gpu_hz']))
                        c['resolved_cpu_khz'], c['resolved_gpu_hz'] = resolved['cpu_khz'], resolved['gpu_hz']
                print('[DVFS] device frequencies: CPU kHz', device_caps.get('cpu_khz_available') or
                      device_caps.get('cpu_khz_range'), '| GPU Hz', device_caps.get('gpu_hz_available'),
                      '| power mode:', device_caps.get('power_mode'), flush=True)
                for c in dvfs_configs:
                    print(f"[DVFS] {c['name']}: " + (f"CPU {c['resolved_cpu_khz']} kHz, GPU {c['resolved_gpu_hz']} Hz"
                          if c['cpu_khz'] is not None else f"governors {c['cpu_governor']}/{c['gpu_governor']}"),
                          flush=True)
        if args.energy and 'request_timing' not in features:
            print('[WARN] Policy server predates per-request timing/energy; inference energy, EDP and '
                  'E_comp (active window) will be NaN. Update jetson_orin_nx/ on the Jetson.', flush=True)
    elif args.policy == 'local':
        if args.model is None:
            raise ValueError('--model is required when --policy local is used')
        from simulator.evaluation.policies import LocalPolicy
        policy = LocalPolicy(args.model, device=args.device, execute_steps=args.execute_steps,
                             mode=args.mode, prefetch=args.prefetch,
                             starvation_abort_s=args.starvation_abort_s)
    elif args.policy == 'scripted':
        from simulator.evaluation.controllers import ScriptedPolicy
        policy = ScriptedPolicy()
    else:
        module, name = args.policy.rsplit(':', 1)
        policy = getattr(importlib.import_module(module), name)()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'run_config.json', dict(arguments={**vars(args), 'output': str(args.output)},
              metrics=asdict(config), physics_hz=60, action_hz=20, isaac_sim_target='6.0.0',
              policy_health=getattr(policy, 'health', None),
              ajj_definition='mean_t norm_2(delta^3(q_t)/dt^3), arm joints 1-7, rad/s^3',
              dvfs_configs=dvfs_configs))
    from urllib.parse import urlparse
    host = urlparse(args.server).hostname if args.policy == 'remote' else None
    save_json(args.output/'experiment_config.json', dict(
        experiment_id=args.experiment_id, created_utc=args.experiment_id.split('_')[0],
        arguments={k: v for k, v in vars(args).items() if k != 'server'} | dict(output=str(args.output)),
        # Address omitted (double-blind); only whether inference ran on this host.
        server=dict(loopback=host in ('127.0.0.1', 'localhost', '::1'), port=urlparse(args.server).port)
        if host else None,
        policy_health=getattr(policy, 'health', None), device_capabilities=device_caps,
        dvfs_configs=dvfs_configs, reference_config=reference,
        protocol=dict(physics_hz=60, control_hz=20, chunk_size=50, horizon_s=2.5, timeout_s=config.timeout_s,
                      hold_s=config.hold_s, execute_steps=args.execute_steps, prefetch=args.prefetch,
                      mode=args.mode, realtime_pacing=args.mode == 'continuous' and not args.no_realtime,
                      seeds=args.seeds, episodes_per_seed=args.episodes, config_order='cyclic Latin square per episode',
                      warmup=f'{args.dvfs_warmup} server-side inferences after each DVFS switch, excluded'),
        metrics=asdict(config),
        measurement=dict(
            power_sensor='Jetson INA3221 channel 1 (VDD_IN, module input), hwmon in1_input*curr1_input',
            power_sampling='server thread, --power-interval-s on the Jetson; achieved rate per episode in '
                           'episodes.csv (power_sample_rate_hz, power_sensor_update_rate_hz)',
            time_sync='No cross-host clock subtraction. Server energy window starts when /episode/start is '
                      'handled; the client starts its wall clock on the reply and sends its wall duration '
                      '(active_s) with /episode/stop. Offset bound: episode_start_rtt_ms.',
            joint_effort='PhysX projected joint force (get_dof_projected_joint_forces) at 60 Hz, arm joints 1-7',
            actuator_model=actuator_status,
            actuator_parameters=actuator_model,
            energy_definitions=__import__('simulator.evaluation.energy', fromlist=['x']).__doc__)))
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    monitor = ContactMonitor(config.collision_force_n)
    scene = None
    rows = []
    exit_code = 0
    current_dvfs = {'name': None, 'status': None}
    exp_writer = E.ExperimentWriter(args.output)
    drive_checks = []
    consecutive_failures = 0

    def switch_dvfs(cfg):
        if cfg is None or current_dvfs['name'] == cfg['name']:
            return current_dvfs['status']
        body = dict(name=cfg['name'], settle_s=args.dvfs_settle_s, warmup=args.dvfs_warmup)
        if cfg.get('cpu_khz') is None:
            body.update(cpu_governor=cfg['cpu_governor'], gpu_governor=cfg['gpu_governor'])
        else:
            body.update(cpu_khz=cfg['cpu_khz'], gpu_hz=cfg['gpu_hz'])
        status, _ = request_json(server+'/dvfs', body, timeout=120)
        if not status.get('frequency_valid'):
            print(f"[WARN] DVFS {cfg['name']} not reached: {status}", flush=True)
        current_dvfs.update(name=cfg['name'], status=status)
        with (args.output/'dvfs_switches.jsonl').open('a') as f:
            f.write(json.dumps(dict(status, wall_time=time.time()), allow_nan=False)+'\n')
        return status

    try:
        from simulator.franka.scene import FrankaScene, array
        scene = FrankaScene(app, before_simulation=monitor.install)
        # Settling must produce cube/table contact reports; otherwise do not publish a zero collision rate.
        if monitor.report_count == 0:
            raise RuntimeError('No contact reports during settling. Verify PhysX contact reporting before evaluation.')
        tasks = ['grasping', 'pushing'] if args.task == 'both' else [args.task]
        multi_seed = len(args.seeds) > 1
        for task, seed in ((t, s) for t in tasks for s in args.seeds):
            rng = np.random.default_rng(seed)
            prefix = f'{task}_seed{seed}' if multi_seed else task
            for episode in range(args.episodes):
                initial = [float(rng.uniform(.42, .48)), float(rng.uniform(-.25, -.15)), .0258]
                target = [initial[0], initial[1]+.20, .0258]
                requested_initial = initial
                for order_index, dvfs_cfg in enumerate(rotated_order(dvfs_configs, episode)):
                    initial = requested_initial
                    dvfs_status = switch_dvfs(dvfs_cfg)
                    dvfs_tag = dvfs_cfg['name'] if dvfs_cfg else None
                    suffix = f'_{dvfs_tag}' if dvfs_tag else ''
                    scene.reset(initial)
                    initial = scene.cube_position().tolist()
                    # Visual target has no collider and cannot interfere with object physics.
                    from pxr import UsdGeom, Gf
                    import omni.usd
                    stage = omni.usd.get_context().get_stage()
                    marker = UsdGeom.Cylinder.Get(stage, '/World/EvaluationTarget')
                    if not marker:
                        marker = UsdGeom.Cylinder.Define(stage, '/World/EvaluationTarget')
                        marker.CreateRadiusAttr(config.target_tolerance_m)
                        marker.CreateHeightAttr(.001)
                        marker.CreateDisplayColorAttr([Gf.Vec3f(.1, .9, .2)])
                        marker.AddTranslateOp()
                    UsdGeom.Xformable(marker).GetOrderedXformOps()[0].Set(Gf.Vec3d(target[0], target[1], .001))
                    marker.GetPrim().GetAttribute('visibility').Set('inherited' if task == 'pushing' else 'invisible')
                    noise_seed = episode_noise_seed(seed, tasks.index(task), episode)
                    if hasattr(policy, 'episode_seed'):
                        # Same chunk index -> same flow-matching noise under every DVFS config.
                        policy.reset(scene, task, initial, target, episode_seed=noise_seed)
                    else:
                        policy.reset(scene, task, initial, target)
                    metrics = EpisodeMetrics(task, initial, target, config)
                    monitor.clear()
                    clipped = 0
                    deadline_misses = 0
                    preview_frames = []
                    energy_tag = f"{dvfs_tag or 'default'}/{prefix}/{episode:04d}"
                    start_rtt_ms = None
                    if args.energy:
                        start_rtt_ms = request_json(server+'/episode/start', dict(tag=energy_tag))[1]
                    wall_start = time.perf_counter()
                    pace = args.mode == 'continuous' and not args.no_realtime and args.policy in ('remote', 'local')
                    trace = args.output/f'{prefix}_{episode:04d}{suffix}_trace.jsonl'
                    joint_positions = []
                    # 60 Hz arm dynamics from t = 0 (state after reset) for mechanical energy.
                    q0, qd0, tau0 = scene.dynamics()
                    dyn_t, dyn_q, dyn_qd, dyn_tau = [0.0], [q0[:7]], [qd0[:7]], [tau0[:7]]
                    drive_check = len(drive_checks) < args.drive_check_episodes
                    if drive_check:
                        diag = scene.drive_diagnostics()
                        id_samples = []  # (t, qd 9, tau 9, M 9x9, c 9, g 9) per physics step
                    cube_positions = []
                    cube_peak_speed = 0.0
                    # Sample success at 60 Hz; contacts at every physics step, actions at 20 Hz.
                    start_sim = scene.sm.get_simulation_time()
                    with trace.open('w') as log:
                        tick = 0
                        while not metrics.done:
                            if not app.is_running():
                                raise KeyboardInterrupt('Simulator closed')
                            elapsed = scene.sm.get_simulation_time() - start_sim
                            try:
                                action = policy.act(elapsed)
                            except ActionStarvation as exc:
                                metrics.reason = 'action_starvation'
                                print(f'[{energy_tag}] {exc}', flush=True)
                                break
                            except InferenceFailure as exc:
                                metrics.reason = 'inference_failure'
                                print(f'[{energy_tag}] {exc}', flush=True)
                                break
                            _, was_clipped = scene.apply(action)
                            clipped += int(was_clipped)
                            for _ in range(3):
                                monitor.clear()
                                before = scene.sm.get_simulation_time()
                                scene.sm.step(steps=1)
                                after = scene.sm.get_simulation_time()
                                if not math.isclose(after-before, 1/60, abs_tol=1e-4):
                                    raise RuntimeError('Unexpected physics timestep')
                                position = scene.cube_position().tolist()
                                velocity = array(scene.cube.get_velocities()[0])[0].tolist()
                                joint_state = np.asarray(scene.state(), dtype=np.float64)
                                # Actual executed 7-DOF arm pose, sampled each 60 Hz physics step.
                                joint_positions.append(joint_state[:7].copy())
                                q_, qd_, tau_ = scene.dynamics()
                                dyn_t.append(after-start_sim); dyn_q.append(q_[:7])
                                dyn_qd.append(qd_[:7]); dyn_tau.append(tau_[:7])
                                if drive_check:
                                    id_samples.append((after-start_sim, qd_, tau_, *scene.inverse_dynamics_terms()))
                                cube_positions.append(position)
                                cube_peak_speed = max(cube_peak_speed, float(np.linalg.norm(velocity[:3])))
                                metrics.update(after-start_sim, position, velocity,
                                               monitor.fingers == FINGERS, monitor.collision)
                                log.write(json.dumps(dict(t=after-start_sim, position=position,
                                          velocity=velocity, joint_positions=scene.state().tolist(),
                                          hand_position=array(scene.robot.get_current_state()[1]).reshape(3).tolist(),
                                          fingers=sorted(monitor.fingers),
                                          contacts=monitor.events), allow_nan=False)+'\n')
                                if metrics.done:
                                    break
                            scene.rm.render()
                            if args.save_preview and tick % 10 == 0:
                                from PIL import Image
                                _, rgb = scene.observation()
                                preview_frames.append(Image.fromarray(rgb))
                            tick += 1
                            if pace:
                                remaining = wall_start + tick/20 - time.perf_counter()
                                if remaining > 0:
                                    time.sleep(remaining)
                                else:
                                    deadline_misses += 1
                    wall_s = time.perf_counter() - wall_start
                    if hasattr(policy, 'finish'):
                        policy.finish()  # The outstanding inference belongs to this energy window.
                    energy = request_json(server+'/episode/stop', dict(tag=energy_tag, active_s=wall_s,
                                                                       include_samples=True))[0] if args.energy else {}
                    if preview_frames:
                        preview_frames[0].save(args.output/f'{prefix}_{episode:04d}{suffix}.gif',
                                               save_all=True, append_images=preview_frames[1:],
                                               duration=500, loop=0)
                        preview_frames[-1].save(args.output/f'{prefix}_{episode:04d}{suffix}_final.png')
                    metrics.tct_wall_s = wall_s
                    if metrics.tct_sim_s is None:
                        metrics.tct_sim_s = metrics.elapsed
                    if len(joint_positions) >= 6:
                        metrics.ajj = average_joint_jerk(np.asarray(joint_positions, dtype=np.float64), dt_s=1/60)
                    queue_stats = policy.queue_stats() if hasattr(policy, 'queue_stats') else {}
                    metrics.inference_stats = dict(
                        mean_rtt_ms=statistics.mean(policy.rtts) if getattr(policy, 'rtts', []) else None,
                        p95_rtt_ms=np.percentile(policy.rtts, 95) if getattr(policy, 'rtts', []) else None,
                        mean_inference_ms=statistics.mean(policy.inference_times)
                        if getattr(policy, 'inference_times', []) else None,
                        p95_inference_ms=np.percentile(policy.inference_times, 95)
                        if getattr(policy, 'inference_times', []) else None,
                        number_of_inference_requests=len(getattr(policy, 'inference_times', []) or []),
                    )
                    metrics.inference_stats.update(queue_stats)
                    result = dict(metrics.result(), episode=episode, seed=seed,
                                  initial=initial, target=target, policy=args.policy,
                                  clipped_actions=clipped, wall_time_s=metrics.tct_wall_s,
                                  tct_wall_s=metrics.tct_wall_s, tct_sim_s=metrics.tct_sim_s,
                                  requested_initial=requested_initial, noise_seed=noise_seed,
                                  mode=args.mode, control_ticks=tick, deadline_misses=deadline_misses,
                                  real_time_factor=metrics.elapsed / wall_s if wall_s > 0 else None)
                    if cube_positions:
                        result.update(
                            task_progress=float(task_progress(task, cube_positions, target, config.lift_m,
                                                              config.target_tolerance_m).max()),
                            cube_peak_speed=cube_peak_speed,
                            tip_over=float(tip_over(cube_positions, 0.010)) if task == 'pushing' else None)
                    if dvfs_cfg is not None:
                        result.update(dvfs_config=dvfs_tag, requested_cpu_khz=dvfs_cfg['cpu_khz'],
                                      requested_gpu_hz=dvfs_cfg['gpu_hz'],
                                      dvfs_governors=None if dvfs_cfg['cpu_khz'] is not None else
                                      [dvfs_cfg['cpu_governor'], dvfs_cfg['gpu_governor']],
                                      actual_cpu_khz=(dvfs_status or {}).get('cpu_khz', [None])[0],
                                      actual_gpu_hz=(dvfs_status or {}).get('gpu_hz'),
                                      dvfs_frequency_valid=bool((dvfs_status or {}).get('frequency_valid')))
                    if energy:
                        result.update(energy_j=energy.get('energy_j'), mean_power_w=energy.get('mean_power_w'),
                                      energy_window_s=energy.get('wall_s'),
                                      inference_energy_j=energy.get('inference_energy_j'),
                                      inference_duty_cycle=energy.get('inference_duty_cycle'),
                                      n_power_samples=energy.get('n_power_samples'),
                                      mean_cpu_khz=energy.get('mean_cpu_khz'),
                                      mean_gpu_hz=energy.get('mean_gpu_hz'),
                                      soc_temp_start_c=max(energy.get('temps_start_c', {}).values(), default=None),
                                      soc_temp_end_c=max(energy.get('temps_end_c', {}).values(), default=None))
                    # ---------------- DVFS energy evaluation logging (episodes.csv and raw streams)
                    consecutive_failures = consecutive_failures + 1 if metrics.reason == 'inference_failure' else 0
                    uid = f"{dvfs_tag or 'none'}|{task}|{seed}|{episode}"
                    ident = dict(experiment_id=args.experiment_id, episode_uid=uid, configuration=dvfs_tag or 'none',
                                 task=task, seed=seed, episode=episode)
                    result.update(failure_reason=E.failure_reason(result['reason'], result['success'],
                                                                  bool(getattr(policy, 'received_chunk', True))),
                                  episode_start_rtt_ms=start_rtt_ms, episode_uid=uid, order_in_episode=order_index)
                    req_rows = E.request_rows(getattr(policy, 'requests', {}), wall_start,
                                              energy.get('inferences') or [], ident)
                    dyn_t_a, dyn_qd_a, dyn_tau_a = map(np.asarray, (dyn_t, dyn_qd, dyn_tau))
                    mech = E.mechanical_energy(dyn_t_a, dyn_qd_a, dyn_tau_a)
                    act = E.actuator_energy(dyn_t_a, dyn_qd_a, dyn_tau_a, actuator_model)
                    governed = dvfs_cfg is not None and dvfs_cfg['cpu_khz'] is None
                    exp_writer.write('episodes', [E.episode_row(dict(ident, order_in_episode=order_index), result,
                                                            energy, mech, act, req_rows, dvfs_status, governed)])
                    exp_writer.write('inference', req_rows)
                    exp_writer.write('power', ([uid, t_, p_] for t_, p_ in energy.get('power_samples') or []))
                    exp_writer.write('joints', ([uid, t_, *q_, *qd_, *tau_]
                                            for t_, q_, qd_, tau_ in zip(dyn_t, dyn_q, dyn_qd, dyn_tau)))
                    exp_writer.write('ticks', ([uid, k, k/20, w_-wall_start, b_, src, held]
                                           for k, w_, b_, src, held in getattr(policy, 'tick_log', [])))
                    if drive_check and len(id_samples) > 4:
                        drive_checks.append(dict(episode_uid=uid, **E.inverse_dynamics_check(id_samples),
                                                 user_actuation_effort_nm=diag['user_actuation_effort'][:7].tolist(),
                                                 max_effort_nm=diag['max_effort'][:7].tolist(),
                                                 link_gravity_disabled=diag['link_gravity_disabled']))
                        save_json(args.output/'drive_crosscheck.json', drive_checks)
                    if consecutive_failures >= 3:
                        raise RuntimeError('Three consecutive inference/communication failures; aborting the run '
                                           'instead of recording further failed episodes.')
                    if pace and result['real_time_factor'] is not None and result['real_time_factor'] < 0.9:
                        print(f"[WARN] {energy_tag}: real-time factor {result['real_time_factor']:.2f}; "
                              'chunk release still enforces the measured delay, but wall-clock energy '
                              'windows are stretched.', flush=True)
                    rows.append(result)
                    with (args.output/'episodes.jsonl').open('a') as f:
                        f.write(json.dumps(result, allow_nan=False, default=_json_default)+'\n')
                    if reference is not None:
                        # One summary per (DVFS config, task); paired tests vs. the first config.
                        dvfs_summaries = group_summaries(rows)
                        save_json(args.output/'summary_dvfs.json', dvfs_summaries)
                        with (args.output/'summary_dvfs.csv').open('w', newline='') as f:
                            fields = list(dict.fromkeys(k for x in dvfs_summaries for k in x))
                            writer = csv.DictWriter(f, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(dvfs_summaries)
                        save_json(args.output/'paired_tests.json', paired_success_tests(rows, reference))
                        write_dvfs_tables(args.output, rows, dvfs_configs, reference)
                    if reference is None:  # Mixed-config rows would make these meaningless.
                        summaries, by_seed = summarize_runs(rows, tasks, args.seeds)
                        save_json(args.output/'summary.json', summaries)
                        with (args.output/'summary.csv').open('w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=['task', *next(iter(summaries.values())).keys()])
                            writer.writeheader()
                            writer.writerows(dict(task=t, **s) for t, s in summaries.items())
                        save_json(args.output/'summary_by_seed.json', by_seed)
                        with (args.output/'summary_by_seed.csv').open('w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=list(by_seed[0].keys()))
                            writer.writeheader()
                            writer.writerows(by_seed)
                    print(json.dumps(result, default=_json_default), flush=True)
        save_json(args.output/'status.json', dict(status='completed', episodes=len(rows)))
        exp_writer.close()
        try:
            from simulator.analyze_energy import analyze
            analyze(args.output, reference=reference)
        except Exception as exc:  # Analysis can be rerun offline; never lose the raw data.
            print(f'[WARN] Energy analysis failed ({type(exc).__name__}: {exc}); rerun '
                  f'python simulator/analyze_energy.py {args.output}', flush=True)
        if reference is not None and rows:
            print('\n' + (args.output/'table_dvfs.md').read_text(), flush=True)
    except BaseException as exc:
        exit_code = 1
        import traceback
        traceback.print_exc()
        save_json(args.output/'status.json', dict(status='aborted', completed_episodes=len(rows),
                  error=f'{type(exc).__name__}: {exc}', results_are_partial=True))
        raise
    finally:
        exp_writer.close()
        monitor.close()
        try:
            if hasattr(policy, 'executor'):
                policy.executor.shutdown(wait=True, cancel_futures=True)
            if scene is not None:
                scene.close()
        finally:
            app.close(exit_code=exit_code)


if __name__ == '__main__':
    main()
