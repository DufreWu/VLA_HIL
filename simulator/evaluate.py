"""Run with Isaac Sim 6.0 Python: python simulator/evaluate.py --task both."""
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['grasping', 'pushing', 'both'], default='both')
    p.add_argument('--episodes', type=int, default=10, help='Episodes per task')
    p.add_argument('--seed', type=int, default=42)
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
    a = p.parse_args()
    if a.episodes < 1 or not 1 <= a.execute_steps <= 50 or not 0 <= a.prefetch < a.execute_steps:
        p.error('episodes must be positive; execute-steps must be between 1 and 50; prefetch must be in [0, execute_steps)')
    try:
        config = TaskConfig(timeout_s=a.timeout, hold_s=a.hold, collision_force_n=a.collision_force)
    except ValueError as exc:
        p.error(str(exc))
    return a, config


def main():
    args, config = parse_args()
    import numpy as np
    from simulator.evaluation.contacts import ContactMonitor, FINGERS
    if args.policy == 'remote':
        from simulator.evaluation.policies import RemotePolicy
        policy = RemotePolicy(args.server, args.execute_steps, args.allow_mock,
                              mode=args.mode, prefetch=args.prefetch)
    elif args.policy == 'local':
        if args.model is None:
            raise ValueError('--model is required when --policy local is used')
        from simulator.evaluation.policies import LocalPolicy
        policy = LocalPolicy(args.model, device=args.device, execute_steps=args.execute_steps,
                             mode=args.mode, prefetch=args.prefetch)
    elif args.policy == 'scripted':
        from simulator.evaluation.controllers import ScriptedPolicy
        policy = ScriptedPolicy()
    else:
        module, name = args.policy.rsplit(':', 1)
        policy = getattr(importlib.import_module(module), name)()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'run_config.json', dict(arguments={**vars(args), 'output': str(args.output)},
              metrics=asdict(config), physics_hz=60, action_hz=20, isaac_sim_target='6.0.0',
              policy_health=getattr(policy, 'health', None)))
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    monitor = ContactMonitor(config.collision_force_n)
    scene = None
    rows = []
    exit_code = 0
    try:
        from simulator.franka.scene import FrankaScene, array
        scene = FrankaScene(app, before_simulation=monitor.install)
        # Settling must produce cube/table contact reports; otherwise do not publish a zero collision rate.
        if monitor.report_count == 0:
            raise RuntimeError('No contact reports during settling. Verify PhysX contact reporting before evaluation.')
        tasks = ['grasping', 'pushing'] if args.task == 'both' else [args.task]
        for task in tasks:
            rng = np.random.default_rng(args.seed)
            for episode in range(args.episodes):
                initial = [float(rng.uniform(.42, .48)), float(rng.uniform(-.22, -.18)), .0258]
                target = [initial[0], initial[1]+.20, .0258]
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
                policy.reset(scene, task, initial, target)
                metrics = EpisodeMetrics(task, initial, target, config)
                monitor.clear()
                clipped = 0
                preview_frames = []
                wall_start = time.perf_counter()
                trace = args.output/f'{task}_{episode:04d}_trace.jsonl'
                joint_positions = []
                # Sample success at 60 Hz; contacts at every physics step, actions at 20 Hz.
                start_sim = scene.sm.get_simulation_time()
                with trace.open('w') as log:
                    tick = 0
                    while not metrics.done:
                        if not app.is_running():
                            raise KeyboardInterrupt('Simulator closed')
                        elapsed = scene.sm.get_simulation_time() - start_sim
                        action = policy.act(elapsed)
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
                            joint_positions.append(np.asarray(scene.state()[:7], dtype=np.float64).copy())
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
                if preview_frames:
                    preview_frames[0].save(args.output/f'{task}_{episode:04d}.gif',
                                           save_all=True, append_images=preview_frames[1:],
                                           duration=500, loop=0)
                    preview_frames[-1].save(args.output/f'{task}_{episode:04d}_final.png')
                metrics.tct_wall_s = time.perf_counter() - wall_start
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
                result = dict(metrics.result(), episode=episode, seed=args.seed,
                              initial=initial, target=target, policy=args.policy,
                              clipped_actions=clipped, wall_time_s=metrics.tct_wall_s,
                              tct_wall_s=metrics.tct_wall_s, tct_sim_s=metrics.tct_sim_s)
                rows.append(result)
                with (args.output/'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(result, allow_nan=False)+'\n')
                summaries = {t: summarize([r for r in rows if r['task'] == t])
                             for t in tasks if any(r['task'] == t for r in rows)}
                save_json(args.output/'summary.json', summaries)
                with (args.output/'summary.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['task', *next(iter(summaries.values())).keys()])
                    writer.writeheader()
                    writer.writerows(dict(task=t, **s) for t, s in summaries.items())
                print(json.dumps(result), flush=True)
        save_json(args.output/'status.json', dict(status='completed', episodes=len(rows)))
    except BaseException as exc:
        exit_code = 1
        import traceback
        traceback.print_exc()
        save_json(args.output/'status.json', dict(status='aborted', completed_episodes=len(rows),
                  error=f'{type(exc).__name__}: {exc}', results_are_partial=True))
        raise
    finally:
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
