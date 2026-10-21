"""Collect 20 Hz grasping/pushing expert demonstrations in Isaac Sim 6.0.

Suggested location in repo: simulator/franka/collect_task_demos.py
This collector relies on the project's local multitask contract(task) and
 task_instruction(task) functions.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.contract import FPS, contract, task_instruction, write_json
from simulator.evaluation.contacts import ContactMonitor, FINGERS
from simulator.evaluation.controllers import ScriptedPolicy
from simulator.evaluation.metrics import EpisodeMetrics, TaskConfig


PHYSICS_HZ = 60
STEPS_PER_ACTION = PHYSICS_HZ // FPS
assert STEPS_PER_ACTION * FPS == PHYSICS_HZ

def ensure_target_marker(task, target, config):
    from pxr import UsdGeom, Gf
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    marker = UsdGeom.Cylinder.Get(stage, '/World/EvaluationTarget')
    if not marker:
        marker = UsdGeom.Cylinder.Define(stage, '/World/EvaluationTarget')
        marker.CreateRadiusAttr(config.target_tolerance_m)
        marker.CreateHeightAttr(0.001)
        marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])
        marker.AddTranslateOp()
    ops = UsdGeom.Xformable(marker).GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(target[0], target[1], 0.001))
    marker.GetPrim().GetAttribute('visibility').Set(
        'inherited' if task == 'pushing' else 'invisible'
    )


def collect_episode(scene, monitor, task, initial, target, config):
    """One attempt. Do not create a dataset episode until success is verified."""
    from simulator.franka.scene import array

    policy = ScriptedPolicy()
    policy.reset(scene, task, initial, target)
    metrics = EpisodeMetrics(task, initial, target, config)
    monitor.clear()

    states, actions, images, stamps = [], [], [], []
    cube_positions, cube_velocities = [], []
    clipped_actions = 0
    start_sim = scene.sm.get_simulation_time()
    max_controls = math.ceil(config.timeout_s * FPS) + 1

    for _ in range(max_controls):
        if not scene.app.is_running():
            raise KeyboardInterrupt('Simulator closed')
        t = scene.sm.get_simulation_time() - start_sim

        # Important: capture camera and measured joints on the simulation thread,
        # at the beginning of the 20 Hz action interval.
        measured_state, rgb = scene.observation()
        command = np.asarray(policy.act(t), dtype=np.float32).copy()
        if command.shape != (9,) or not np.isfinite(command).all():
            raise RuntimeError(f'{task}: invalid expert action')
        if np.max(np.abs(scene.state() - measured_state)) > 1e-5:
            raise RuntimeError('Expert altered measured joint positions before physics step')
        applied, was_clipped = scene.apply(command)
        clipped_actions += int(was_clipped)

        states.append(measured_state.copy())
        actions.append(np.asarray(applied, dtype=np.float32).copy())
        images.append(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8).copy())
        stamps.append(float(t))
        cube_positions.append(scene.cube_position().copy())
        cube_velocities.append(array(scene.cube.get_velocities()[0])[0].copy())

        # ONE control action spans THREE physics steps: 60 Hz / 20 Hz = 3.
        # Success and contacts must be evaluated every physics step (60 Hz).
        for _ in range(STEPS_PER_ACTION):
            monitor.clear()
            before = scene.sm.get_simulation_time()
            scene.sm.step(steps=1)
            after = scene.sm.get_simulation_time()
            if not np.isclose(after - before, 1.0 / PHYSICS_HZ, atol=1e-4):
                raise RuntimeError('Unexpected physics timestep')
            position = scene.cube_position().tolist()
            velocity = array(scene.cube.get_velocities()[0])[0].tolist()
            metrics.update(
                after - start_sim,
                position,
                velocity,
                monitor.fingers == FINGERS,
                monitor.collision,
            )
            if metrics.done:
                break
        scene.rm.render()
        if metrics.done:
            break

    if not metrics.done:
        raise RuntimeError('Collector reached maximum steps without task termination')
    success = bool(metrics.success and clipped_actions == 0)
    reason = ('action_clipped' if metrics.success and clipped_actions else
              'success' if metrics.success else metrics.reason)
    info = dict(
        task=task, success=success, reason=reason,
        duration_s=float(metrics.elapsed),
        completion_time_s=(float(metrics.completion_time) if metrics.completion_time is not None else None),
        frames=len(states), clipped_actions=clipped_actions,
        initial=list(initial), target=list(target),
    )
    samples = dict(
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        timestamp=np.asarray(stamps, dtype=np.float64),
        cube_position=np.asarray(cube_positions, dtype=np.float32),
        cube_velocity=np.asarray(cube_velocities, dtype=np.float32),
    )
    # Ensure that training timestamps are aligned with action frequency.
    if len(stamps) > 1 and not np.allclose(np.diff(samples['timestamp']), 1.0 / FPS, atol=1e-4):
        raise RuntimeError('Collector produced irregular 20 Hz training timestamps')
    return info, samples, images


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['grasping', 'pushing', 'both'], default='both')
    p.add_argument('--output', type=Path, default=Path('data/raw_multitask'))
    p.add_argument('--episodes', type=int, default=5, help='Successful episodes per task')
    p.add_argument('--max-attempts', type=int, default=50, help='Maximum attempts per task')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--headless', action='store_true')
    args = p.parse_args()
    if args.episodes < 1 or args.max_attempts < args.episodes:
        p.error('Require 1 <= episodes <= max-attempts')

    tasks = ['grasping', 'pushing'] if args.task == 'both' else [args.task]
    config = TaskConfig()
    args.output.mkdir(parents=True, exist_ok=False)
    # Store each task's contract explicitly; a downstream multitask converter
    # must validate both rather than assuming all data is from grasping.
    write_json(args.output / 'task_contracts.json', {t: contract(t) for t in tasks})
    if len(tasks) == 1:
        write_json(args.output / 'contract.json', contract(tasks[0]))
    for task in tasks:
        (args.output / task).mkdir()

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    monitor = ContactMonitor(config.collision_force_n)
    scene = None
    summary = {task: {'attempted': 0, 'successful': 0, 'failed': 0} for task in tasks}
    try:
        from simulator.franka.scene import FrankaScene
        scene = FrankaScene(app, before_simulation=monitor.install)
        if monitor.report_count == 0:
            raise RuntimeError('No PhysX contact reports during scene settling')

        for task in tasks:
            rng = np.random.default_rng(args.seed)  # Same initial-position sequence per task
            attempts_log = args.output / task / 'attempts.jsonl'
            attempt = 0
            while summary[task]['successful'] < args.episodes and attempt < args.max_attempts:
                proposed = [
                    float(rng.uniform(0.42, 0.48)),
                    float(rng.uniform(-0.22, -0.18)),
                    0.0258,
                ]
                scene.reset(proposed)
                # The settled cube position is what the evaluator/expert actually sees.
                initial = scene.cube_position().tolist()
                target = [initial[0], initial[1] + 0.20, 0.0258]
                ensure_target_marker(task, target, config)
                scene.rm.render()

                info, samples, images = collect_episode(
                    scene, monitor, task, initial, target, config
                )
                info.update(attempt=attempt, seed=args.seed,
                            task_instruction=task_instruction(task),
                            marker_visible=(task == 'pushing'))
                summary[task]['attempted'] += 1
                if info['success']:
                    index = summary[task]['successful']
                    episode_dir = args.output / task / f'episode_{index:06d}'
                    episode_dir.mkdir()
                    rgb_dir = episode_dir / 'rgb'
                    rgb_dir.mkdir()
                    for i, frame in enumerate(images):
                        Image.fromarray(frame).save(rgb_dir / f'{i:06d}.png')
                    np.savez_compressed(episode_dir / 'trajectory.npz', **samples)
                    info.update(episode_id=f'{task}_{index:06d}', fps=FPS,
                                use_case='demo')
                    write_json(episode_dir / 'metadata.json', info)
                    summary[task]['successful'] += 1
                    print(f'{task} success attempt={attempt}: {episode_dir}', flush=True)
                else:
                    summary[task]['failed'] += 1
                    print(f'{task} failed attempt={attempt}: {info}', flush=True)
                with attempts_log.open('a') as f:
                    f.write(json.dumps(info, allow_nan=False) + '\n')
                write_json(args.output / 'summary.json', summary)
                attempt += 1
            if summary[task]['successful'] < args.episodes:
                raise RuntimeError(
                    f'Collected {summary[task]["successful"]}/{args.episodes} '
                    f'successful {task} demonstrations'
                )
    finally:
        monitor.close()
        if scene is not None:
            scene.close()
        app.close()


if __name__ == '__main__':
    main() 