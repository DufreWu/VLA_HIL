"""Run a trained policy remotely. Sync is functional validation; continuous measures delay."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from common.contract import (FPS, TASK, contract, check_contract, encode_image,
                             action_chunk, trim_chunk, write_json, success_metrics)


def request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read(2_000_000))
    return result, (time.perf_counter()-started)*1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', default='http://127.0.0.1:8080')
    p.add_argument('--mode', choices=['sync', 'continuous'], default='sync')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seconds', type=float, default=35, help='Simulation seconds per episode')
    p.add_argument('--execute-steps', type=int, default=25)
    p.add_argument('--prefetch', type=int, default=10)
    p.add_argument('--seed', type=int, default=10042)
    p.add_argument('--output', type=Path, default=Path('outputs/hil'))
    p.add_argument('--headless', action='store_true')
    p.add_argument('--allow-mock', action='store_true')
    a = p.parse_args()
    if not 1 <= a.execute_steps <= 50 or not 0 <= a.prefetch < a.execute_steps:
        p.error('Require 0 <= prefetch < execute-steps <= 50')
    if a.episodes < 1 or a.seconds <= 0:
        p.error('episodes and seconds must be positive')
    health, _ = request_json(a.server.rstrip('/')+'/health')
    check_contract(health['contract'])
    if health['mock'] and not a.allow_mock:
        raise RuntimeError('Mock server detected. Use --allow-mock only for transport testing.')
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output/'run_config.json', dict(vars(a), output=str(a.output), health=health))
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': a.headless})
    scene = None
    executor = ThreadPoolExecutor(max_workers=1)
    all_results = []
    try:
        from scene import FrankaScene
        scene = FrankaScene(app)
        rng = np.random.default_rng(a.seed)
        for ep in range(a.episodes):
            initial = [float(rng.uniform(.40, .50)), float(rng.uniform(-.25, -.15)), .0258]
            scene.reset(initial)
            eid = str(uuid.uuid4())
            queue = deque()
            future = None
            hold = scene.state().copy()
            positions, fingers = [], []
            lifted = False
            clamped = 0
            starvation = 0
            rtts, inference_times = [], []
            started = time.monotonic()
            log_path = a.output/f'episode_{ep:04d}.jsonl'
            with log_path.open('w') as log:
                def event(value):
                    log.write(json.dumps(value, allow_nan=False)+'\n'); log.flush()

                def submit(tick):
                    state, rgb = scene.observation()
                    return executor.submit(request_json, a.server.rstrip('/')+'/infer',
                         {'contract': contract(), 'episode_id': eid, 'obs_id': tick,
                          'state': state.tolist(), 'rgb_png': encode_image(rgb), 'task': TASK})

                for tick in range(round(a.seconds*FPS)):
                    # At most one outstanding observation; no growing image backlog.
                    if future is None and len(queue) <= (0 if a.mode == 'sync' else a.prefetch):
                        future = submit(tick)
                    if future is not None and (a.mode == 'sync' or future.done()):
                        result, rtt = future.result()
                        future = None
                        if result['episode_id'] != eid:
                            raise RuntimeError('Response from a different episode')
                        source = int(result['obs_id'])
                        if source < 0 or source > tick:
                            raise RuntimeError('Invalid observation sequence in response')
                        actions = action_chunk(result['actions'])
                        if a.mode == 'sync':
                            selected, skipped = actions[:a.execute_steps], 0
                        else:
                            selected, skipped = trim_chunk(actions, source, tick, a.execute_steps)
                        # Replace future slots; never append a complete stale chunk.
                        if len(selected):
                            queue = deque(selected)
                        rtts.append(rtt); inference_times.append(result['inference_ms'])
                        event(dict(type='response', tick=tick, obs_id=source, skipped=skipped,
                                   usable=len(selected), rtt_ms=rtt, inference_ms=result['inference_ms']))
                    if queue:
                        target = queue.popleft()
                        starvation = 0
                    else:
                        # Queue exhausted: capture measured position once, then hold it.
                        if starvation == 0:
                            hold = scene.state().copy()
                        target = hold
                        starvation += 1
                        if starvation >= 5*FPS:
                            raise RuntimeError('No usable actions for 5 simulation seconds; '
                                               'inspect latency/horizon, model server, and logs.')
                    applied, clipped = scene.apply(target)
                    clamped += int(clipped)
                    event(dict(type='action', tick=tick, sim_time=tick/FPS,
                               state=scene.state().tolist(), target=applied.tolist(),
                               clipped=clipped, queue_length=len(queue), holding=not bool(queue) and starvation > 0))
                    scene.advance()
                    pos = scene.cube_position()
                    positions.append(pos); fingers.append(scene.state()[7:])
                    lifted |= bool(pos[2] > .08)
                    if a.mode == 'continuous':
                        deadline = started + (tick+1)/FPS
                        remaining = deadline-time.monotonic()
                        if remaining > 0:
                            time.sleep(remaining)
                # Drain the old episode before resetting the scene.
                if future is not None:
                    future.result()
            elapsed = time.monotonic()-started
            metrics = success_metrics(positions, fingers, lifted)
            metrics.update(episode=ep, initial=initial, mode=a.mode, mock=health['mock'],
                           sim_seconds=len(positions)/FPS, wall_seconds=elapsed,
                           real_time_factor=(len(positions)/FPS)/elapsed, clipped_actions=clamped,
                           mean_rtt_ms=float(np.mean(rtts)) if rtts else None,
                           p95_rtt_ms=float(np.percentile(rtts, 95)) if rtts else None,
                           mean_inference_ms=float(np.mean(inference_times)) if inference_times else None)
            all_results.append(metrics)
            write_json(a.output/'results.json', all_results)
            print(metrics, flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if scene is not None:
            scene.close()
        app.close()


if __name__ == '__main__':
    main()
