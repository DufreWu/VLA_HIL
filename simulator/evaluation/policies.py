"""Synchronous and continuous remote/local policy clients for the Franka task server."""
from collections import deque
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np

from common.contract import contract, check_contract, encode_image, action_chunk, FPS, trim_chunk
from simulator.franka.run_hil import request_json

Engine = None


def _load_engine():
    global Engine
    if Engine is None:
        from jetson_orin_nx.policy_server import Engine as _Engine
        Engine = _Engine
    return Engine


TASKS = {'grasping': 'Grasp the red cube and lift it above the table.',
         'pushing': 'Push the red cube to the green target on the table.'}


def validate_response(response, episode_id, obs_id):
    if response['episode_id'] != episode_id or response['obs_id'] != obs_id:
        raise ValueError('Policy response does not match the requested observation')
    latency = float(response['inference_ms'])
    if not math.isfinite(latency) or latency < 0:
        raise ValueError('Invalid server inference time')
    return action_chunk(response['actions']), latency


class _ChunkPolicy:
    """Shared action-queue logic for chunked policies.

    sync: physics waits for each chunk, then executes ``execute_steps`` actions.
    continuous: a request runs in the background while queued actions keep
    executing. When a chunk arrives, slots whose tick already passed are skipped
    and the remaining slots REPLACE the queue, so action i of a chunk is always
    executed at tick ``obs_id + i``. If the queue runs dry the last commanded
    target is held (never the episode's initial pose).
    """

    def reset(self, scene, task, initial, target):
        import uuid
        self.scene, self.task = scene, task
        self.episode_id = str(uuid.uuid4())
        self.tick = 0
        self.queue = deque()
        self.future = None
        self.last_target = np.asarray(scene.state(), dtype=np.float32).copy()
        self.received_chunk = False
        self.hold_steps = 0
        self.starvation_events = 0
        self.queue_lengths = []
        self.dropped_slots = 0
        self.rtts, self.inference_times = [], []

    def _build_request(self, tick):
        state, rgb = self.scene.observation()
        return dict(contract=contract(), episode_id=self.episode_id, obs_id=tick,
                    state=state.tolist(), rgb_png=encode_image(rgb), task=TASKS[self.task])

    def _consume_future(self):
        payload, response, rtt = self.future.result()
        self.future = None
        try:
            actions, latency = validate_response(response, payload['episode_id'], int(payload['obs_id']))
        except ValueError:
            return
        if payload['episode_id'] != self.episode_id:
            return
        if self.mode == 'sync':
            selected, skipped = actions[:self.execute_steps], 0
            self.queue.extend(selected)
        else:
            selected, skipped = trim_chunk(actions, int(response['obs_id']), self.tick, self.execute_steps)
            self.queue = deque(selected)
        self.received_chunk = True
        self.dropped_slots += int(skipped)
        self.rtts.append(rtt)
        self.inference_times.append(latency)

    def queue_stats(self):
        if self.queue_lengths:
            queue_mean = float(np.mean(self.queue_lengths))
            queue_peak = int(np.max(self.queue_lengths))
        else:
            queue_mean, queue_peak = None, 0
        return dict(mean_queue_length=queue_mean,
                    max_queue_length=queue_peak,
                    dropped_action_slots=self.dropped_slots,
                    queue_starvation_events=self.starvation_events,
                    mean_rtt_ms=float(np.mean(self.rtts)) if self.rtts else None,
                    p95_rtt_ms=float(np.percentile(self.rtts, 95)) if self.rtts else None,
                    mean_inference_ms=float(np.mean(self.inference_times)) if self.inference_times else None,
                    p95_inference_ms=float(np.percentile(self.inference_times, 95)) if self.inference_times else None,
                    number_of_inference_requests=len(self.inference_times))

    def act(self, elapsed):
        threshold = 0 if self.mode == 'sync' else self.prefetch
        if self.future is None and len(self.queue) <= threshold:
            self.future = self.executor.submit(self._request, self._build_request(self.tick))
        if self.future is not None and (self.mode == 'sync' or self.future.done()):
            self._consume_future()

        if self.queue:
            self.last_target = np.asarray(self.queue.popleft(), dtype=np.float32)
            self.hold_steps = 0
        else:
            # Waiting for the first chunk of an episode is not starvation.
            if self.received_chunk:
                self.starvation_events += 1
            if self.future is None:
                self.hold_steps += 1
                if self.hold_steps >= 5*FPS:
                    raise RuntimeError('No usable actions for 5 simulation seconds; inspect latency/horizon, model server, and logs.')
        self.tick += 1
        self.queue_lengths.append(len(self.queue))
        return self.last_target.copy()


class RemotePolicy(_ChunkPolicy):
    def __init__(self, server, execute_steps, allow_mock, mode='sync', prefetch=10):
        self.server = server.rstrip('/')
        self.execute_steps = execute_steps
        self.mode = mode
        self.prefetch = max(0, int(prefetch))
        self.executor = ThreadPoolExecutor(max_workers=1)
        health, _ = request_json(self.server+'/health')
        check_contract(health['contract'])
        if health['mock'] and not allow_mock:
            raise ValueError('Mock server requires --allow-mock')
        self.health = health

    def _request(self, payload):
        response, rtt = request_json(self.server+'/infer', payload)
        return payload, response, rtt


class LocalPolicy(_ChunkPolicy):
    def __init__(self, model, device='cpu', execute_steps=10, mode='sync', prefetch=10):
        self.model = str(Path(model).resolve())
        self.device = str(device)
        self.execute_steps = execute_steps
        self.mode = mode
        self.prefetch = max(0, int(prefetch))
        self.executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.engine = _load_engine()(self.model, self.device)
        except Exception as exc:
            if 'env_isaacsim' in sys.executable or 'isaacsim' in sys.executable.lower():
                raise RuntimeError(
                    'Local inference was launched from the Isaac Sim environment, which is not compatible '
                    'with LeRobot 0.4.3. Run local inference from the training environment instead: '
                    'activate the training env, then execute the same command there.'
                ) from exc
            raise RuntimeError(
                'Local inference failed while loading the model. Ensure the training environment has '
                'LeRobot 0.4.3 and a compatible torch build.'
            ) from exc
        self.health = {'mock': False, 'contract': contract()}

    def _request(self, payload):
        actions, inference_ms = self.engine.infer(payload)
        response = dict(episode_id=payload['episode_id'], obs_id=payload['obs_id'],
                        actions=np.asarray(actions, dtype=np.float32).tolist(),
                        inference_ms=float(inference_ms))
        return payload, response, 0.0
