"""Synchronous and continuous remote policy clients for the NX HTTP protocol."""
from collections import deque
import math
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

from common.contract import contract, check_contract, encode_image, action_chunk, FPS, trim_chunk
from simulator.franka.run_hil import request_json

TASKS = {'grasping': 'Grasp the red cube and lift it above the table.',
         'pushing': 'Push the red cube to the green target on the table.'}


def validate_response(response, episode_id, obs_id):
    if response['episode_id'] != episode_id or response['obs_id'] != obs_id:
        raise ValueError('Policy response does not match the requested observation')
    latency = float(response['inference_ms'])
    if not math.isfinite(latency) or latency < 0:
        raise ValueError('Invalid server inference time')
    return action_chunk(response['actions']), latency


class RemotePolicy:
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

    def reset(self, scene, task, initial, target):
        import uuid
        self.scene, self.task = scene, task
        self.episode_id = str(uuid.uuid4())
        self.tick = 0
        self.queue = deque()
        self.future = None
        self.hold = np.asarray(scene.state(), dtype=np.float32).copy()
        self.hold_steps = 0
        self.starvation_events = 0
        self.queue_lengths = []
        self.dropped_slots = 0
        self.rtts, self.inference_times = [], []

    def _build_request(self, tick):
        state, rgb = self.scene.observation()
        payload = dict(contract=contract(), episode_id=self.episode_id, obs_id=tick,
                       state=state.tolist(), rgb_png=encode_image(rgb), task=TASKS[self.task])
        return payload

    def _request(self, payload):
        response, rtt = request_json(self.server+'/infer', payload)
        return payload, response, rtt

    def _consume_future(self):
        if self.future is None:
            return
        payload, response, rtt = self.future.result()
        self.future = None
        try:
            validate_response(response, payload['episode_id'], int(payload['obs_id']))
        except ValueError:
            return
        if payload['episode_id'] != self.episode_id:
            return
        source = int(response['obs_id'])
        actions = action_chunk(response['actions'])
        if self.mode == 'sync':
            selected, skipped = actions[:self.execute_steps], 0
        else:
            selected, skipped = trim_chunk(actions, source, self.tick, self.execute_steps)
        if len(selected):
            self.queue.extend(selected)
        self.dropped_slots += int(skipped)
        self.rtts.append(rtt)
        self.inference_times.append(float(response['inference_ms']))
        self.queue_lengths.append(len(self.queue))

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
        if self.future is None and len(self.queue) <= (0 if self.mode == 'sync' else self.prefetch):
            payload = self._build_request(self.tick)
            self.future = self.executor.submit(self._request, payload)
        if self.future is not None and (self.mode == 'sync' or self.future.done()):
            self._consume_future()
        if self.queue:
            target = self.queue.popleft()
            self.hold_steps = 0
        else:
            if self.hold_steps == 0:
                self.hold = np.asarray(self.scene.state(), dtype=np.float32).copy()
            target = self.hold.copy()
            self.hold_steps += 1
            self.starvation_events += 1
            if self.hold_steps >= 5*FPS:
                raise RuntimeError('No usable actions for 5 simulation seconds; inspect latency/horizon, model server, and logs.')
        self.tick += 1
        self.queue_lengths.append(len(self.queue))
        return target
