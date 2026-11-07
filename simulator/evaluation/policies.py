"""Synchronous and continuous remote/local policy clients for the Franka task server."""
from collections import deque
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

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


class ActionStarvation(RuntimeError):
    """Raised when no usable action has been available for too long."""


def release_delay_ticks(delay_s, fps=FPS):
    """Control ticks that must elapse before a chunk observed ``delay_s`` ago may act."""
    delay_s = float(delay_s)
    if not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError('delay_s must be finite and non-negative')
    return max(1, int(math.ceil(delay_s * fps - 1e-9)))


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

    Delay-faithful release (continuous): a returned chunk only takes effect once
    the simulation has advanced ``ceil(capture-to-arrival delay * FPS)`` ticks.
    With wall-clock pacing this keeps the action delay in simulated time equal to
    the measured delay even if the simulator runs slower than real time.

    ``episode_seed`` (optional, per reset) adds a ``noise_seed`` to each request so
    the k-th chunk of an episode uses the same flow-matching noise in every run.
    ``starvation_abort_s`` (0 disables) raises ActionStarvation after that long
    with no usable action and no request in flight.
    """

    def _init_common(self, execute_steps, mode, prefetch, starvation_abort_s):
        self.execute_steps = execute_steps
        self.mode = mode
        self.prefetch = max(0, int(prefetch))
        self.starvation_abort_s = float(starvation_abort_s or 0.0)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.episode_seed = None

    def reset(self, scene, task, initial, target, episode_seed=None):
        import uuid
        self.finish()
        self.scene, self.task = scene, task
        self.episode_id = str(uuid.uuid4())
        self.episode_seed = None if episode_seed is None else int(episode_seed)
        self.tick = 0
        self.queue = deque()
        self.future = None
        self.ready = None
        self.last_target = np.asarray(scene.state(), dtype=np.float32).copy()
        self.received_chunk = False
        self.hold_steps = 0
        self.starvation_events = 0
        self.inflight_hold_ticks = 0
        self.hold_steps_total = 0
        self.release_wait_ticks = 0
        self.request_index = 0
        self.queue_lengths = []
        self.dropped_slots = 0
        self.action_delays = []
        self.rtts, self.inference_times = [], []

    def finish(self):
        """Wait for and discard any outstanding request (call before the next reset)."""
        future = getattr(self, 'future', None)
        if future is not None:
            try:
                future.result()
            except Exception:
                pass
        self.future = None
        self.ready = None

    def _build_request(self, tick):
        state, rgb = self.scene.observation()
        payload = dict(contract=contract(), episode_id=self.episode_id, obs_id=tick,
                       state=state.tolist(), rgb_png=encode_image(rgb), task=TASKS[self.task])
        if getattr(self, 'episode_seed', None) is not None:
            payload['noise_seed'] = (self.episode_seed * 100_003 + self.request_index) % (2**31 - 1)
        self.request_index += 1
        return payload

    def _timed_request(self, payload, captured):
        payload, response, rtt = self._request(payload)
        return payload, response, rtt, time.perf_counter() - captured

    def _submit(self):
        captured = time.perf_counter()
        self.future = self.executor.submit(self._timed_request, self._build_request(self.tick), captured)

    def _consume_future(self):
        payload, response, rtt, delay_s = self.future.result()
        self.future = None
        try:
            actions, latency = validate_response(response, payload['episode_id'], int(payload['obs_id']))
        except ValueError:
            return
        if payload['episode_id'] != self.episode_id:
            return
        self.rtts.append(rtt)
        self.inference_times.append(latency)
        source = int(response['obs_id'])
        release = source + (0 if self.mode == 'sync' else release_delay_ticks(delay_s))
        self.ready = (source, release, actions)

    def _apply_ready(self):
        source, release, actions = self.ready
        if self.tick < release:
            self.release_wait_ticks += 1
            return
        self.ready = None
        if self.mode == 'sync':
            selected, skipped = actions[:self.execute_steps], 0
            self.queue.extend(selected)
        else:
            selected, skipped = trim_chunk(actions, source, self.tick, self.execute_steps)
            self.action_delays.append(self.tick - source)
            self.queue = deque(selected)
        self.received_chunk = True
        self.dropped_slots += int(skipped)

    def queue_stats(self):
        if self.queue_lengths:
            queue_mean = float(np.mean(self.queue_lengths))
            queue_peak = int(np.max(self.queue_lengths))
        else:
            queue_mean, queue_peak = None, 0
        delays = np.asarray(self.action_delays, dtype=np.float64)
        return dict(mean_queue_length=queue_mean,
                    max_queue_length=queue_peak,
                    dropped_action_slots=self.dropped_slots,
                    queue_starvation_events=self.starvation_events,
                    inflight_hold_ticks=self.inflight_hold_ticks,
                    holding_duration_s=(self.inflight_hold_ticks + self.hold_steps_total) / FPS,
                    release_wait_ticks=self.release_wait_ticks,
                    mean_action_delay_ticks=float(delays.mean()) if delays.size else None,
                    max_action_delay_ticks=int(delays.max()) if delays.size else None,
                    mean_rtt_ms=float(np.mean(self.rtts)) if self.rtts else None,
                    p95_rtt_ms=float(np.percentile(self.rtts, 95)) if self.rtts else None,
                    mean_inference_ms=float(np.mean(self.inference_times)) if self.inference_times else None,
                    p95_inference_ms=float(np.percentile(self.inference_times, 95)) if self.inference_times else None,
                    number_of_inference_requests=len(self.inference_times))

    def act(self, elapsed):
        threshold = 0 if self.mode == 'sync' else self.prefetch
        if self.future is None and self.ready is None and len(self.queue) <= threshold:
            self._submit()
        if self.future is not None and (self.mode == 'sync' or self.future.done()):
            self._consume_future()
        if self.ready is not None:
            self._apply_ready()

        if self.queue:
            self.last_target = np.asarray(self.queue.popleft(), dtype=np.float32)
            self.hold_steps = 0
        else:
            # Waiting for the first chunk of an episode is not starvation.
            if self.received_chunk:
                self.starvation_events += 1
            if self.future is not None or self.ready is not None:
                self.inflight_hold_ticks += 1
            else:
                self.hold_steps += 1
                self.hold_steps_total += 1
                abort_s = getattr(self, 'starvation_abort_s', 5.0)
                if abort_s and self.hold_steps >= round(abort_s * FPS):
                    raise ActionStarvation(f'No usable actions for {abort_s:g} simulation seconds; '
                                           'inspect latency/horizon, model server, and logs.')
        self.tick += 1
        self.queue_lengths.append(len(self.queue))
        return self.last_target.copy()


class RemotePolicy(_ChunkPolicy):
    def __init__(self, server, execute_steps, allow_mock, mode='sync', prefetch=10,
                 starvation_abort_s=5.0):
        self._init_common(execute_steps, mode, prefetch, starvation_abort_s)
        self.server = server.rstrip('/')
        health, _ = request_json(self.server+'/health')
        check_contract(health['contract'])
        if health['mock'] and not allow_mock:
            raise ValueError('Mock server requires --allow-mock')
        self.health = health

    def _request(self, payload):
        response, rtt = request_json(self.server+'/infer', payload)
        return payload, response, rtt


class LocalPolicy(_ChunkPolicy):
    def __init__(self, model, device='cpu', execute_steps=10, mode='sync', prefetch=10,
                 starvation_abort_s=5.0):
        self._init_common(execute_steps, mode, prefetch, starvation_abort_s)
        self.model = str(Path(model).resolve())
        self.device = str(device)
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
