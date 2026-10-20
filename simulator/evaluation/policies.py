"""Synchronous client for the existing NX HTTP protocol."""
from collections import deque
import math
from common.contract import contract, check_contract, encode_image, action_chunk
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
    def __init__(self, server, execute_steps, allow_mock):
        self.server = server.rstrip('/')
        self.execute_steps = execute_steps
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
        self.rtts, self.inference_times = [], []

    def act(self, elapsed):
        if not self.queue:
            state, rgb = self.scene.observation()
            response, rtt = request_json(self.server+'/infer',
                dict(contract=contract(), episode_id=self.episode_id, obs_id=self.tick,
                     state=state.tolist(), rgb_png=encode_image(rgb), task=TASKS[self.task]))
            actions, inference = validate_response(response, self.episode_id, self.tick)
            self.queue.extend(actions[:self.execute_steps])
            self.rtts.append(rtt)
            self.inference_times.append(inference)
        self.tick += 1
        return self.queue.popleft()
