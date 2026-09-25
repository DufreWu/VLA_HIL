"""Shared dataset/action contract. No Isaac Sim or torch dependencies."""
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image

SCHEMA = 'franka_joint_targets_v1'
JOINTS = [f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2']
IMAGE_KEY = 'observation.images.camera1'
TASK = 'Pick up the red cube and place it at the target on the table.'
FPS = 20
RESOLUTION = (256, 256)
TARGET = [0.45, 0.20, 0.0258]
CAMERA_EYE = [1.15, -0.85, 0.95]
CAMERA_TARGET = [0.40, 0.0, 0.15]


def contract():
    return dict(schema=SCHEMA, joints=JOINTS, fps=FPS, image_key=IMAGE_KEY,
                image_size=list(RESOLUTION), state_dim=9, action_dim=9,
                action_semantics='absolute_joint_position_targets',
                units=['rad'] * 7 + ['m'] * 2, task=TASK, target=TARGET,
                camera_eye=CAMERA_EYE, camera_target=CAMERA_TARGET, hand_offset_m=0.10)


def check_contract(value):
    if value != contract():
        raise ValueError('Model/dataset scene contract differs from this project; do not execute it.')


def vector(value):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (9,) or not np.isfinite(arr).all():
        raise ValueError('Expected nine finite joint values in canonical order.')
    return arr


def action_chunk(value):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 9 or not 1 <= len(arr) <= 1000 or not np.isfinite(arr).all():
        raise ValueError('Expected finite action array of shape [H,9].')
    return arr


def encode_image(rgb):
    b = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(b, format='PNG')
    return base64.b64encode(b.getvalue()).decode('ascii')


def decode_image(value):
    with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as im:
        if im.size != RESOLUTION:
            raise ValueError(f'Expected image size {RESOLUTION}.')
        return np.asarray(im.convert('RGB')).copy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def trim_chunk(actions, source_tick, current_tick, execute_steps):
    """Skip actions whose intended simulation slots have elapsed."""
    actions = action_chunk(actions)
    age = max(0, current_tick - source_tick)
    return actions[age:age + execute_steps].copy(), age


def success_metrics(positions, finger_positions, lifted):
    p = np.asarray(positions)
    g = np.asarray(finger_positions)
    if len(p) < FPS or len(g) != len(p):
        return dict(success=False, reason='less than one second of settling evidence')
    p, g = p[-FPS:], g[-FPS:]
    xy = np.linalg.norm(p[:, :2] - np.asarray(TARGET[:2]), axis=1)
    height = np.abs(p[:, 2] - TARGET[2])
    drift = np.linalg.norm(p - p.mean(axis=0), axis=1).max()
    ok = bool(lifted and (xy < 0.04).all() and (height < 0.015).all()
              and (g > 0.025).all() and drift < 0.005)
    return dict(success=ok, lifted=bool(lifted), max_xy_error_m=float(xy.max()),
                max_height_error_m=float(height.max()), drift_m=float(drift),
                min_finger_opening_m=float(g.min()))
