"""Validate a multi-task raw demo set before converting it to the LeRobot dataset."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contract import FPS, JOINTS, RESOLUTION, check_contract, multitask_contract, task_instruction


def validate_episode(ep):
    meta = json.loads((ep / 'metadata.json').read_text())
    if meta.get('success') is not True:
        raise ValueError(f'{ep}: episode did not succeed')
    if meta.get('fps') != FPS:
        raise ValueError(f'{ep}: expected {FPS} Hz, found {meta.get("fps")}')
    task = str(meta.get('task', '')).lower()
    if task not in {'grasping', 'pushing'}:
        raise ValueError(f'{ep}: unsupported task {task!r}')
    if meta.get('task_instruction') != task_instruction(task):
        raise ValueError(f'{ep}: task instruction mismatch for {task}')
    if task == 'pushing':
        marker_visible = meta.get('marker_visible', True)
        if marker_visible is not True:
            raise ValueError(f'{ep}: pushing target marker must be visible')
    with np.load(ep / 'trajectory.npz', allow_pickle=False) as raw:
        arr = {k: raw[k].copy() for k in raw.files}
    state = arr['state']; action = arr['action']; timestamps = arr['timestamp']
    if state.shape[1] != 9 or action.shape[1] != 9:
        raise ValueError(f'{ep}: state/action must be [N,9] arrays')
    if state.shape != action.shape:
        raise ValueError(f'{ep}: state and action shape mismatch')
    if len(timestamps) != len(state):
        raise ValueError(f'{ep}: timestamp count mismatch')
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f'{ep}: non-finite values')
    if not np.allclose(np.diff(timestamps), 1.0 / FPS, atol=1e-4):
        raise ValueError(f'{ep}: timestamps are not aligned to {FPS} Hz')
    image_dir = ep / 'rgb'
    if not image_dir.is_dir():
        raise ValueError(f'{ep}: missing rgb directory')
    images = sorted(image_dir.glob('*.png'))
    if len(images) != len(state):
        raise ValueError(f'{ep}: image count mismatch ({len(images)} != {len(state)})')
    for idx, path in enumerate(images):
        with Image.open(path) as im:
            if im.size != RESOLUTION or im.mode != 'RGB':
                raise ValueError(f'{ep}: invalid image at frame {idx}: {im.size}, {im.mode}')
    # Default robot limits; strict contract-level validation for the scene is kept in common/contract.
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                      0.0, 0.0], dtype=np.float32)
    upper = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,
                      0.04, 0.04], dtype=np.float32)
    if not ((action >= lower).all() and (action <= upper).all()):
        raise ValueError(f'{ep}: action violates robot joint limits')
    return meta, state, action, timestamps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('data/raw_multitask'))
    p.add_argument('--task', choices=['grasping', 'pushing', 'both'], default='both')
    p.add_argument('--strict', action='store_true', help='Fail on the first invalid episode')
    a = p.parse_args()
    check_contract(multitask_contract())
    tasks = ['grasping', 'pushing'] if a.task == 'both' else [a.task]
    total, good = 0, 0
    per_task = {task: {'count': 0, 'frames': 0, 'durations': []} for task in tasks}
    for task in tasks:
        task_root = a.root / task
        if not task_root.exists():
            raise FileNotFoundError(f'Missing task folder: {task_root}')
        for ep in sorted(task_root.glob('episode_*')):
            total += 1
            try:
                meta, state, action, timestamps = validate_episode(ep)
            except Exception as exc:
                if a.strict:
                    raise
                print(f'INVALID {ep}: {exc}', flush=True)
                continue
            good += 1
            per_task[task]['count'] += 1
            per_task[task]['frames'] += len(state)
            per_task[task]['durations'].append(len(state) / FPS)
            print(f'VALID {ep}: task={meta["task"]}, frames={len(state)}, duration={len(state)/FPS:.2f}s', flush=True)
    summary = {
        'total_episodes': total,
        'valid_episodes': good,
        'success_rate_pct': 100.0 * good / total if total else 0.0,
        'tasks': {task: {
            'count': per_task[task]['count'],
            'frames': per_task[task]['frames'],
            'mean_duration_s': float(np.mean(per_task[task]['durations'])) if per_task[task]['durations'] else 0.0,
            'episodes': per_task[task]['count'],
        } for task in tasks},
    }
    print(json.dumps(summary, indent=2), flush=True)
    if good == 0:
        raise RuntimeError('No valid episodes found; collect successful demonstrations first.')


if __name__ == '__main__':
    main()
