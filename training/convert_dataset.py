"""Convert successful multi-task raw episodes into a balanced LeRobot train/validation split."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contract import (FPS, IMAGE_KEY, JOINTS, RESOLUTION, check_contract,
                            multitask_contract, task_instruction, write_json)


def validate_episode(ep):
    meta = json.loads((ep / 'metadata.json').read_text())
    if meta.get('success') is not True or meta.get('fps') != FPS:
        raise ValueError(f'{ep}: unsuccessful episode or FPS mismatch')
    task = str(meta.get('task', '')).lower()
    if task not in {'grasping', 'pushing'}:
        inferred = str(ep.parent.name).lower()
        task = inferred if inferred in {'grasping', 'pushing'} else ''
    if task in {'grasping', 'pushing'}:
        expected_instruction = task_instruction(task)
        if meta.get('task_instruction') not in {None, expected_instruction}:
            raise ValueError(f'{ep}: wrong task instruction for {task}')
        if task == 'pushing':
            marker_visible = meta.get('marker_visible', True)
            if marker_visible is not True and meta.get('task_instruction') is not None:
                raise ValueError(f'{ep}: pushing target marker not recorded in metadata')
    with np.load(ep / 'trajectory.npz', allow_pickle=False) as raw:
        d = {k: raw[k].copy() for k in raw.files}
    for key in ('state', 'action', 'timestamp'):
        if key not in d:
            raise ValueError(f'{ep}: missing {key} array')
    state = d['state']; action = d['action']; timestamps = d['timestamp']
    n = len(state)
    if n < 2 or state.shape != (n, 9) or action.shape != (n, 9):
        raise ValueError(f'{ep}: bad state/action shapes')
    if len(timestamps) != n:
        raise ValueError(f'{ep}: timestamp length mismatch')
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f'{ep}: non-finite values')
    if not np.allclose(np.diff(timestamps), 1 / FPS, atol=1e-4):
        raise ValueError(f'{ep}: irregular sampling; expected {1/FPS} s spacing at {FPS} FPS')
    image_dir = ep / 'rgb'
    if not image_dir.is_dir() or len(list(image_dir.glob('*.png'))) != n:
        raise ValueError(f'{ep}: image count mismatch')
    for i in range(n):
        with Image.open(image_dir / f'{i:06d}.png') as im:
            if im.size != RESOLUTION or im.mode != 'RGB':
                raise ValueError(f'{ep}: image format mismatch at frame {i}')
    return meta, d


def build_split(episodes, val_fraction, seed):
    rng = np.random.default_rng(seed)
    episodes = sorted(episodes)
    rng.shuffle(episodes)
    nval = max(1, min(len(episodes) - 1, round(len(episodes) * val_fraction))) if len(episodes) > 1 else 0
    return {'train': episodes[nval:], 'val': episodes[:nval]} if nval else {'train': episodes, 'val': []}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=Path('data/raw_multitask'))
    p.add_argument('--output', type=Path, default=Path('data/lerobot'))
    p.add_argument('--repo-id', default='local/franka_multitask_v2')
    p.add_argument('--val-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--min-per-task', type=int, default=2)
    a = p.parse_args()
    if not 0 < a.val_fraction < 1:
        p.error('val-fraction must be between 0 and 1')
    if a.min_per_task < 1:
        p.error('min-per-task must be at least 1')
    tasks = ['grasping', 'pushing']
    episode_map = {}
    for task in tasks:
        task_root = a.raw / task
        if not task_root.exists():
            raise FileNotFoundError(f'Missing task root: {task_root}')
        episodes = sorted(task_root.glob('episode_*'))
        if len(episodes) < a.min_per_task:
            raise ValueError(f'Need at least {a.min_per_task} successful episodes in {task_root}; collect more demonstrations.')
        for ep in episodes:
            meta, _ = validate_episode(ep)
            if meta['task'] != task:
                raise ValueError(f'{ep}: metadata task does not match directory task {task}')
        episode_map[task] = episodes
    contract_data = multitask_contract()
    check_contract(contract_data)
    split = {}
    for task in tasks:
        split[task] = build_split(episode_map[task], a.val_fraction, a.seed + (0 if task == 'grasping' else 1))
    train_episodes = sorted([ep for task in tasks for ep in split[task]['train']])
    val_episodes = sorted([ep for task in tasks for ep in split[task]['val']])
    if not train_episodes or not val_episodes:
        raise ValueError('Split produced an empty train or validation set; collect more episodes.')
    a.output.mkdir(parents=True, exist_ok=False)
    from importlib.metadata import version
    if version('lerobot') != '0.4.3':
        raise RuntimeError('Use LeRobot 0.4.3 in the separate training environment.')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (9,), 'names': JOINTS},
        'action': {'dtype': 'float32', 'shape': (9,), 'names': JOINTS},
        IMAGE_KEY: {'dtype': 'image', 'shape': (RESOLUTION[1], RESOLUTION[0], 3),
                    'names': ['height', 'width', 'channels']},
    }
    for name, episodes in [('train', train_episodes), ('val', val_episodes)]:
        root = a.output / name
        repo = f'{a.repo_id}_{name}'
        ds = LeRobotDataset.create(repo_id=repo, fps=FPS, root=root,
                                   robot_type='franka_panda_isaacsim6', features=features,
                                   use_videos=False, image_writer_threads=4)
        try:
            for ep in episodes:
                meta, d = validate_episode(ep)
                for i, (state, action) in enumerate(zip(d['state'], d['action'])):
                    with Image.open(ep / 'rgb' / f'{i:06d}.png') as im:
                        rgb = np.asarray(im).copy()
                    ds.add_frame({'observation.state': state.astype(np.float32),
                                  'action': action.astype(np.float32),
                                  IMAGE_KEY: rgb,
                                  'task': meta['task_instruction']})
                ds.save_episode()
        finally:
            ds.finalize(); ds.stop_image_writer()
        write_json(root / 'scene_contract.json', contract_data)
    write_json(a.output / 'split.json', {
        'train': [str(ep.resolve()) for ep in train_episodes],
        'val': [str(ep.resolve()) for ep in val_episodes],
        'tasks': {task: [str(ep.resolve()) for ep in episode_map[task]] for task in tasks},
        'seed': a.seed,
        'val_fraction': a.val_fraction,
    })
    print(f'Converted multi-task dataset to {a.output}. Train episodes: {len(train_episodes)}; Val episodes: {len(val_episodes)}')


if __name__ == '__main__':
    main()
