"""Stage 2: successful raw episodes -> separate LeRobot train/validation datasets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from common.contract import FPS, IMAGE_KEY, JOINTS, RESOLUTION, check_contract, write_json


def validate_episode(ep):
    meta = json.loads((ep/'metadata.json').read_text())
    if meta['success'] is not True or meta['fps'] != FPS:
        raise ValueError(f'{ep}: unsuccessful episode or FPS mismatch')
    with np.load(ep/'trajectory.npz', allow_pickle=False) as raw:
        d = {k: raw[k].copy() for k in raw.files}
    n = len(d['state'])
    if n < 2 or d['state'].shape != (n, 9) or d['action'].shape != (n, 9):
        raise ValueError(f'{ep}: bad state/action shapes')
    if not np.isfinite(d['state']).all() or not np.isfinite(d['action']).all():
        raise ValueError(f'{ep}: non-finite values')
    if len(d['timestamp']) != n or not np.allclose(np.diff(d['timestamp']), 1/FPS, atol=1e-4):
        raise ValueError(f'{ep}: irregular sampling; cannot assign a constant training FPS')
    for i in range(n):
        with Image.open(ep/'rgb'/f'{i:06d}.png') as im:
            if im.size != RESOLUTION or im.mode != 'RGB':
                raise ValueError(f'{ep}: image format mismatch')
    return meta, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=Path('data/raw'))
    p.add_argument('--output', type=Path, default=Path('data/lerobot'))
    p.add_argument('--repo-id', default='local/franka_pick_place')
    p.add_argument('--val-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    if not 0 < a.val_fraction < 1:
        p.error('val-fraction must be between 0 and 1')
    from importlib.metadata import version
    if version('lerobot') != '0.4.3':
        raise RuntimeError('Use LeRobot 0.4.3 in the separate training environment.')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    c = json.loads((a.raw/'contract.json').read_text()); check_contract(c)
    episodes = sorted(a.raw.glob('episode_*'))
    if len(episodes) < 2:
        raise ValueError('At least two successful episodes required; collect ~50+ for training.')
    rng = np.random.default_rng(a.seed); rng.shuffle(episodes)
    nval = max(1, min(len(episodes)-1, round(len(episodes)*a.val_fraction)))
    split = {'train': episodes[nval:], 'val': episodes[:nval]}
    # Validate before creating a partial dataset.
    for ep in episodes:
        validate_episode(ep)
    a.output.mkdir(parents=True, exist_ok=False)
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (9,), 'names': JOINTS},
        'action': {'dtype': 'float32', 'shape': (9,), 'names': JOINTS},
        IMAGE_KEY: {'dtype': 'image', 'shape': (RESOLUTION[1], RESOLUTION[0], 3),
                    'names': ['height', 'width', 'channels']},
    }
    for name, eps in split.items():
        root = a.output/name
        repo = a.repo_id if name == 'train' else a.repo_id+'_val'
        ds = LeRobotDataset.create(repo_id=repo, fps=FPS, root=root,
             robot_type='franka_panda_isaacsim6', features=features, use_videos=False,
             image_writer_threads=4)
        try:
            for ep in eps:
                meta, d = validate_episode(ep)
                for i, (state, action) in enumerate(zip(d['state'], d['action'])):
                    with Image.open(ep/'rgb'/f'{i:06d}.png') as im:
                        rgb = np.asarray(im).copy()
                    ds.add_frame({'observation.state': state.astype(np.float32),
                                  'action': action.astype(np.float32), IMAGE_KEY: rgb,
                                  'task': meta['task']})
                ds.save_episode()
        finally:
            ds.finalize()
            ds.stop_image_writer()
        write_json(root/'scene_contract.json', c)
    write_json(a.output/'split.json', {name: [str(ep.resolve()) for ep in eps] for name, eps in split.items()})
    print('Converted locally; no dataset was uploaded. Train:', len(split['train']), 'Val:', nval)


if __name__ == '__main__':
    main()
