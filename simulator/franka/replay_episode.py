"""Replay recorded targets in physics; validates action semantics before training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from common.contract import FPS, success_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episode', required=True, type=Path)
    p.add_argument('--headless', action='store_true')
    a = p.parse_args()
    metadata = json.loads((a.episode/'metadata.json').read_text())
    with np.load(a.episode/'trajectory.npz', allow_pickle=False) as d:
        actions = d['action'].copy()
        states = d['state'].copy()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': a.headless})
    scene = None
    try:
        from scene import FrankaScene
        scene = FrankaScene(app)
        scene.reset(metadata['initial'])
        pos, fingers, errors = [], [], []
        lifted = False
        for i, action in enumerate(actions):
            errors.append(float(np.max(np.abs(scene.state()-states[i]))))
            scene.apply(action)
            scene.advance()
            v = scene.cube_position()
            lifted |= bool(v[2] > .08)
            pos.append(v); fingers.append(scene.state()[7:])
        result = success_metrics(pos, fingers, lifted)
        result['max_joint_replay_error'] = max(errors)
        print(json.dumps(result, indent=2))
        if not result['success']:
            raise RuntimeError('Target replay did not reproduce a successful episode. Fix before training.')
    finally:
        if scene is not None:
            scene.close()
        app.close()


if __name__ == '__main__':
    main()
