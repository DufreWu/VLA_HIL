"""Stage 1/2: collect aligned RGB, states and issued joint targets."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from common.contract import FPS, TASK, contract, write_json, success_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('data/raw'))
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--max-attempts', type=int, default=150)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--headless', action='store_true')
    args = p.parse_args()
    if args.episodes < 1 or args.max_attempts < args.episodes:
        p.error('Require 1 <= episodes <= max-attempts')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'contract.json', contract())
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    scene = None
    try:
        from scene import FrankaScene
        scene = FrankaScene(app)
        rng = np.random.default_rng(args.seed)
        saved = 0
        for attempt in range(args.max_attempts):
            initial = [float(rng.uniform(.40, .50)), float(rng.uniform(-.25, -.15)), .0258]
            scene.reset(initial)
            states, actions, images, poses, fingers, stamps = [], [], [], [], [], []
            lifted = False
            settle = 0
            started = time.monotonic()
            for tick in range(700):
                state, rgb = scene.observation()
                sim_time = scene.sm.get_simulation_time() - scene.start_time
                done = scene.expert.is_done()
                if not done:
                    scene.expert.forward()
                # Targets, NOT measured next-state joint positions.
                action = scene.command()
                # Fail loudly if an expert method teleports joints during action computation.
                if np.max(np.abs(scene.state() - state)) > 1e-5:
                    raise RuntimeError('Expert changed measured DOFs without stepping physics. '
                                       'This controller requires adaptation before target-action recording.')
                states.append(state.copy()); actions.append(action.copy())
                images.append(rgb); stamps.append(sim_time)
                scene.advance()
                pos = scene.cube_position()
                poses.append(pos); fingers.append(scene.state()[7:])
                lifted |= bool(pos[2] > .08)
                if done:
                    settle += 1
                    if settle >= 2*FPS:
                        break
            result = success_metrics(poses[-FPS:], fingers[-FPS:], lifted)
            result.update(attempt=attempt, initial=initial, seed=args.seed,
                          wall_seconds=time.monotonic()-started, controller_done=scene.expert.is_done())
            result['success'] &= result['controller_done']
            with (args.output/'attempts.jsonl').open('a') as f:
                f.write(json.dumps(result)+'\n')
            # Save one preview even if the first episode fails.
            Image.fromarray(images[0]).save(args.output/'latest_preview.png')
            if result['success']:
                ep = args.output/f'episode_{saved:06d}'
                ep.mkdir(); (ep/'rgb').mkdir()
                for i, im in enumerate(images):
                    Image.fromarray(im).save(ep/'rgb'/f'{i:06d}.png')
                np.savez_compressed(ep/'trajectory.npz', state=np.stack(states),
                                    action=np.stack(actions), timestamp=np.asarray(stamps),
                                    cube_after_action=np.stack(poses))
                write_json(ep/'metadata.json', dict(result, task=TASK, fps=FPS, frames=len(states)))
                saved += 1
            print(f'Attempt {attempt+1}: {result}; saved {saved}/{args.episodes}', flush=True)
            if saved == args.episodes:
                break
        if saved < args.episodes:
            raise RuntimeError(f'Only {saved} successes. Inspect attempts.jsonl and latest_preview.png.')
    finally:
        if scene is not None:
            scene.close()
        app.close()


if __name__ == '__main__':
    main()
