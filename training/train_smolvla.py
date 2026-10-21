"""Train a multi-task SmolVLA policy and export a deployable NX bundle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contract import IMAGE_KEY, RESOLUTION, check_contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=Path('data/lerobot/train'))
    p.add_argument('--repo-id', default='local/franka_multitask_v2')
    p.add_argument('--output', type=Path, default=Path('outputs/smolvla_multitask'))
    p.add_argument('--export', type=Path, default=Path('outputs/nx_multitask_v2'))
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--base-model', default='lerobot/smolvla_base')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Training device override; use cpu when the local GPU build is incompatible.')
    p.add_argument('--smoke', action='store_true', help='Run the minimal 20-step smoke training configuration.')
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    if a.smoke:
        a.steps = 20
        a.output = Path('outputs/smolvla_multitask_smoke')
        a.export = Path('outputs/nx_multitask_v2_smoke')
    from importlib.metadata import version
    if version('lerobot') != '0.4.3':
        raise RuntimeError('This launcher targets LeRobot 0.4.3.')
    if a.output.exists() or a.export.exists():
        raise FileExistsError('Choose fresh output/export directories; existing training will not be overwritten.')
    contract_data = json.loads((a.dataset / 'scene_contract.json').read_text())
    check_contract(contract_data)
    inputs = {'observation.state': {'type': 'STATE', 'shape': [9]},
              IMAGE_KEY: {'type': 'VISUAL', 'shape': [3, RESOLUTION[1], RESOLUTION[0]]}}
    outputs = {'action': {'type': 'ACTION', 'shape': [9]}}
    cmd = [sys.executable, '-m', 'lerobot.scripts.lerobot_train',
           f'--policy.path={a.base_model}', f'--dataset.repo_id={a.repo_id}',
           f'--dataset.root={a.dataset.resolve()}', f'--output_dir={a.output.resolve()}',
           '--job_name=franka_smolvla_multitask', f'--policy.device={a.device}', '--policy.push_to_hub=false',
           '--wandb.enable=false', f'--steps={a.steps}', f'--batch_size={a.batch_size}',
           f'--num_workers={a.workers}', '--save_freq=20', '--eval_freq=0', f'--seed={a.seed}',
           '--policy.chunk_size=50', '--policy.n_action_steps=50', '--policy.empty_cameras=0',
           '--policy.input_features=' + json.dumps(inputs),
           '--policy.output_features=' + json.dumps(outputs)]
    print('Launching:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    source = a.output / 'checkpoints' / 'last' / 'pretrained_model'
    if not source.is_dir():
        raise FileNotFoundError(f'Expected {source}; inspect trainer output before exporting.')
    if not list(source.glob('*preprocessor*.json')) or not list(source.glob('*postprocessor*.json')):
        raise RuntimeError('Checkpoint lacks saved processors; incomplete deployment bundle.')
    shutil.copytree(source, a.export)
    shutil.copy2(a.dataset / 'scene_contract.json', a.export / 'scene_contract.json')
    (a.export / 'training_dataset.txt').write_text(str(a.dataset.resolve()) + '\n')
    metadata = {'dataset': str(a.dataset.resolve()), 'repo_id': a.repo_id,
                'base_model': a.base_model, 'steps': a.steps, 'batch_size': a.batch_size,
                'seed': a.seed, 'task': 'grasping+pushing', 'contract': contract_data}
    (a.export / 'training_metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print('NX bundle:', a.export.resolve())


if __name__ == '__main__':
    main()
