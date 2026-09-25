"""Held-out offline loss; NOT a closed-loop success-rate measurement."""
import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--model', required=True, type=Path)
p.add_argument('--dataset', type=Path, default=Path('data/lerobot/val'))
p.add_argument('--repo-id', default='local/franka_pick_place_val')
p.add_argument('--batches', type=int, default=20)
p.add_argument('--batch-size', type=int, default=4)
a = p.parse_args()
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
policy = SmolVLAPolicy.from_pretrained(str(a.model)).to('cuda').eval()
pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(a.model),
                                 preprocessor_overrides={'device_processor': {'device': 'cuda'}})
# This project records 20 Hz action targets, unchanged between train and validation.
ds = LeRobotDataset(repo_id=a.repo_id, root=a.dataset,
                    delta_timestamps={'action': [i/20 for i in range(policy.config.chunk_size)]})
loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
total, count = 0., 0
torch.manual_seed(123)
with torch.inference_mode():
    for i, batch in enumerate(loader):
        if i >= a.batches:
            break
        n = len(batch['action'])
        loss, _ = policy.forward(pre(batch))
        total += float(loss)*n; count += n
if count == 0:
    raise RuntimeError('No validation examples')
print({'mean_validation_loss': total/count, 'examples': count,
       'note': 'Flow-matching loss; validate task success separately with run_hil.py.'})
