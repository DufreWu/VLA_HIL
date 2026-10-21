# Multi-task VLA Training and Validation Workflow

This document records the recommended execution order for the Franka multi-task pipeline: data collection -> validation -> conversion -> smoke training -> full training.

## 1. CPU-only regression check

Run this first to confirm the core protocol, mock transport, and multi-task contract logic still pass before any Isaac Sim collection.

```bash
cd /home/chenwu/Documents/Github/VLA_HIL
python -m unittest discover -s tests -v
```

This is the minimal validation gate for the repo. It does not require Isaac Sim.

## 2. Collect demonstrations

This step must run on a machine with Isaac Sim installed and the Franka scene runtime available.

```bash
cd /home/chenwu/Documents/Github/VLA_HIL
python simulator/franka/collect_demos.py \
  --task both \
  --episodes 5 \
  --max-attempts 50 \
  --output data/raw_multitask \
  --seed 42 \
  --headless
```

Expected output:
- `data/raw_multitask/grasping/episode_000000/...`
- `data/raw_multitask/pushing/episode_000000/...`

This collection step generates the raw successful demonstrations used by the conversion pipeline.

## 3. Validate the raw multi-task dataset

Validate the collected demonstrations before trying to train.

```bash
cd /home/chenwu/Documents/Github/VLA_HIL
python training/validate_multitask_dataset.py \
  --root data/raw_multitask \
  --task both \
  --strict
```

Validation checks:
- episode success flag
- task metadata and task instruction
- FPS alignment
- image format and resolution
- state/action finiteness and shape
- timestamp spacing at 20 Hz
- pushing marker visibility where applicable

If this step fails, do not proceed to training.

## 4. Convert raw episodes to LeRobot format

Run the conversion step only after the raw dataset is valid.

```bash
cd /home/chenwu/Documents/Github/VLA_HIL
python training/convert_dataset.py \
  --raw data/raw_multitask \
  --output data/lerobot \
  --repo-id local/franka_multitask_v2 \
  --val-fraction 0.2 \
  --seed 7 \
  --min-per-task 2
```

This creates the train/validation split used by the model trainer:
- `data/lerobot/train`
- `data/lerobot/val`
- `data/lerobot/split.json`
- `data/lerobot/train/scene_contract.json`
- `data/lerobot/val/scene_contract.json`

## 5. Run a smoke training pass

This is the safety check before full fine-tuning.

```bash
cd /home/chenwu/Documents/Github/VLA_HIL
python training/train_smolvla.py \
  --dataset data/lerobot/train \
  --device cpu \
  --smoke
```

This uses a short training run and exports to:
- `outputs/smolvla_multitask_smoke`
- `outputs/nx_multitask_v2_smoke`

## 6. Run the full training job

Only after the smoke pass is confirmed.

```bash
cd /home/chenwu/Documents/Github/VLA_HIL
python training/train_smolvla.py \
  --dataset data/lerobot/train \
  --device cpu \
  --output outputs/smolvla_multitask \
  --export outputs/nx_multitask_v2 \
  --steps 5000 \
  --batch-size 8 \
  --workers 4 \
  --seed 7
```

This produces:
- trained model checkpoints under `outputs/smolvla_multitask`
- exported NX bundle under `outputs/nx_multitask_v2`

## 7. Recommended execution order

```bash
python -m unittest discover -s tests -v
python simulator/franka/collect_demos.py --task both --episodes 5 --max-attempts 50 --output data/raw_multitask --seed 42 --headless
python training/validate_multitask_dataset.py --root data/raw_multitask --task both --strict
python training/convert_dataset.py --raw data/raw_multitask --output data/lerobot --repo-id local/franka_multitask_v2 --val-fraction 0.2 --seed 7 --min-per-task 2
python training/train_smolvla.py --dataset data/lerobot/train --device cpu --smoke
python training/train_smolvla.py --dataset data/lerobot/train --device cpu --output outputs/smolvla_multitask --export outputs/nx_multitask_v2 --steps 5000 --batch-size 8 --workers 4 --seed 7
```

## Notes

- The simulator collection stage is the only one that requires Isaac Sim.
- The validation and conversion stages are required before training is allowed.
- Full training should not begin until the collected demonstrations have been validated and converted successfully.
