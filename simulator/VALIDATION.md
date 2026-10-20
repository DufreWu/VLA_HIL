# Evaluation results

## Environment

Ubuntu 24.04.4; NVIDIA RTX 5090, 32 GB VRAM; driver 580.82.07;
Python 3.12.3; Isaac Sim 6.0.0.1. Physics runs on CPU at 60 Hz and rendering uses
the RTX GPU. Commands update joint targets at 20 Hz.

The installed environment passed `pip check` and a CUDA tensor operation.
The package list is recorded in [environment.txt](validation/environment.txt).

## 100 episodes per task

```bash
simulator/run.sh --headless --task both --episodes 100 --seed 42 \
  --output outputs/isaac_100_per_task_seed42
```

| Task | Episodes | Success rate | Collision rate | Mean completion time | Time standard deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grasping | 100 | 100% | 0% | 8.5833 s | < 0.0001 s |
| Pushing | 100 | 100% | 0% | 9.9252 s | 0.1394 s |

The run completed all 200 episodes, with 100 unique episode indices per task and
zero clipped actions. Completion times use the simulation clock and include the
one-second success hold. Grasp times follow the reference controller's fixed
phase schedule. Standard deviation uses the population definition.

- [Summary JSON](validation/100_per_task_seed42/summary.json)
- [Summary CSV](validation/100_per_task_seed42/summary.csv)
- [Episode metrics](validation/100_per_task_seed42/episodes.jsonl)
- [Run settings](validation/100_per_task_seed42/run_config.json)
- [Run status](validation/100_per_task_seed42/status.json)

An earlier ten-episode-per-task run achieved 100% success and 0% collision for
both tasks, with mean times of 8.5833 s for grasping and 9.9700 s for pushing.
Its [metrics](validation/episodes.jsonl) and [settings](validation/run_config.json)
are included separately. The seed-42 sequence repeats those initial ten positions
in the larger run; the runs should not be pooled as independent samples.

## Camera previews

The one-episode smoke-test sequences show the cube lifting during grasping and
sliding along the table during pushing:

- [Grasping](validation/grasping_0000.gif)
- [Pushing](validation/pushing_0000.gif)
- [Contact sheet](validation/contact_sheet.png)

## Contact and camera checks

```bash
python simulator/validate_runtime.py
```

The probe passed RGB capture at 256x256, a physics-time change of exactly 0.0 s
during camera capture, and detection of deliberate robot/table contact.
It measured a panda_rightfinger/Table contact at 22.6426 N, above the configured
1 N threshold. See [probe result](validation/contact_probe.json).
This probe is separate from the task evaluation.

## HTTP transport checks

Both simulated tasks were run against the mock HTTP policy server on localhost.
The hold-joint policy produced the expected timeout at 2 s: 0% success, 0% collision
and null successful completion times. RGB/state transport, action chunks and
latency logging passed.

[Transport results](validation/mock_transport/episodes.jsonl) and
[settings](validation/mock_transport/run_config.json) are included.

## Regression checks

Passed: 11 simulator CPU tests, six existing pipeline tests, Python compilation,
shell syntax and whitespace checks.

## Scope

The task measurements apply to scripted controllers in the simple scene and
seeded position range specified in README.md. Trained VLA performance, Jetson Orin
inference latency and delayed control are not measured. The collision metric
excludes asset-filtered self contacts.

The installed Franka example emits a nonfatal USD/tensor finger DOF-type warning;
rendering also emits low-resolution DLSS warnings. Joint-name, camera-shape,
physics-time and contact checks passed.
