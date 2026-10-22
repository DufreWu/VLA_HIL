# Isaac Sim manipulation evaluation

Franka Panda grasping and pushing tasks with episode-level success, collision and
completion-time metrics. The package includes scripted reference controllers,
a remote policy client, regression tests and results for 100 episodes per task.

## Installation

Requirements: an RTX-capable NVIDIA GPU, a compatible NVIDIA driver, Python 3.12,
and Isaac Sim 6.0.0.1. From the package root:

```bash
python3.12 -m venv .venv-isaacsim
.venv-isaacsim/bin/python -m pip install --upgrade pip
.venv-isaacsim/bin/python -m pip install 'isaacsim[all,extscache]==6.0.0.1' \
  --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com
```

Isaac Sim prompts for the NVIDIA Omniverse license agreement on first use.
An existing Isaac Sim 6.0 environment or its `python.sh` can also run the scripts.
The virtual environment and NVIDIA assets are installed separately from this ZIP.

## Running tasks

Run commands from the project root. Output directories must be new.

```bash
# Inspect one episode of each task.
python simulator/evaluate.py --task both --episodes 1 --save-preview \
  --output outputs/smoke

# Run a single-task remote evaluation with a live model server.
python simulator/evaluate.py --task grasping --policy remote --server http://127.0.0.1:8080 \
  --mode sync --episodes 5 --execute-steps 10 --timeout 20 --output outputs/evaluation_grasping

# Run local inference directly from a trained checkpoint on this machine.
# IMPORTANT: do this in the training environment, not the Isaac Sim environment.
python simulator/evaluate.py --task grasping --policy local --model outputs/nx_multitask_v2 \
  --device cpu --mode sync --episodes 2 --execute-steps 10 --output outputs/evaluation_local

# Run a continuous remote evaluation on the same model server.
python simulator/evaluate.py --task both --policy remote --server http://127.0.0.1:8080 \
  --mode continuous --episodes 2 --execute-steps 10 --prefetch 5 --output outputs/evaluation_continuous

# Mock transport test before Jetson inference.
python jetson_orin_nx/policy_server.py --mock --host 127.0.0.1 --port 8081
python simulator/evaluate.py --task grasping --policy remote --server http://127.0.0.1:8081 \
  --mode continuous --allow-mock --episodes 1 --output outputs/hil_grasping_mock

# Remote evaluation against a real server.
python simulator/evaluate.py --task grasping --policy remote --server http://JETSON_IP:8080 \
  --mode continuous --episodes 10 --output outputs/hil_grasping
python simulator/evaluate.py --task pushing --policy remote --server http://JETSON_IP:8080 \
  --mode continuous --episodes 10 --output outputs/hil_pushing
```

Direct Python commands use the active environment. `--save-preview` exports a 2 Hz
camera GIF and a final PNG. For the current project, `--mode continuous` is the
relevant remote-control mode; the remote action queue keeps at most one in-flight
request and trims stale action slots instead of blindly appending future actions.

The scene contains a Franka Panda, a 5 cm cube and a flat table at z=0. Initial
cube positions are sampled from x=[0.42,0.48] and y=[-0.22,-0.18] m. The pushing
target is 20 cm along +y, shown by a marker without collision geometry. Physics
runs at 60 Hz; joint targets update at 20 Hz. Each task uses the same seeded
sequence of initial positions. Object pose and velocity are reset between episodes.
The reference controllers use ground-truth state and Cartesian inverse kinematics.

## Metrics

| Metric | Definition |
| --- | --- |
| Task Success Rate (%) | Successful episodes divided by completed episodes, multiplied by 100 |
| Collision Rate (%) | Episodes with at least one unintended contact above the threshold, divided by completed episodes, multiplied by 100 |
| Task Completion Time (s) | Mean simulation time to confirmed success over successful episodes; null when none succeed |

**Grasping:** lift the cube centre at least 10 cm above its settled initial height,
with positive-impulse contact from both fingers continuously for one second.
Contact loss or a drop restarts the hold.

**Pushing:** keep the cube centre within 3 cm of the target in XY, within 1.5 cm
of its initial surface height, and below 3 cm/s linear speed for one second.
Lifting above the height tolerance disqualifies the episode from pushing success.

An object below z=-2 cm fails immediately. Other unsuccessful episodes time out
after 30 simulation seconds by default. Timeouts and falls remain in the rate
denominators. Success and collision are recorded independently.

PhysX contact reports are evaluated at each physics step. Contact force is the
sum of contact impulse magnitudes divided by the physics timestep; the default
threshold is 1 N. Finger/cube contact, cube/table contact and fixed robot-base/table
support are expected contacts and are excluded. Other robot/environment contacts,
non-finger robot/cube contacts and cube/floor contacts count as collisions.
Asset-filtered self contacts are outside the collision metric's scope.

Task thresholds are defined in `evaluation/metrics.py:TaskConfig`. The CLI exposes
timeout, hold duration and collision-force threshold.

## Output files

- `run_config.json`: task, controller, seed and metric settings.
- `episodes.jsonl`: per-episode metrics, initial state, target and timing.
- `summary.json` and `summary.csv`: aggregate metrics for each task.
- `<task>_<episode>_trace.jsonl`: 60 Hz object state, robot state and contact data.
- `status.json`: completion status; aborted runs are marked as partial.

Completion time includes the required hold and uses the simulation clock. Wall
time is recorded separately. Summaries include successful completion-time
standard deviation, episode duration and clipped-action counts.

## Remote policy interface

The HTTP client uses the project's `/health` and `/infer` endpoints. Observations
contain a 256x256 RGB image, nine measured joint positions and a task instruction.
Actions are nine absolute joint targets in canonical Panda joint order:
`panda_joint1` through `panda_joint7`, followed by both finger joints.
Arm units are radians; finger units are metres.

```bash
# On the inference host, using a task-appropriate trained checkpoint:
python jetson_orin_nx/policy_server.py --model outputs/nx_multitask_v2 --host 0.0.0.0 --port 8080 --device cuda

# On the simulation host:
python simulator/evaluate.py --task both --policy remote --server http://JETSON_IP:8080 \
  --mode sync --episodes 5 --output outputs/remote_grasping
```

The server requires the model dependencies in `requirements-training.txt` and a
CUDA PyTorch build compatible with its host. Jetson requires an ARM64 build
compatible with the installed JetPack. Keep training dependencies separate from
the Isaac Sim environment.

The current project uses a multi-task scene contract with supported tasks
`grasping` and `pushing`. The task is passed through the HTTP request as a
natural-language instruction, and the action chunk is trimmed by simulation time
when the queue is behind. In continuous mode, stale slots are skipped and the
next valid chunk replaces the expired queued actions; the queue does not grow
without bound while inference is delayed.

For local execution, `--policy local` loads a checkpoint in-process through the
same `Engine` used by the HTTP server. This is useful for smoke tests or when the
inference host is the same machine as the simulator. It must be run in the
training environment, not the Isaac Sim environment. The Isaac Sim Python has a
separate environment and is not compatible with the LeRobot 0.4.3 stack used by
this model. Remote evaluation does not require the physics thread to wait for each
request in continuous mode. Mean HTTP round-trip time and server-reported
inference time are logged separately from simulation completion time.
`--allow-mock` enables hold-joint transport tests.

Custom policies can be supplied with `--policy package.module:PolicyClass`.
The class needs a no-argument constructor, `reset(scene, task, initial, target)`,
and `act(elapsed_simulation_seconds)` returning nine finite joint targets.
`scene.observation()` provides state and RGB. Policy errors abort the run and
preserve completed episodes as partial results.

## Validation

| Task | Episodes | Success rate | Collision rate | Mean completion time |
| --- | ---: | ---: | ---: | ---: |
| Grasping | 100 | 100% | 0% | 8.5833 s |
| Pushing | 100 | 100% | 0% | 9.9252 s |

These results use the scripted reference controller, seed 42 and the scene above.
See [VALIDATION.md](VALIDATION.md) for run settings, episode records and previews.

```bash
python -m unittest discover -s simulator/tests -v
python -m unittest discover -s tests -v
python simulator/validate_runtime.py
```

The runtime probe checks RGB capture, unchanged physics time during capture, and
a deliberate robot/table contact. It runs separately from benchmark episodes.

## Source layout

- `evaluate.py`: evaluation runner and result export.
- `evaluation/`: controllers, policy client, metrics and contact monitoring.
- `franka/scene.py`: shared scene, camera and joint adapter.
- `validate_runtime.py`: camera/contact integration probe.
- `tests/`: CPU regression tests.
- `validation/`: measured results and camera previews.

The package also retains the project's demonstration collection, replay, training
and Jetson utilities. Source references are listed in the package's `SOURCES.md`.
