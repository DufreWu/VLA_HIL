# Franka VLA-HIL: Isaac Sim 6.0 + SmolVLA + Jetson Orin NX 16GB

This package implements the four requested stages. Merge its folders into your
`VLA_HIL` project. It does not replace your existing `franka_pick_place.py` or
`README.md`. Run commands below from the `VLA_HIL` project root.

**Validation status:** Python syntax and six CPU-only data/protocol tests passed.
Isaac Sim rendering/physics, LeRobot dataset conversion/training, and real model
inference on NX have NOT been executed here. This is an implementation to validate
on your hardware, not a claim of measured success, speed, or memory use.

## Files

| Stage | File | Purpose |
|---|---|---|
| 1–2 | `simulator/franka/collect_demos.py` | Camera + expert demonstrations + success filtering |
| 1–2 | `simulator/franka/scene.py` | Shared Isaac Sim 6.0 scene, stepping, state/action adapter |
| 2 | `simulator/franka/replay_episode.py` | Reproduce a demonstration using only saved targets |
| 2 | `training/preview_episode.py` | GIF and contact sheet from recorded images |
| 2 | `training/convert_dataset.py` | Raw episodes -> local LeRobot train/validation datasets |
| 3 | `training/train_smolvla.py` | Fine-tuning launcher and complete checkpoint export |
| 3 | `training/evaluate_loss.py` | Held-out offline loss; not task success |
| 4 | `jetson_orin_nx/policy_server.py` | Real SmolVLA or mock HTTP inference server |
| 4 | `simulator/franka/run_hil.py` | Host-side synchronous/continuous closed loop |
| 4 | `jetson_orin_nx/diagnose.py` | Read-only JetPack/Python/CUDA report |

## 0. Environments

Keep THREE environments conceptually separate:

1. **Simulation host:** your existing working `(env isaacsim)` environment, Isaac
   Sim **6.0.0**, NumPy and Pillow. Do not install LeRobot into it. Use `python`
   as you did for the working demo. A binary installation can instead use
   `/path/to/isaac-sim/python.sh` for the simulator scripts.
2. **Training host:** a separate Python 3.10/3.11 environment with a CUDA PyTorch
   build compatible with your host GPU (including Blackwell if applicable).
3. **NX:** a separate environment with ARM64 CUDA PyTorch/torchvision matching
   its actual JetPack. Training and NX both use **LeRobot 0.4.3** for this project.

In the training environment:

```bash
python -m pip install -r requirements-training.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

LeRobot 0.4.3 is deliberately pinned; current `main` uses different APIs. The
host's CUDA/PyTorch combination still needs validation. This package does not
change system drivers or install a CPU PyTorch build onto Jetson.

For a lightweight preview/mock/test environment only:

```bash
python -m pip install -r requirements-tools.txt
```

## 1. First collect TWO episodes, inspect them, and replay

In your Isaac Sim environment:

```bash
python simulator/franka/collect_demos.py \
  --output data/smoke_raw --episodes 2 --max-attempts 6 --seed 42
```

A fixed RGB camera views the whole workspace; the viewport is also positioned
near the robot. The simulation uses a 60 Hz physics step and 20 Hz action/data
sampling. Each observation contains:

- `observation.images.camera1`: 256 x 256 RGB pixels;
- `observation.state`: nine measured joint positions;
- `action`: nine **issued absolute joint position targets**, not next states;
- timestamp, episode ID (directory), frame index, task and seed/initial position.

The order is `panda_joint1 ... panda_joint7`, then both finger joints.
Arm units are radians, finger units are meters. Joint-name mapping is checked.
The camera image and state are captured before `expert.forward()` writes the
next targets. Those targets are held for three physics steps (0.05 s).
Rendering must not advance physics; the adapter checks the actual physics-time
increment and the converter rejects irregular timestamps.

The controller is called at 20 Hz, making the demonstration slower than your
original 60 Hz example. This is intentional and must remain consistent with
training and execution. The full teacher trajectory takes about 19.5 simulation
seconds plus settling; wall time depends on rendering hardware.

**Scene correction:** NVIDIA's `target_position` is a hand-link position. This
adapter adds the controller's 0.10 m approach offset to the desired cube center.
The work surface remains at z=0. The default ground is replaced with a lower
floor, avoiding a coplanar ground collider on the tabletop.

A successful episode must have lifted the cube above 0.08 m, then leave it within
4 cm XY and 1.5 cm height of the fixed target, with both fingers open, and less
than 5 mm variation over the last second. This is a basic manipulation criterion,
not a full contact/collision safety assessment. Only successful trajectories
are saved; all attempts retain metrics in `attempts.jsonl`.

Preview (any NumPy/Pillow environment):

```bash
python training/preview_episode.py --episode data/smoke_raw/episode_000000
```

Open `preview.gif` or `contact_sheet.jpg`. Confirm the camera sees the cube,
gripper and placement, and that the saved frame sequence contains the grasp.

Replay in the Isaac Sim environment:

```bash
python simulator/franka/replay_episode.py --episode data/smoke_raw/episode_000000
```

**Do not launch a long training run unless replay reports success.** If there
are no successful episodes, inspect `latest_preview.png`, `attempts.jsonl`, and
the terminal. Check hand/cube geometry and the teacher trajectory first.

## 2. Collect a dataset and convert it

In Isaac Sim:

```bash
python simulator/franka/collect_demos.py \
  --output data/raw --episodes 60 --max-attempts 180 --seed 42
```

Each episode randomizes the cube over x=[0.40,0.50], y=[-0.25,-0.15] meters.
The target, lighting, camera and language task stay fixed. This is a narrow
single-task starting dataset, not evidence of language generalization. Broaden
variation only after a baseline works. Collection keeps one episode in memory.
PNG storage can consume several GB; output directories must be new.

In the separate **training** environment:

```bash
python training/convert_dataset.py \
  --raw data/raw --output data/lerobot \
  --repo-id local/franka_pick_place --val-fraction 0.2
```

This creates separate `data/lerobot/train` and `data/lerobot/val` datasets.
Splitting is by episode, not frame, and split provenance is saved. Images are
stored as images rather than encoded video. Nothing is uploaded to Hugging Face.

## 3. Fine-tune, evaluate and export

First run a short separate smoke training job to check dependencies and shapes:

```bash
python training/train_smolvla.py \
  --dataset data/lerobot/train --steps 20 --batch-size 2 \
  --output outputs/train_smoke --export outputs/model_smoke
```

A 20-step checkpoint is only an integration check, not a usable control policy.
Then run the actual training job:

```bash
python training/train_smolvla.py \
  --dataset data/lerobot/train --steps 20000 --batch-size 8 \
  --output outputs/smolvla --export outputs/nx_model
```

Reduce batch size if memory is insufficient. 20,000 steps is a starting setting,
not a guaranteed optimum. The first run downloads pretrained SmolVLA/VLM/tokenizer
assets and requires internet access. The launcher uses upstream LeRobot training,
overrides input/output features for this Franka scene, and disables Hub/W&B uploads.

The exported `outputs/nx_model` includes policy weights/config, preprocessing and
postprocessing files, normalization tensors, and `scene_contract.json`. Copy the
**entire directory**, not just model weights. Base SO-101 normalization statistics
must not be substituted for your Franka statistics.

Optional held-out loss:

```bash
python training/evaluate_loss.py --model outputs/nx_model --dataset data/lerobot/val
```

This evaluates a configurable number of batches. Flow-matching loss is stochastic
(seed fixed by this script); it is NOT closed-loop task success. Validate using
fresh simulation seeds with the model server on your host GPU first.

## 4A. Test networking before loading a model

On NX (or another host), using NumPy/Pillow only:

```bash
python jetson_orin_nx/policy_server.py --mock --host 0.0.0.0 --port 8080
```

In the host Isaac Sim environment, replace `NX_IP` with its LAN IP:

```bash
python simulator/franka/run_hil.py \
  --server http://NX_IP:8080 --mode sync --allow-mock \
  --episodes 1 --seconds 5 --output outputs/mock_link
```

The mock repeats the current joint state. The robot should hold position; a
failed manipulation score is expected. This tests RGB/state upload, schema,
request IDs, response decoding and actuation only. It does not emulate VLA.

## 4B. Real model on host, then NX

On the training host, start the real server:

```bash
python jetson_orin_nx/policy_server.py --model outputs/nx_model --device cuda
```

In another terminal with Isaac Sim activated:

```bash
python simulator/franka/run_hil.py \
  --server http://127.0.0.1:8080 --mode sync \
  --episodes 5 --seed 10042 --output outputs/host_policy
```

Shared host GPU memory must fit both simulation and inference. If it does not,
move inference to NX once its CUDA environment is ready.

On NX, first inspect the actual installed stack:

```bash
python jetson_orin_nx/diagnose.py
```

JetPack is not known in this conversation, so no unverified NVIDIA wheel URL or
universal Docker tag is provided. Install compatible CUDA-enabled ARM64 PyTorch
and torchvision for that JetPack first. Verify `torch.cuda.is_available()`.
To keep pip from replacing those working wheels, create constraints from their
installed versions in the NX environment:

```bash
python -m pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > jetson-torch-constraints.txt
python -m pip install -c jetson-torch-constraints.txt 'lerobot[smolvla]==0.4.3'
python jetson_orin_nx/diagnose.py
```

If dependency resolution conflicts, resolve it against the actual JetPack report;
do not remove constraints and silently install different torch wheels.

Copy the project and full `outputs/nx_model` bundle to NX. First model load may
also need Hugging Face cache/internet for the backbone/tokenizer. Then:

```bash
python jetson_orin_nx/policy_server.py --model outputs/nx_model --device cuda --port 8080
```

On the simulator host:

```bash
python simulator/franka/run_hil.py \
  --server http://NX_IP:8080 --mode sync \
  --episodes 5 --output outputs/nx_sync
```

After successful synchronous trials:

```bash
python simulator/franka/run_hil.py \
  --server http://NX_IP:8080 --mode continuous \
  --execute-steps 25 --prefetch 10 --episodes 10 \
  --output outputs/nx_continuous
```

### Timing and scope

- `sync`: physics waits for each response. Functional validation only.
- `continuous`: inference runs in one worker while the main thread advances
  physics and attempts to pace at real time. It keeps at most one outstanding
  observation. Returned chunks are indexed relative to the originating tick;
  expired action slots are skipped and future slots replaced, not blindly appended.
- This is a basic asynchronous baseline, **not RTC/Jetson-PI**. It does not
  condition predictions on already committed actions. Discontinuities can still
  reduce task success. The entire 50-step prediction horizon is 2.5 s at 20 Hz;
  if latency exceeds it, no action remains usable. Do not change FPS to conceal it.
- Queue exhaustion holds the measured joint position. Five simulation seconds
  without a usable action aborts. Joint limits are enforced and clipping logged.
- The server is a single-client LAN prototype using JSON and lossless PNG, not a
  production multi-robot service. HTTP is intentionally simple; no ROS 2 is required.
- Logs include action targets/state, hold/clipping events, observation IDs, skipped
  slots, request round-trip milliseconds and server processor/model milliseconds.
  Round-trip timing starts after JSON encoding; server timing excludes PNG decoding.
  These fields are not full camera-to-actuator latency. No cross-host clocks are
  subtracted. `real_time_factor` reports simulated time / measured wall time.
- A continuous run with low real-time factor is **not evidence of real-time HIL**.
  Inspect it before studying deployment delay or DVFS.
- Success is checked over the final second of a fixed-duration episode. A policy
  that completes and later moves the cube away will count as failure.
- This package controls simulation only. It does not include a physical Franka
  driver, collision-monitoring layer, battery model or power logger. On NX you can
  separately record `tegrastats --interval 100 --logfile nx_tegrastats.log`.

## Validation and troubleshooting

```bash
python -m unittest discover -s tests -v
```

Six tests cover PNG transport, joint/schema validation, stale-chunk trimming,
success criteria, raw sampling validation and a live mock-server HTTP round trip.
They do not substitute for the simulator replay, training smoke test or GPU tests.

| Symptom | Check |
|---|---|
| Experimental Franka import error | Isaac Sim must be 6.0.0 with the examples extension installed |
| No RGB / black frames | View `latest_preview.png`; check render product and GPU rendering |
| Unexpected physics increment | Inspect manual stepping in `scene.advance`; do not bypass timestamp checks |
| No successful demonstrations | Check actual grasp/lift, hand target offset and camera preview |
| Replay fails | Resolve action semantics and drive settings before training |
| Training feature mismatch | Use the converter/launcher together with LeRobot 0.4.3 |
| Missing processors in exported model | Copy/export the complete `pretrained_model` directory |
| HTTP connection refused | Check NX IP, port 8080, server process and network reachability |
| Entire returned chunk expired | Check measured inference latency against the 2.5 s horizon |

## API references used

- Isaac Sim 6.0 manipulation: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/core_api_tutorials/tutorial_core_adding_manipulator.html
- Explicit stepping: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/introduction/quickstart_isaacsim_robot.html
- RenderingManager: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.core.rendering_manager/docs/index.html
- Franka controller source (tag v6.0.0): https://github.com/isaac-sim/IsaacSim/blob/v6.0.0/source/extensions/isaacsim.robot.experimental.manipulators.examples/isaacsim/robot/experimental/manipulators/examples/franka/pick_place.py
- LeRobot SmolVLA: https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla
- LeRobot code tag: https://github.com/huggingface/lerobot/tree/v0.4.3