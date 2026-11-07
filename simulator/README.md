# Isaac Sim manipulation evaluation

Franka Panda grasping and pushing in Isaac Sim 6.0, driven by a scripted reference
controller, a remote policy server (RTX host or Jetson Orin NX) or a local checkpoint.
Run all commands from the project root; output directories must be new.

## Installation

Requires an RTX-capable NVIDIA GPU, Python 3.12 and Isaac Sim 6.0.0.1:

```bash
python3.12 -m venv .venv-isaacsim
.venv-isaacsim/bin/python -m pip install 'isaacsim[all,extscache]==6.0.0.1' \
  --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com
```

The policy server and `--policy local` need the separate training environment
(`requirements-training.txt`, LeRobot 0.4.3); it is not compatible with Isaac Sim's Python.

## Running

```bash
# Scripted controller, one episode per task, with camera GIF.
python simulator/evaluate.py --task both --episodes 1 --save-preview --output outputs/smoke

# Transport test with the mock server (robot holds its pose; failure is expected).
python jetson_orin_nx/policy_server.py --mock --host 127.0.0.1 --port 8081
python simulator/evaluate.py --task grasping --policy remote --server http://127.0.0.1:8081 \
  --mode sync --allow-mock --episodes 1 --output outputs/mock

# Policy server (training env), on this host or on the Jetson:
python jetson_orin_nx/policy_server.py --model outputs/nx_multitask_v2 --host 0.0.0.0 --port 8080 --device cuda
```

Key options of `evaluate.py`:

| Option | Meaning |
|---|---|
| `--policy` | `scripted`, `remote`, `local` (needs `--model`, training env) or `module:Class` |
| `--mode sync` | Physics waits for every chunk; measures policy quality without latency effects |
| `--mode continuous` | Inference runs in the background while physics advances at 20 Hz wall-clock pace (`--no-realtime` disables pacing) |
| `--execute-steps`, `--prefetch` | Actions used per chunk; request the next chunk when the queue is at or below `prefetch` |
| `--seeds 42 43 ...` | Run `--episodes` per seed; initial cube poses depend only on task, seed and episode |
| `--headless` | No GUI (needed over SSH without a display) |

In continuous mode, slots of a returned chunk whose tick has already passed are skipped
and the rest replace the queue, so slot *i* always executes at tick `obs_id + i`.
If the queue runs dry, the last commanded target is held.

## Server vs Jetson comparison

```bash
SEEDS="42 43 44 45 46"
COMMON="--task both --policy remote --episodes 20 --seeds $SEEDS --execute-steps 10 --prefetch 5 --timeout 30 --headless"

# Server inference (policy server on this host, port 18080)
python simulator/evaluate.py $COMMON --server http://127.0.0.1:18080 --mode sync       --output outputs/cmp/server_sync
python simulator/evaluate.py $COMMON --server http://127.0.0.1:18080 --mode continuous --output outputs/cmp/server_cont

# Jetson inference (record power on the Jetson with tegrastats meanwhile)
python simulator/evaluate.py $COMMON --server http://JETSON_IP:8080 --mode sync       --output outputs/cmp/jetson_sync
python simulator/evaluate.py $COMMON --server http://JETSON_IP:8080 --mode continuous --output outputs/cmp/jetson_cont

# Offline metrics and paired statistics (no re-simulation; first run is the reference)
python simulator/posthoc_metrics.py --run server=outputs/cmp/server_cont \
  --run jetson=outputs/cmp/jetson_cont --output outputs/cmp/posthoc_cont
```

`posthoc_metrics.py` pairs episodes by (task, seed, episode), checks that initial
states match, and writes `episodes_posthoc.csv`, `summary.csv` (per-seed mean ± SD),
`comparison.csv` (paired difference, bootstrap 95% CI, exact McNemar or permutation
p-value) and `summary_table.tex`.

## Metrics

| Metric | Definition |
|---|---|
| Success rate | Successful / completed episodes |
| Collision rate | Episodes with an unintended contact ≥ 1 N |
| Completion time | Simulation time to confirmed success (incl. the 1 s hold), successful episodes only |
| AJJ | Mean norm of the third difference of arm joints 1–7 (rad/s³), lower is smoother |
| Task progress *(posthoc)* | Grasping: max rise / 10 cm; pushing: (d₀ − d) / (d₀ − 3 cm); clipped to [0, 1], 1 = success threshold |
| Cube peak speed *(posthoc)* | Max cube linear speed in the episode |
| Tip-over rate *(posthoc, pushing)* | Cube centre rises ≥ 1.0 cm (5 cm cube on an edge: 1.04 cm) |
| Inference / RTT | Server model time and HTTP round trip; RTT − inference = transport and encoding overhead |
| Dropped slots / starvation | Expired chunk actions skipped / steps with an empty queue |

**Grasping success:** cube centre ≥ 10 cm above its settled height with both fingers
in contact continuously for 1 s. **Pushing success:** cube within 3 cm of the target
(20 cm along +y) in XY, within 1.5 cm of its surface height and below 3 cm/s for 1 s.
Episodes time out after 30 s; a cube below z = −2 cm fails immediately.

**Collisions:** PhysX contact reports at every physics step; force = summed impulse / dt.
Finger–cube, cube–table and robot-base–table contacts are expected and excluded, so a
finger knocking the cube is *not* a collision. Other robot–environment, non-finger
robot–cube and cube–floor contacts count. Thresholds live in `evaluation/metrics.py:TaskConfig`.

Scene: 5 cm cube on a table at z = 0, initial x ∈ [0.42, 0.48], y ∈ [−0.22, −0.18] m;
physics 60 Hz, actions 20 Hz; camera 256×256 RGB; actions are 9 absolute joint targets
(`panda_joint1–7` in rad, two fingers in m).

## Output files

- `run_config.json`: arguments, metric settings, server health/contract.
- `episodes.jsonl`: per-episode metrics, initial state, target, latency and queue stats.
- `summary.json/.csv`, `summary_by_seed.json/.csv`: aggregates per task (and per seed).
- `<task>[_seed<seed>]_<episode>_trace.jsonl`: 60 Hz cube pose/velocity, joint state,
  hand position, finger contacts and unintended-contact events.
- `status.json`: `completed` or `partial`.

## Tests and layout

```bash
python -m unittest discover -s simulator/tests
python -m unittest discover -s tests
```

- `evaluate.py`: evaluation runner; `posthoc_metrics.py`: offline metrics and comparison.
- `evaluation/`: controllers, policy clients, metrics, contact monitor.
- `franka/`: shared scene, demo collection, replay, standalone HIL loop.
- `tests/`: CPU regression tests.
