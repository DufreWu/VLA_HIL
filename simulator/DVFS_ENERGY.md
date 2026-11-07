# DVFS energy evaluation

Closed-loop evaluation of how Jetson Orin NX CPU/GPU DVFS settings affect inference
latency, computational energy, robot motion and task success. The policy (SmolVLA,
`outputs/nx_multitask_v2`), tasks, action processing and inference API are unchanged.

## Protocol

| Item | Setting |
|---|---|
| Simulation | Isaac Sim 6.0, Franka Panda, physics 60 Hz, control 20 Hz |
| Tasks | grasping, pushing; success, collision and AJJ definitions as in `README.md` |
| Inference | Jetson Orin NX 16 GB, continuous mode, chunk 50 actions (2.5 s), `--execute-steps 25 --prefetch 20` |
| Pacing | Wall-clock pacing (physics never waits for inference); RTF logged per episode |
| Timeout | 30 s simulation time; success needs a 1 s hold |
| Configs | `default` (schedutil / nvhost_podgov), `gpu_max_cpu_max`, `gpu_min_cpu_max`, `gpu_max_cpu_min`, `gpu_min_cpu_min` |
| Episodes | 5 seeds × 10 episodes × 2 tasks × 5 configs = 500 |
| Pairing | Every (task, seed, episode) runs under all 5 configs from the same initial cube pose and the same flow-matching noise seeds; config order rotates per episode (cyclic Latin square) |
| Warm-up | 3 server-side inferences after every DVFS switch, excluded from all statistics |

`min`/`max` in `jetson_orin_nx/benchmark/closed_loop_dvfs_energy.yaml` are resolved by the
server against the frequencies the device lists as available (CPU: intersection of every
core's `scaling_available_frequencies`; GPU: devfreq `available_frequencies`). The device
lists, the resolved values and the `nvpmodel` mode are printed before Isaac Sim starts and
stored in `experiment_config.json`. Unsupported values abort the run before any episode.

## Commands

```bash
# 1. Jetson: install this version of jetson_orin_nx/ and common/ (e.g. git pull), then
sudo -E $(which python) jetson_orin_nx/policy_server.py --model outputs/nx_multitask_v2 \
    --host 0.0.0.0 --port 8080 --dvfs-control           # restores original DVFS on exit

# 2. Host (Isaac Sim environment): one-episode smoke test (1 seed x 1 episode x 2 tasks x 5 configs)
python simulator/evaluate.py --policy remote --server http://JETSON_IP:8080 --mode continuous \
    --task both --seeds 41 --episodes 1 --execute-steps 25 --prefetch 20 --timeout 30 \
    --dvfs-configs jetson_orin_nx/benchmark/closed_loop_dvfs_energy.yaml --energy --headless \
    --output outputs/dvfs_energy_evaluation/smoke_$(date -u +%Y%m%dT%H%M%SZ)

# 3. Full 500-episode evaluation (several hours; output directory must not exist)
python simulator/evaluate.py --policy remote --server http://JETSON_IP:8080 --mode continuous \
    --task both --seeds 41 42 43 44 45 --episodes 10 --execute-steps 25 --prefetch 20 --timeout 30 \
    --dvfs-configs jetson_orin_nx/benchmark/closed_loop_dvfs_energy.yaml --energy --headless \
    --output outputs/dvfs_energy_evaluation/$(date -u +%Y%m%dT%H%M%SZ)
#    optional: --actuator-model simulator/config/actuator_model.yaml  (see "Actuator model")

# 4. Aggregation, validation and figures (also run automatically at the end of step 3)
python simulator/analyze_energy.py outputs/dvfs_energy_evaluation/<ID> --strict
```

**Full CPU × GPU grid.** `jetson_orin_nx/benchmark/closed_loop_dvfs_grid.yaml` expands to
`default` plus one fixed config per (CPU, GPU) pair, named `cpu<MHz>_gpu<MHz>`. `all`
takes the device lists, `board` takes `jetson_orin_nx.yaml`, or an explicit list can be
given. The expanded count and episode total are printed before Isaac Sim starts.
`--dvfs-names` selects a subset after expansion, e.g. to resume or split a grid run.

| Grid | Configs | Episodes at 5 seeds × 10 | Estimated time (≈ 40 s / episode incl. switch + warm-up) |
|---|---|---|---|
| Device lists (typically 26 CPU × 7 GPU) | 183 | 18 300 | ≈ 8.5 days |
| `board` lists (12 × 7) | 85 | 8 500 | ≈ 4 days |
| Device lists, `--seeds 41 --episodes 10` | 183 | 3 660 | ≈ 1.7 days |
| `closed_loop_dvfs_3x3.yaml`: CPU, GPU ∈ {min, mid, max} + `default` | 10 | 1 000 | ≈ 11 h |

Grid lists may contain `min`, `mid` (listed value closest to (min + max)/2) and `max`,
resolved against the device's own frequency lists.

With more than 8 grid configs, Figures 1–6 are drawn as CPU × GPU heatmaps. The default
governor's value is given in each panel title, and success rates use a fixed 0–100 % scale.

`--dvfs-configs` refuses `--mode sync` and `--no-realtime`. Three consecutive inference or
communication failures abort the run, so the episodes after a server crash are not
recorded as task failures. Raw logs are append-only and flushed after every episode.

## Measurements

**Clock domains.** No timestamp is ever subtracted across hosts.
- *Simulation time:* used for joint dynamics, task timing, AJJ, E_mech and E_act.
- *Host wall time:* `perf_counter` from the episode start. Used for request timestamps, buffer events and RTF.
- *Jetson monotonic time:* used for VDD_IN samples and inference intervals.

The client starts its wall clock when `/episode/start` returns and sends its wall duration
(`active_s`) with `/episode/stop`. E_comp is integrated over `[server start, + active_s]`,
and the offset between the two windows is bounded by `episode_start_rtt_ms`. With pacing,
the wall and sim intervals differ by `time_alignment_error_s`, which is validated to be ≤ 5 %.

**Computational energy (measured).**
- **Source:** INA3221 channel 1 (VDD_IN, module input), `in1_input × curr1_input`, sampled by a server thread at a nominal 200 Hz.
- **Rates:** `power_sample_rate_hz` is the achieved polling rate; `power_sensor_update_rate_hz` is the rate at which the reading actually changed.
- **E_comp:** trapezoidal integral over the active window, linearly interpolated at the window ends.
- **E_inf,i:** integral over the model forward pass of request *i*, the same timed region as `inference_ms`. Reported only if the sensor refreshed at least twice inside that interval; otherwise it is NaN, and the episode-mean inference energy and EDP are NaN too.
- **EDP:** EDP_i = E_inf,i × L_i, and EDP_comp = mean_i EDP_i. Whole-episode energy is never multiplied by latency.

**Mechanical energy (simulated).**
- **Signals:** q, q̇ and τ for arm joints 1–7 at every physics step (60 Hz) from t = 0. τ is the PhysX projected joint force, i.e. the force the joint drive applied, not the target.
- **Effort check:** `drive_crosscheck.json` compares τ with inverse dynamics, M(q)q̈ + c + g, over the first 2 episodes. The smoke test gave a 7 % relative residual in free motion and a correlation of 0.98 on the gravity-loaded joints. The user-set actuation force is 0.
- **E_mech:** ∫ Σ_j max(0, τ_j q̇_j) dt (positive work). `mechanical_energy_signed_j` is kept separately, and no regeneration is assumed.
- **Gripper:** finger joints are excluded.

**Actuator electrical energy (model-based).**
- **Model:** P_act,j = max(0, τ_j q̇_j)/η_j + α_j τ_j² + P0_j, so E_act already contains the mechanical work and E_mech is never added to it.
- **Parameters:** the repository contains **no** calibrated parameters. Copy `simulator/config/actuator_model.example.yaml`, fill every value with its provenance, and pass `--actuator-model`. Otherwise E_act and E_total are NaN, Figure 3 is not drawn and Figure 5 shows computational energy per success.
- **Labels:** the model's provenance and its `estimate` flag are stored in `experiment_config.json`, and the figures label actuation energy as a model-based estimate.

**Totals.** E_total = E_comp + E_act. Energy per success is Σ E over **all** attempted episodes of a group (failures and timeouts included) divided by the number of successes, and is NaN when there are no successes.

**Action timing.** Each request has the following timestamps:
- **Observation capture:** `capture_wall_s`.
- **Payload encoded:** `encoded_wall_s`.
- **Request sent:** `sent_wall_s`.
- **Response received:** `received_wall_s`.
- **Chunk available:** `available_wall_s`, when the control loop sees it.
- **First action execution:** `first_exec_wall_s` and `first_exec_tick`.

The server returns its own durations (`server_timing_ms`), so `communication_ms = rtt_ms − server_total_ms`. The remaining timing metrics:
- **Action delay:** first executed tick − observation tick (in ticks), and the wall time from capture to first execution (in ms).
- **Observation age:** tick − source tick of every executed action.
- **Holding time:** ticks with an empty buffer, i.e. the initial wait plus underrun.
- **Underrun rate:** fraction of the ticks after the first chunk that had an empty buffer.

`latency_horizon_analysis.csv` relates latency above the 2.5 s horizon to underruns and
success. It is descriptive only: success rates with Wilson CIs per latency group, and Spearman ρ.

## Output files (`outputs/dvfs_energy_evaluation/<experiment_id>/`)

| File | Content |
|---|---|
| `experiment_config.json` | Protocol, arguments, device frequency lists, resolved configs, power mode, sensor and clock conventions, actuator model and provenance |
| `episodes.csv` | One row per episode (schema below) |
| `inference_requests.csv` | One row per request, incl. drained and failed ones |
| `power_samples.csv.gz` | `episode_uid, server_t_s, power_w` (raw VDD_IN, server clock from episode start) |
| `joint_samples.csv.gz` | `episode_uid, sim_t_s, q1..q7_rad, qd1..qd7_rad_s, tau1..tau7_nm` at 60 Hz |
| `action_ticks.csv.gz` | `episode_uid, tick, sim_t_s, wall_t_s, buffer_after, source_obs_tick, held` |
| `per_task_summary.csv`, `per_config_summary.csv` | Mean, SD (episode level), n per metric; success/collision rate with Wilson 95% CI; seed-level SD of success rate and E_comp; energy per success; failure reasons |
| `paired_tests.csv` | Each config vs. `default` on matched episodes: exact McNemar (success); paired sign-flip permutation + bootstrap 95% CI (energies, latency) |
| `validation_report.json` | All automated checks with pass/fail and details |
| `drive_crosscheck.json` | Effort-signal verification |
| `figures/fig1..fig6.{pdf,png}` | Latency, power and inference energy, energy breakdown (actuator model only), success rate, energy per success, action timing |
| `episodes.jsonl`, `table_dvfs.*`, `*_trace.jsonl` | Existing outputs, unchanged |

`episodes.csv` columns. Missing values are `nan`, never 0; `frequency_valid` and
`frequency_stable` are empty or `nan` when not applicable.

| Group | Columns |
|---|---|
| Identity | `experiment_id, episode_uid, configuration, task, seed, episode, order_in_episode` |
| Task | `success, failure_reason, raw_reason, collision, episode_duration_s, completion_time_s` (successes only), `task_progress, ajj` |
| DVFS | `cpu_mhz_requested, gpu_mhz_requested` (resolved; NaN for governor), `cpu_mhz_actual, gpu_mhz_actual` (mean of 10 Hz samples in the episode), `cpu/gpu_mhz_min/max, cpu_governor, gpu_governor, power_mode, frequency_valid` (read-back after the switch), `frequency_stable` (pinned configs: every in-episode sample within 2 %), `soc_temp_start_c, soc_temp_end_c` |
| Inference | `inference_ms_mean, inference_ms_p95` (forward pass, all requests that ran), `rtt_ms_mean, communication_ms_mean, n_inference_requests` |
| Computation | `computational_power_w, computational_energy_j` (active window), `computational_energy_window_j` (incl. draining the last request), `inference_energy_j` (mean per inference), `inference_energy_total_j, inference_power_w, computational_edp_js, n_inference_energy, power_sample_rate_hz, power_sensor_update_rate_hz` |
| Mechanics | `mechanical_power_w, mechanical_energy_j, mechanical_energy_signed_j, mechanical_peak_power_w, actuator_power_w, actuator_energy_j, total_energy_j, total_power_w` |
| Timing | `action_delay_ms, action_delay_ticks, obs_age_ms, holding_time_s, initial_wait_s, underrun_s, buffer_underrun_rate, dropped_action_slots, mean_buffer_occupancy` |
| Pacing | `wall_duration_s, comp_window_s, real_time_factor, deadline_misses, time_alignment_error_s, episode_start_rtt_ms` |

`failure_reason`: `timeout_task_not_reached`, `no_valid_action` (no chunk ever executed, or
starvation abort), `constraint_violation` (cube fell off the table),
`inference_or_communication_failure`. `collision` is reported separately and never ends an
episode; expected finger–cube and cube–table contacts are excluded.

## Validation

`validation_report.json` (`--strict` exits 1 on any error-level failure) checks:
- required columns, and no duplicated episode IDs or (config, task, seed, episode);
- E_comp, E_mech, E_act and E_total ≥ 0, and E_total = E_comp + E_act where available;
- no exact zeros in measured columns (a missing value must be NaN, not 0);
- success-rate denominators equal attempted episodes, and zero-success groups have NaN energy per success;
- DVFS read-back validity, plus in-episode frequency stability (warning);
- episode timestamps, completion times only for successes, and monotone per-request timestamps with communication ≥ 0;
- continuous paced mode, 0.95 ≤ RTF ≤ 1.05, and wall/sim interval agreement ≤ 5 % (warnings);
- E_comp and E_mech recomputed from the raw sample files (relative error ≤ 1e-6), and joint samples exactly at 1/60 s;
- effort-signal verification (warning).

Unit tests: `python -m unittest discover -s tests` (`tests/test_energy_evaluation.py`; synthetic data only).

## Limitations

- **Measurement scope of VDD_IN:** VDD_IN is module input power (SoC, memory, regulators, fan supply rail as wired). It is not CPU or GPU power, and it includes the HTTP server and the sampler thread itself. At the lowest CPU frequency the sampler may run below its nominal 200 Hz; check `power_sample_rate_hz`.
- **Sensor refresh rate:** the INA3221 refresh rate (`power_sensor_update_rate_hz`) bounds how finely energy can be attributed to individual inferences. Short requests may be NaN by design.
- **No actuator model:** E_act and E_total need actuator parameters that are not in this repository. Any values supplied must be cited; with assumed values the results are model-based estimates.
- **E_mech is simulated:** it is the work of PhysX position drives in a simulated Franka, not a measured robot. The finger joints are excluded.
- **Wall vs simulation time:** computational energy is wall-clock while mechanical energy is simulation time. They are combined only under real-time pacing, and the residual mismatch is reported per episode.
- **Throttling detection:** detection is indirect. It flags pinned-frequency deviations sampled at 10 Hz and logs SoC temperatures, but doesn't read throttle event counters.
- **Not comparable with `outputs/dvfs_corners`:** that run pinned CPU min at 422.4 MHz, while this protocol uses the device minimum, which may be lower. Treat them as separate experiments.
