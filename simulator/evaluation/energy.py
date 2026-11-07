"""Energy, timing and logging definitions for the DVFS energy evaluation (no Isaac Sim / torch).

Clock domains (never subtracted across hosts):
  * Simulation time ``sim_t`` (s): physics clock, 0 at the start of the episode. Used for
    joint dynamics, mechanical/actuator energy, task timing and AJJ.
  * Client wall time ``wall_t`` (s): host ``time.perf_counter()`` minus the episode wall start,
    which is taken immediately after the Jetson acknowledged ``/episode/start``. Used for
    request timestamps, action availability/execution and the real-time factor.
  * Server time (s): Jetson ``time.monotonic()`` minus the server episode start. Used for VDD_IN
    samples and inference intervals. Only server-side durations are reported to the client.
  With wall-clock pacing (RTF ~ 1) the three intervals coincide up to the episode-start
  message latency (``episode_start_rtt_ms`` bounds the offset) and the RTF deviation.

Energy definitions (J):
  E_comp   integral of Jetson VDD_IN power over [server start, server start + active_s], where
           active_s is the client wall duration of the closed-loop episode (measured).
  E_inf_i  integral of VDD_IN over the model forward pass of request i (measured; unavailable
           when the sensor did not refresh at least twice inside the interval).
  EDP_i    E_inf_i * L_i, with L_i the forward-pass latency in s. EDP_comp = mean_i EDP_i.
  E_mech   integral of sum_j max(0, tau_j qdot_j) over simulation time (positive shaft work,
           simulated). E_mech_signed keeps the sign (no regeneration is assumed anywhere).
  E_act    integral of sum_j [max(0, tau_j qdot_j)/eta_j + alpha_j tau_j^2 + P0_j] (model-based
           estimate; only with an explicit parameter file). It already contains the mechanical
           work, so E_mech is never added to it.
  E_total  E_comp + E_act. Not defined when E_act is unavailable (E_comp + E_mech is never
           reported as a total electrical energy).
"""
from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ARM_JOINTS = [f'panda_joint{i}' for i in range(1, 8)]
NAN = float('nan')
HORIZON_TICKS = 50
FPS = 20


def nan_if_none(v):
    if v is None:
        return NAN
    try:
        v = float(v)
    except (TypeError, ValueError):
        return NAN
    return v


def finite(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(v)


def trapezoid(t, y) -> float:
    t, y = np.asarray(t, np.float64), np.asarray(y, np.float64)
    if t.size < 2:
        return NAN
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


# ------------------------------------------------------------------ actuator model

class ActuatorModelUnavailable(ValueError):
    pass


def load_actuator_model(path) -> dict[str, Any]:
    """Validated actuator model parameters with provenance.

    Raises ActuatorModelUnavailable if any parameter is missing (null) or invalid; the caller
    then reports E_mech only and marks E_act/E_total as unavailable.
    """
    import yaml
    data = yaml.safe_load(Path(path).read_text()) or {}
    provenance = str(data.get('provenance') or '').strip()
    if not provenance:
        raise ActuatorModelUnavailable(f'{path}: "provenance" is required (source of every parameter)')
    if 'estimate' not in data or not isinstance(data['estimate'], bool):
        raise ActuatorModelUnavailable(f'{path}: "estimate" must be true (assumed/fitted) or false (measured)')
    joints = data.get('joints') or {}
    eta, alpha, p0 = [], [], []
    for j in ARM_JOINTS:
        par = joints.get(j) or {}
        try:
            e, a, p = float(par['eta']), float(par['alpha_w_per_nm2']), float(par['p0_w'])
        except (KeyError, TypeError, ValueError):
            raise ActuatorModelUnavailable(f'{path}: {j} needs numeric eta, alpha_w_per_nm2 and p0_w') from None
        if not (0 < e <= 1 and a >= 0 and p >= 0) or not all(map(math.isfinite, (e, a, p))):
            raise ActuatorModelUnavailable(f'{path}: {j} requires 0 < eta <= 1, alpha >= 0, p0 >= 0')
        eta.append(e); alpha.append(a); p0.append(p)
    return dict(path=str(path), provenance=provenance, estimate=data['estimate'],
                eta=eta, alpha_w_per_nm2=alpha, p0_w=p0, joints=ARM_JOINTS,
                label='model-based estimate' if data['estimate'] else 'model with measured parameters')


# ------------------------------------------------------------------ per-episode energy

def mechanical_energy(t, qd, tau) -> dict[str, float]:
    """Arm shaft power/energy from simulation-time samples t (N,), qd/tau (N, 7)."""
    t = np.asarray(t, np.float64)
    p = np.asarray(tau, np.float64) * np.asarray(qd, np.float64)
    if t.size < 2 or p.shape != (t.size, 7) or not np.all(np.isfinite(p)):
        return dict(mechanical_energy_j=NAN, mechanical_energy_signed_j=NAN,
                    mechanical_power_w=NAN, mechanical_peak_power_w=NAN)
    pos = np.maximum(p, 0.0).sum(axis=1)
    duration = t[-1] - t[0]
    e_pos = trapezoid(t, pos)
    return dict(mechanical_energy_j=e_pos,
                mechanical_energy_signed_j=trapezoid(t, p.sum(axis=1)),
                mechanical_power_w=e_pos / duration if duration > 0 else NAN,
                mechanical_peak_power_w=float(pos.max()))


def actuator_energy(t, qd, tau, model) -> dict[str, float]:
    if model is None:
        return dict(actuator_energy_j=NAN, actuator_power_w=NAN)
    t = np.asarray(t, np.float64)
    tau = np.asarray(tau, np.float64)
    p_mech = np.maximum(tau * np.asarray(qd, np.float64), 0.0)
    p = (p_mech / np.asarray(model['eta']) + np.asarray(model['alpha_w_per_nm2']) * tau ** 2
         + np.asarray(model['p0_w'])).sum(axis=1)
    if t.size < 2 or not np.all(np.isfinite(p)):
        return dict(actuator_energy_j=NAN, actuator_power_w=NAN)
    e = trapezoid(t, p)
    return dict(actuator_energy_j=e, actuator_power_w=e / (t[-1] - t[0]))


def inference_energy_stats(inferences: list[dict[str, Any]]) -> dict[str, float]:
    """Episode means of per-request E_inf_i and EDP_i; NaN unless every request is attributable."""
    if not inferences:
        return dict(inference_energy_j=NAN, inference_energy_total_j=NAN, computational_edp_js=NAN,
                    inference_power_w=NAN, n_inference_energy=0)
    e = np.array([nan_if_none(r.get('energy_j')) for r in inferences])
    lat = np.array([nan_if_none(r.get('latency_s')) for r in inferences])
    ok = np.isfinite(e) & np.isfinite(lat) & (lat > 0)
    if not ok.all():
        return dict(inference_energy_j=NAN, inference_energy_total_j=NAN, computational_edp_js=NAN,
                    inference_power_w=NAN, n_inference_energy=int(ok.sum()))
    return dict(inference_energy_j=float(e.mean()), inference_energy_total_j=float(e.sum()),
                computational_edp_js=float((e * lat).mean()),
                inference_power_w=float(e.sum() / lat.sum()), n_inference_energy=int(e.size))


def inverse_dynamics_check(samples) -> dict[str, Any]:
    """Compare logged joint efforts with tau_id = M(q) qddot + c(q, qdot) + g(q) (arm joints 1-7).

    ``samples``: per physics step (t, qd[9], tau[9], M[9x9], c[9], g[9]); qddot by central
    differences of qd. Contacts (grasp, pushing) add J^T F terms that tau_id omits, so the
    agreement is reported, not forced; a high correlation shows the effort is the simulated
    joint load (gravity, inertia) and not the commanded target.
    """
    t = np.array([x[0] for x in samples])
    qd = np.array([x[1] for x in samples])
    tau = np.array([x[2] for x in samples])[1:-1, :7]
    qdd = (qd[2:] - qd[:-2]) / (t[2:] - t[:-2])[:, None]
    tau_id = np.array([m[:7] @ a + c[:7] + g[:7] for (_, _, _, m, c, g), a in zip(samples[1:-1], qdd)])
    corr = [float(np.corrcoef(tau[:, j], tau_id[:, j])[0, 1]) if tau[:, j].std() > 1e-9 and tau_id[:, j].std() > 1e-9
            else None for j in range(7)]
    rms = np.sqrt((tau ** 2).mean(axis=0))
    res = np.sqrt(((tau - tau_id) ** 2).mean(axis=0))
    return dict(n_samples=int(len(tau)), corr_per_joint=corr, rms_effort_nm=rms.tolist(),
                rms_residual_nm=res.tolist(), relative_residual=float(res.sum() / max(rms.sum(), 1e-12)),
                definition='tau_id = M(q) qddot + c(q, qdot) + g(q); qddot by central differences at 60 Hz')


def failure_reason(raw_reason: str | None, success: bool, received_chunk: bool) -> str:
    """Map the simulator's termination reason to the reporting categories."""
    if success:
        return ''
    if raw_reason == 'inference_failure':
        return 'inference_or_communication_failure'
    if raw_reason == 'action_starvation' or not received_chunk:
        return 'no_valid_action'
    if raw_reason == 'object_fell':
        return 'constraint_violation'
    if raw_reason == 'timeout':
        return 'timeout_task_not_reached'
    return str(raw_reason or 'unknown')


# ------------------------------------------------------------------ CSV schemas

EPISODE_COLUMNS = [
    # identity
    'experiment_id', 'episode_uid', 'configuration', 'task', 'seed', 'episode', 'order_in_episode',
    # task outcome (simulation time)
    'success', 'failure_reason', 'raw_reason', 'collision', 'episode_duration_s', 'completion_time_s',
    'task_progress', 'ajj',
    # DVFS
    'cpu_mhz_requested', 'gpu_mhz_requested', 'cpu_mhz_actual', 'gpu_mhz_actual',
    'cpu_mhz_min', 'cpu_mhz_max', 'gpu_mhz_min', 'gpu_mhz_max', 'cpu_governor', 'gpu_governor',
    'power_mode', 'frequency_valid', 'frequency_stable', 'soc_temp_start_c', 'soc_temp_end_c',
    # inference timing (server forward pass / client round trip)
    'inference_ms_mean', 'inference_ms_p95', 'rtt_ms_mean', 'communication_ms_mean', 'n_inference_requests',
    # computational energy (measured, wall clock)
    'computational_power_w', 'computational_energy_j', 'computational_energy_window_j',
    'inference_energy_j', 'inference_energy_total_j', 'inference_power_w', 'computational_edp_js',
    'n_inference_energy', 'power_sample_rate_hz', 'power_sensor_update_rate_hz',
    # mechanical (simulated) and actuator (model-based) energy, simulation time
    'mechanical_power_w', 'mechanical_energy_j', 'mechanical_energy_signed_j', 'mechanical_peak_power_w',
    'actuator_power_w', 'actuator_energy_j', 'total_energy_j', 'total_power_w',
    # action timing
    'action_delay_ms', 'action_delay_ticks', 'obs_age_ms', 'holding_time_s', 'initial_wait_s',
    'underrun_s', 'buffer_underrun_rate', 'dropped_action_slots', 'mean_buffer_occupancy',
    # pacing / time alignment
    'wall_duration_s', 'comp_window_s', 'real_time_factor', 'deadline_misses', 'time_alignment_error_s',
    'episode_start_rtt_ms',
]

INFERENCE_COLUMNS = [
    'experiment_id', 'episode_uid', 'configuration', 'task', 'seed', 'episode', 'request_index', 'obs_tick',
    # client wall clock, s from episode wall start
    'capture_wall_s', 'encoded_wall_s', 'sent_wall_s', 'received_wall_s', 'available_wall_s', 'first_exec_wall_s',
    # control ticks (sim time = tick / 20 s)
    'available_tick', 'release_tick', 'first_exec_tick',
    # durations
    'inference_ms', 'rtt_ms', 'server_total_ms', 'server_pre_ms', 'communication_ms', 'end_to_end_ms',
    'obs_to_action_ms', 'action_delay_ticks', 'exceeds_horizon',
    # server clock, s from server episode start
    'server_inference_start_s', 'server_inference_end_s',
    'inference_energy_j', 'inference_power_w', 'inference_edp_js', 'n_power_samples', 'n_sensor_updates',
    'executed_slots', 'dropped_slots', 'used', 'drained', 'error',
]

POWER_COLUMNS = ['episode_uid', 'server_t_s', 'power_w']
JOINT_COLUMNS = (['episode_uid', 'sim_t_s']
                 + [f'q{j}_rad' for j in range(1, 8)] + [f'qd{j}_rad_s' for j in range(1, 8)]
                 + [f'tau{j}_nm' for j in range(1, 8)])
TICK_COLUMNS = ['episode_uid', 'tick', 'sim_t_s', 'wall_t_s', 'buffer_after', 'source_obs_tick', 'held']


def _fmt(v):
    if v is None:
        return 'nan'
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)  # repr of NumPy 2 scalars would write 'np.float64(...)'
        return 'nan' if math.isnan(v) else repr(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


class ExperimentWriter:
    """Append-only CSV writers; raw high-rate streams are gzip-compressed."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.files = {}
        self.specs = dict(episodes=('episodes.csv', EPISODE_COLUMNS),
                          inference=('inference_requests.csv', INFERENCE_COLUMNS),
                          power=('power_samples.csv.gz', POWER_COLUMNS),
                          joints=('joint_samples.csv.gz', JOINT_COLUMNS),
                          ticks=('action_ticks.csv.gz', TICK_COLUMNS))

    def _writer(self, kind):
        if kind not in self.files:
            name, cols = self.specs[kind]
            path = self.root / name
            new = not path.exists()
            fh = gzip.open(path, 'at', newline='') if name.endswith('.gz') else path.open('a', newline='')
            w = csv.writer(fh)
            if new:
                w.writerow(cols)
            self.files[kind] = (fh, w, cols)
        return self.files[kind]

    def write(self, kind, rows: Iterable[dict[str, Any] | list]):
        fh, w, cols = self._writer(kind)
        for r in rows:
            w.writerow([_fmt(x) for x in (r if isinstance(r, (list, tuple)) else [r.get(c) for c in cols])])
        fh.flush()

    def close(self):
        for fh, _, _ in self.files.values():
            fh.close()
        self.files = {}


def read_csv(path) -> list[dict[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as fh:
        return list(csv.DictReader(fh))


def to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def request_rows(requests: dict[int, dict[str, Any]], wall_start: float, server_inferences: list[dict],
                 ident: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the client's per-request records and join server energy by obs_id."""
    by_obs = {int(r['obs_id']): r for r in server_inferences if r.get('obs_id') is not None}
    rows = []
    for i, (obs, rec) in enumerate(sorted(requests.items())):
        rel = lambda k: (rec[k] - wall_start) if rec.get(k) is not None else NAN
        st = rec.get('server_timing_ms') or {}
        srv = by_obs.get(int(obs), {})
        lat_s = nan_if_none(srv.get('latency_s'))
        e = nan_if_none(srv.get('energy_j'))
        rtt = nan_if_none(rec.get('rtt_ms'))
        total = nan_if_none(st.get('total'))
        delay_ticks = (rec['first_exec_tick'] - rec['obs_tick']) if rec.get('first_exec_tick') is not None else NAN
        rows.append(dict(
            ident, request_index=i, obs_tick=rec['obs_tick'],
            capture_wall_s=rel('capture'), encoded_wall_s=rel('encoded'), sent_wall_s=rel('sent'),
            received_wall_s=rel('received'), available_wall_s=rel('available'), first_exec_wall_s=rel('first_exec'),
            available_tick=rec.get('available_tick'), release_tick=rec.get('release_tick'),
            first_exec_tick=rec.get('first_exec_tick'),
            inference_ms=nan_if_none(rec.get('inference_ms')), rtt_ms=rtt, server_total_ms=total,
            server_pre_ms=nan_if_none(st.get('receive_to_parsed')) + nan_if_none(st.get('parsed_to_inference_start')),
            communication_ms=rtt - total,
            end_to_end_ms=(rec['received'] - rec['capture']) * 1000 if rec.get('received') else NAN,
            obs_to_action_ms=(rec['first_exec'] - rec['capture']) * 1000 if rec.get('first_exec') else NAN,
            action_delay_ticks=delay_ticks,
            exceeds_horizon=(rec['received'] - rec['capture']) * FPS > HORIZON_TICKS if rec.get('received') else None,
            server_inference_start_s=nan_if_none(srv.get('start_s')),
            server_inference_end_s=nan_if_none(srv.get('end_s')),
            inference_energy_j=e, inference_power_w=nan_if_none(srv.get('mean_power_w')),
            inference_edp_js=e * lat_s,
            n_power_samples=srv.get('n_power_samples'), n_sensor_updates=srv.get('n_sensor_updates'),
            executed_slots=rec.get('executed_slots', 0), dropped_slots=rec.get('dropped_slots'),
            used=rec.get('used'), drained=rec.get('drained'), error=rec.get('error', '')))
    return rows


def _within(value, target, rel=0.02, abs_tol=0.0):
    return finite(value) and finite(target) and abs(value - target) <= max(abs_tol, rel * target)


def episode_row(ident: dict[str, Any], result: dict[str, Any], energy: dict[str, Any],
                mech: dict[str, float], act: dict[str, float], req_rows: list[dict[str, Any]],
                switch: dict[str, Any] | None, governed: bool) -> dict[str, Any]:
    """One flat episodes.csv row. Missing measurements are NaN, never 0."""
    switch = switch or {}
    energy = energy or {}
    lat = np.array([r['inference_ms'] for r in req_rows if finite(r['inference_ms'])])
    rtt = np.array([r['rtt_ms'] for r in req_rows if finite(r['rtt_ms'])])
    comm = np.array([r['communication_ms'] for r in req_rows if finite(r['communication_ms'])])
    used = [r for r in req_rows if r.get('used')]
    delay_ms = np.array([r['obs_to_action_ms'] for r in used if finite(r['obs_to_action_ms'])])
    delay_ticks = np.array([r['action_delay_ticks'] for r in used if finite(r['action_delay_ticks'])])
    # Every request that ran on the device this episode (incl. one drained after the episode).
    ran = [r for r in req_rows if not r.get('error')]
    inf = inference_energy_stats([dict(energy_j=r['inference_energy_j'] if finite(r['inference_energy_j']) else None,
                                       latency_s=r['server_inference_end_s'] - r['server_inference_start_s'])
                                  for r in ran] if energy.get('inferences') is not None else [])
    req_cpu = nan_if_none(switch.get('requested_cpu_khz'))
    req_gpu = nan_if_none(switch.get('requested_gpu_hz'))
    cpu_lo, cpu_hi = nan_if_none(energy.get('min_cpu_khz')), nan_if_none(energy.get('max_cpu_khz'))
    gpu_lo, gpu_hi = nan_if_none(energy.get('min_gpu_hz')), nan_if_none(energy.get('max_gpu_hz'))
    if governed:
        stable = None  # Governors change frequency by design.
    elif all(map(finite, (cpu_lo, cpu_hi, gpu_lo, gpu_hi))):
        stable = (_within(cpu_lo, req_cpu, abs_tol=20_000) and _within(cpu_hi, req_cpu, abs_tol=20_000)
                  and _within(gpu_lo, req_gpu, abs_tol=5e6) and _within(gpu_hi, req_gpu, abs_tol=5e6))
    else:
        stable = None
    e_comp = nan_if_none(energy.get('energy_active_j'))
    comp_s = nan_if_none(energy.get('active_s'))
    e_act = act['actuator_energy_j']
    duration = float(result['duration_s'])
    total = e_comp + e_act if finite(e_comp) and finite(e_act) else NAN
    wall = nan_if_none(result.get('wall_time_s'))
    temps = lambda k: max(energy.get(k, {}).values(), default=None) if energy.get(k) else None
    governors = switch.get('cpu_governors') or [None]
    return dict(
        ident,
        success=bool(result['success']), failure_reason=result['failure_reason'], raw_reason=result.get('reason'),
        collision=bool(result['collision']), episode_duration_s=duration,
        completion_time_s=nan_if_none(result.get('completion_time_s')) if result['success'] else NAN,
        task_progress=nan_if_none(result.get('task_progress')), ajj=nan_if_none(result.get('ajj')),
        cpu_mhz_requested=req_cpu / 1e3, gpu_mhz_requested=req_gpu / 1e6,
        cpu_mhz_actual=nan_if_none(energy.get('mean_cpu_khz')) / 1e3,
        gpu_mhz_actual=nan_if_none(energy.get('mean_gpu_hz')) / 1e6,
        cpu_mhz_min=cpu_lo / 1e3, cpu_mhz_max=cpu_hi / 1e3, gpu_mhz_min=gpu_lo / 1e6, gpu_mhz_max=gpu_hi / 1e6,
        cpu_governor=governors[0], gpu_governor=switch.get('gpu_governor'), power_mode=switch.get('power_mode'),
        frequency_valid=switch.get('frequency_valid') if switch else None, frequency_stable=stable,
        soc_temp_start_c=nan_if_none(temps('temps_start_c')), soc_temp_end_c=nan_if_none(temps('temps_end_c')),
        inference_ms_mean=float(lat.mean()) if lat.size else NAN,
        inference_ms_p95=float(np.percentile(lat, 95)) if lat.size else NAN,
        rtt_ms_mean=float(rtt.mean()) if rtt.size else NAN,
        communication_ms_mean=float(comm.mean()) if comm.size else NAN,
        n_inference_requests=len(req_rows),
        computational_power_w=e_comp / comp_s if finite(e_comp) and finite(comp_s) and comp_s > 0 else NAN,
        computational_energy_j=e_comp, computational_energy_window_j=nan_if_none(energy.get('energy_j')),
        **inf,
        power_sample_rate_hz=nan_if_none(energy.get('sample_rate_hz')),
        power_sensor_update_rate_hz=nan_if_none(energy.get('sensor_update_rate_hz')),
        **mech, **act,
        total_energy_j=total, total_power_w=total / duration if finite(total) and duration > 0 else NAN,
        action_delay_ms=float(delay_ms.mean()) if delay_ms.size else NAN,
        action_delay_ticks=float(delay_ticks.mean()) if delay_ticks.size else NAN,
        obs_age_ms=nan_if_none(result.get('mean_obs_age_ticks')) * 1000 / FPS,
        holding_time_s=nan_if_none(result.get('holding_duration_s')),
        initial_wait_s=nan_if_none(result.get('initial_wait_s')),
        underrun_s=nan_if_none(result.get('underrun_s')),
        buffer_underrun_rate=nan_if_none(result.get('buffer_underrun_rate')),
        dropped_action_slots=nan_if_none(result.get('dropped_action_slots')),
        mean_buffer_occupancy=nan_if_none(result.get('mean_queue_length')),
        wall_duration_s=wall, comp_window_s=comp_s,
        real_time_factor=nan_if_none(result.get('real_time_factor')),
        deadline_misses=result.get('deadline_misses'),
        time_alignment_error_s=comp_s - duration if finite(comp_s) else NAN,
        episode_start_rtt_ms=nan_if_none(result.get('episode_start_rtt_ms')))
