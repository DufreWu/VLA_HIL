"""Pure Python metric definitions. Distances: metres; time: simulation seconds."""
from dataclasses import dataclass, asdict
import math
import statistics

import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    timeout_s: float = 30.0
    hold_s: float = 1.0
    lift_m: float = 0.10
    target_tolerance_m: float = 0.03
    surface_tolerance_m: float = 0.015
    max_speed_m_s: float = 0.03
    collision_force_n: float = 1.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.hold_s > self.timeout_s:
            raise ValueError('hold_s must not exceed timeout_s')


def average_joint_jerk(joint_positions, dt_s, smoothing_window=1):
    """Compute AJJ from the third derivative of seven arm-joint positions.

    We use a centered finite-difference estimate of the third derivative on the
    measured joint trajectory. A unit-sum moving average is used only when a
    non-default smoothing window is requested; the default uses the raw samples to
    preserve the exact scale implied by the metric definition.
    """
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError('Expected joint positions with shape [N, 7].')
    if q.shape[0] < 6:
        raise ValueError('Need at least 6 samples to estimate jerk reliably.')
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0:
        raise ValueError('dt_s must be finite and positive.')
    smoothing_window = int(smoothing_window)
    if smoothing_window < 1:
        raise ValueError('Smoothing window must be positive.')

    signal = q.copy()
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window, dtype=np.float64) / smoothing_window
        pad = smoothing_window // 2
        padded = np.pad(signal, ((pad, pad), (0, 0)), mode='edge')
        signal = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), 0, padded)

    d1 = np.gradient(signal, dt_s, axis=0, edge_order=2)
    d2 = np.gradient(d1, dt_s, axis=0, edge_order=2)
    d3 = np.gradient(d2, dt_s, axis=0, edge_order=2)
    rms_per_joint = np.sqrt(np.mean(d3 ** 2, axis=0))
    return float(np.mean(rms_per_joint))


def _percentile_higher(values, percent):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.sort(arr)[max(0, int(np.ceil(percent / 100.0 * arr.size)) - 1)])


def summarize_inference_samples(inference_ms, rtt_ms=None):
    inf = np.asarray(inference_ms, dtype=np.float64)
    valid_inf = inf[np.isfinite(inf) & (inf >= 0)]
    if valid_inf.size == 0:
        return dict(mean_inference_ms=None, p95_inference_ms=None,
                    mean_rtt_ms=None, number_of_inference_requests=0)
    stats = dict(mean_inference_ms=float(np.mean(valid_inf)),
                 p95_inference_ms=_percentile_higher(valid_inf, 95),
                 number_of_inference_requests=int(valid_inf.size))
    if rtt_ms is not None:
        rtt = np.asarray(rtt_ms, dtype=np.float64)
        valid_rtt = rtt[np.isfinite(rtt) & (rtt >= 0)]
        if valid_rtt.size:
            stats['mean_rtt_ms'] = float(np.mean(valid_rtt))
    return stats


class EpisodeMetrics:
    def __init__(self, task, initial, target, config):
        if task not in ('grasping', 'pushing'):
            raise ValueError(task)
        self.task, self.initial, self.target, self.config = task, initial, target, config
        self.elapsed = 0.0
        self.valid_since = None
        self.success = False
        self.completion_time = None
        self.collision = False
        self.reason = None
        self.push_lifted = False
        self.tct_wall_s = None
        self.tct_sim_s = None
        self.ajj = None
        self.inference_stats = {}

    @property
    def done(self):
        return self.success or self.reason is not None

    def update(self, t, position, velocity, both_fingers_contact=False, collision=False):
        if self.done:
            raise RuntimeError('Episode already finished')
        if not math.isfinite(t) or t <= self.elapsed:
            raise ValueError('Simulation timestamps must increase')
        if len(position) != 3 or len(velocity) != 3 or not all(
                math.isfinite(x) for x in (*position, *velocity)):
            raise ValueError('Expected finite xyz position and velocity')
        self.elapsed = t
        self.collision |= bool(collision)
        c = self.config
        if position[2] < -0.02:
            self.reason = 'object_fell'
            return
        if self.task == 'grasping':
            valid = position[2] >= self.initial[2] + c.lift_m and both_fingers_contact
        else:
            # Lifting then placing must not pass as a push.
            self.push_lifted |= position[2] > self.initial[2] + c.surface_tolerance_m
            valid = (not self.push_lifted
                     and math.dist(position[:2], self.target[:2]) <= c.target_tolerance_m
                     and abs(position[2] - self.initial[2]) <= c.surface_tolerance_m
                     and math.sqrt(sum(v*v for v in velocity)) <= c.max_speed_m_s)
        if valid:
            if self.valid_since is None:
                self.valid_since = t
            if t - self.valid_since + 1e-9 >= c.hold_s and t <= c.timeout_s + 1e-9:
                self.success, self.completion_time = True, t
                self.tct_sim_s = t
        else:
            self.valid_since = None
        if not self.success and t + 1e-9 >= c.timeout_s:
            self.reason = 'timeout'

    def result(self):
        if not self.done and self.tct_sim_s is None and self.tct_wall_s is None:
            raise RuntimeError('Cannot report an unfinished episode')
        result = dict(task=self.task, success=self.success, collision=self.collision,
                    completion_time_s=self.completion_time, duration_s=self.elapsed,
                    tct_sim_s=self.tct_sim_s, tct_wall_s=self.tct_wall_s,
                    ajj=self.ajj, reason='success' if self.success else self.reason)
        result.update(self.inference_stats)
        return result


def summarize(rows):
    if not rows:
        raise ValueError('No episodes to summarize')
    successes = [r['completion_time_s'] for r in rows if r['success']]
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                task_success_rate_pct=100*sum(r['success'] for r in rows)/len(rows),
                collision_rate_pct=100*sum(r['collision'] for r in rows)/len(rows),
                task_completion_time_s=statistics.mean(successes) if successes else None,
                completion_time_std_s=statistics.pstdev(successes) if successes else None,
                mean_episode_duration_s=statistics.mean(r['duration_s'] for r in rows))
