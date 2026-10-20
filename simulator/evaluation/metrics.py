"""Pure Python metric definitions. Distances: metres; time: simulation seconds."""
from dataclasses import dataclass, asdict
import math
import statistics


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
        else:
            self.valid_since = None
        if not self.success and t + 1e-9 >= c.timeout_s:
            self.reason = 'timeout'

    def result(self):
        if not self.done:
            raise RuntimeError('Cannot report an unfinished episode')
        return dict(task=self.task, success=self.success, collision=self.collision,
                    completion_time_s=self.completion_time, duration_s=self.elapsed,
                    reason='success' if self.success else self.reason)


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
