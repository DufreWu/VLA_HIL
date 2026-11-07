"""DVFS switching and episode-level VDD_IN energy accounting for closed-loop runs.

No torch dependency, so it can be unit-tested anywhere. Writing frequencies
requires root on the Jetson; reading power only needs the INA3221 hwmon node.
All timestamps are the Jetson's own ``time.monotonic()``; no cross-host clock is
ever subtracted.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[2]
BOARD_CONFIG = ROOT / 'jetson_orin_nx' / 'benchmark' / 'jetson_orin_nx.yaml'
CLOSED_LOOP_CONFIG = ROOT / 'jetson_orin_nx' / 'benchmark' / 'closed_loop_dvfs.yaml'


DEFAULT_GOVERNORS = dict(cpu='schedutil', gpu='nvhost_podgov')


def _freq_value(name, value):
    """A positive integer frequency, or the symbols 'min'/'max' resolved on the device."""
    if isinstance(value, str) and value.strip().lower() in ('min', 'max'):
        return value.strip().lower()
    value = int(value)
    if value <= 0:
        raise ValueError(f'{name}: frequencies must be positive')
    return value


def load_dvfs_configs(path=CLOSED_LOOP_CONFIG, names=None) -> list[dict[str, Any]]:
    """Return validated configs in file order.

    Fixed configs: ``{name, cpu_khz, gpu_hz}``; a frequency may be ``min``/``max``, which the
    server resolves against the frequencies the device itself lists as available. Governor configs: ``{name, governor: default}``
    (stock governors) or ``{name, cpu_governor, gpu_governor}``; their cpu_khz/gpu_hz are None.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    configs = data.get('configs')
    if not isinstance(configs, list) or not configs:
        raise ValueError(f'{path}: expected a non-empty "configs" list')
    out, seen = [], set()
    for item in configs:
        if 'grid' in item:
            grid = item['grid'] or {}
            for key in ('cpu_khz', 'gpu_hz'):
                v = grid.get(key)
                if not (v in ('all', 'board') or (isinstance(v, list) and v)):
                    raise ValueError(f'grid.{key} must be "all" (device list), "board" (jetson_orin_nx.yaml) '
                                     'or a non-empty list')
            out.append(dict(grid=dict(cpu_khz=grid['cpu_khz'], gpu_hz=grid['gpu_hz']), name='__grid__'))
            continue
        name = str(item['name'])
        if name in seen:
            raise ValueError(f'Duplicate DVFS config name {name!r}')
        seen.add(name)
        if 'governor' in item or 'cpu_governor' in item or 'gpu_governor' in item:
            if 'cpu_khz' in item or 'gpu_hz' in item:
                raise ValueError(f'{name}: use either a governor or fixed frequencies, not both')
            gov = item.get('governor')
            if gov not in (None, 'default'):
                raise ValueError(f'{name}: governor must be "default" (or set cpu_governor/gpu_governor)')
            out.append(dict(item, name=name, cpu_khz=None, gpu_hz=None,
                            cpu_governor=str(item.get('cpu_governor', DEFAULT_GOVERNORS['cpu'])),
                            gpu_governor=str(item.get('gpu_governor', DEFAULT_GOVERNORS['gpu']))))
            continue
        cpu, gpu = _freq_value(name, item['cpu_khz']), _freq_value(name, item['gpu_hz'])
        out.append(dict(item, name=name, cpu_khz=cpu, gpu_hz=gpu))
    if any('grid' in c for c in out):
        if names:
            raise ValueError('--dvfs-names is applied after grid expansion (see expand_dvfs_grid)')
        return out
    return select_dvfs_configs(out, names)


def pick_frequency(available, which: str) -> int:
    """'min'/'max' of the device list, or 'mid': the listed value closest to (min + max) / 2."""
    listed = sorted(int(x) for x in available)
    if which == 'min':
        return listed[0]
    if which == 'max':
        return listed[-1]
    if which == 'mid':
        centre = (listed[0] + listed[-1]) / 2
        return min(listed, key=lambda f: (abs(f - centre), f))
    raise ValueError(f'Unknown frequency selector {which!r}')


def grid_name(cpu_khz: int, gpu_hz: int) -> str:
    return f'cpu{cpu_khz / 1000:g}_gpu{gpu_hz / 1e6:g}'


def expand_dvfs_grid(configs: list[dict[str, Any]], caps: dict[str, Any] | None = None,
                     board_config=BOARD_CONFIG, names=None) -> list[dict[str, Any]]:
    """Replace ``grid`` entries by one fixed config per (CPU, GPU) pair, CPU-major ascending.

    ``all`` takes the frequencies the device lists (``caps`` from /dvfs/capabilities), ``board``
    the cpu_freqs/gpu_freqs of jetson_orin_nx.yaml; explicit lists are used as given (the server
    still rejects values the device does not list).
    """
    out = []
    board = None
    for c in configs:
        if 'grid' not in c:
            out.append(c)
            continue
        lists = []
        for key, cap_key, board_key in (('cpu_khz', 'cpu_khz_available', 'cpu_freqs'),
                                        ('gpu_hz', 'gpu_hz_available', 'gpu_freqs')):
            v = c['grid'][key]
            if isinstance(v, list) and any(x in ('min', 'mid', 'max') for x in v):
                if not caps or not caps.get(cap_key):
                    raise ValueError(f'grid.{key}: min/mid/max need the device frequency list (/dvfs/capabilities)')
                v = [pick_frequency(caps[cap_key], x) if isinstance(x, str) else x for x in v]
            elif v == 'all':
                if not caps or not caps.get(cap_key):
                    raise ValueError(f'grid.{key}: all needs the device frequency list (/dvfs/capabilities)')
                v = caps[cap_key]
            elif v == 'board':
                board = board or yaml.safe_load(Path(board_config).read_text())['jetson_orin_nx_16g']
                v = board[board_key]
            lists.append(sorted({int(x) for x in v}))
        for cpu in lists[0]:
            for gpu in lists[1]:
                out.append(dict(name=grid_name(cpu, gpu), cpu_khz=cpu, gpu_hz=gpu, grid_point=True))
    seen = set()
    for c in out:
        if c['name'] in seen:
            raise ValueError(f"Duplicate DVFS config name {c['name']!r} after grid expansion")
        seen.add(c['name'])
    return select_dvfs_configs(out, names)


def select_dvfs_configs(out, names):
    if names:
        by_name = {c['name']: c for c in out}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f'Unknown DVFS config(s) {missing}; available: {sorted(by_name)}')
        out = [by_name[n] for n in names]
    return out


def read_text(path) -> str | None:
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return fh.read().strip()
    except OSError:
        return None


def write_text(path, value) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(str(value))


def resolve_ina_path(ina_path: str | None) -> str | None:
    """hwmonN numbering changes across boots; fall back to the device's only hwmon dir."""
    if not ina_path or os.path.exists(os.path.join(ina_path, 'in1_input')):
        return ina_path
    parent = Path(ina_path).parent
    if parent.name == 'hwmon':
        found = sorted(parent.glob('hwmon*/in1_input'))
        if len(found) == 1:
            return str(found[0].parent)
    return ina_path


def read_vdd_in_power(ina_path: str | None) -> float | None:
    """Board input power in W (INA3221 channel 1: mV * mA)."""
    if not ina_path:
        return None
    v = read_text(os.path.join(ina_path, 'in1_input'))
    i = read_text(os.path.join(ina_path, 'curr1_input'))
    try:
        return (float(v) / 1000.0) * (float(i) / 1000.0)
    except (TypeError, ValueError):
        return None


def read_power_mode() -> str | None:
    """Active nvpmodel power mode (e.g. 'NV Power Mode: MAXN'), or None if unavailable."""
    import subprocess
    try:
        out = subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return ' / '.join(lines[:2]) if lines else None


def read_temperatures() -> dict[str, float]:
    temps = {}
    for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        name, raw = read_text(zone / 'type'), read_text(zone / 'temp')
        try:
            temps[name or zone.name] = float(raw) / 1000.0
        except (TypeError, ValueError):
            continue
    return temps


def integrate(samples: list[tuple[float, float]], start: float, end: float) -> tuple[float | None, int]:
    """Trapezoidal integral of (t, P) samples over [start, end], linearly clipped.

    Returns (energy_J or None if fewer than two samples cover the window, n_samples_inside).
    """
    import bisect
    if end <= start:
        return None, 0
    pts = sorted(samples)
    times = [t for t, _ in pts]
    lo, hi = bisect.bisect_right(times, start), bisect.bisect_left(times, end)
    if len(pts) < 2 or pts[0][0] > start or pts[-1][0] < end:
        return None, bisect.bisect_right(times, end) - bisect.bisect_left(times, start)

    def interp(t):
        i = min(max(bisect.bisect_right(times, t), 1), len(pts) - 1)
        (t0, p0), (t1, p1) = pts[i - 1], pts[i]
        return p0 if t1 == t0 else p0 + (p1 - p0) * (t - t0) / (t1 - t0)

    seq = [(start, interp(start)), *pts[lo:hi], (end, interp(end))]
    energy = sum(0.5 * (p0 + p1) * (t1 - t0) for (t0, p0), (t1, p1) in zip(seq, seq[1:]))
    return float(energy), hi - lo


def sensor_updates(samples: list[tuple[float, float]], start: float, end: float) -> int:
    """Number of reading changes strictly inside (start, end): a lower bound on fresh sensor
    conversions. Polling faster than the INA3221 refresh repeats the same value."""
    vals = [p for t, p in samples if start < t < end]
    return sum(a != b for a, b in zip(vals, vals[1:]))


# A per-inference energy is only reported when the sensor refreshed at least this many
# times inside the inference interval; otherwise it cannot be attributed to that request.
MIN_UPDATES_PER_INFERENCE = 2


class PowerMonitor:
    """Background VDD_IN sampler with server-clock episode windows."""

    def __init__(self, ina_path: str | None, interval_s: float = 0.005,
                 reader: Callable[[], float | None] | None = None, clock=time.monotonic,
                 freq_reader: Callable[[], tuple[float | None, float | None]] | None = None,
                 freq_interval_s: float = 0.1):
        self.interval_s = max(float(interval_s), 0.001)
        ina_path = resolve_ina_path(ina_path)
        self.reader = reader or (lambda: read_vdd_in_power(ina_path))
        self.clock = clock
        # Optional (mean CPU kHz, GPU Hz) sampler, so governor-controlled configs can be
        # reported by their actual operating frequency.
        self.freq_reader = freq_reader
        self.freq_every = max(1, round(float(freq_interval_s) / self.interval_s))
        self.n_sampled = 0
        self.freq_samples: list[tuple[float, float | None, float | None]] = []
        self.samples: list[tuple[float, float]] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.episode: dict[str, Any] | None = None

    def start(self):
        if self.reader() is None:
            raise RuntimeError('VDD_IN power sensor not readable; check ina_path in jetson_orin_nx.yaml')
        self.thread = threading.Thread(target=self._run, daemon=True, name='vdd-in-sampler')
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.is_set():
            self.sample_once()
            self.stop_event.wait(self.interval_s)

    def sample_once(self):
        t, p = self.clock(), self.reader()
        freq = None
        if self.freq_reader is not None and self.n_sampled % self.freq_every == 0:
            freq = self.freq_reader()
        self.n_sampled += 1
        with self.lock:
            if p is not None:
                self.samples.append((t, float(p)))
            if freq is not None:
                self.freq_samples.append((t, *freq))
            if self.episode is None:  # Keep memory bounded between episodes.
                if len(self.samples) > 20_000:
                    del self.samples[:-2_000]
                if len(self.freq_samples) > 2_000:
                    del self.freq_samples[:-200]

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def begin_episode(self, tag: str):
        # Slow sysfs reads first: the window must start as close as possible to the reply,
        # because the client starts its wall clock when the reply arrives.
        temps = read_temperatures()
        self.episode = dict(tag=str(tag), start=self.clock(), inferences=[], temps_start=temps)
        return dict(tag=self.episode['tag'])

    def record_inference(self, start: float, end: float, key: dict[str, Any] | None = None):
        """Register an inference interval [start, end] (server monotonic s) for the open episode."""
        if self.episode is not None:
            self.episode['inferences'].append((float(start), float(end), dict(key or {})))

    def end_episode(self, tag: str, settle_s: float | None = None, active_s: float | None = None,
                    include_samples: bool = False) -> dict[str, Any]:
        """Close the episode window and integrate VDD_IN.

        ``energy_j`` covers the whole window [begin, end request]. If ``active_s`` is given
        (the client's wall-clock duration of the closed-loop episode, measured from the reply
        to begin_episode), ``energy_active_j`` covers [begin, begin + active_s] only, which
        excludes draining the last in-flight request. Per-inference energies are integrated
        over each inference's own interval. All times are this host's monotonic clock.
        """
        if self.episode is None or self.episode['tag'] != str(tag):
            raise ValueError(f'No open energy episode with tag {tag!r}')
        end = self.clock()
        # Ensure a sample exists at or after the end of the window.
        deadline = end + (settle_s if settle_s is not None else 4 * self.interval_s)
        while self.clock() < deadline:
            time.sleep(self.interval_s)
        if self.thread is None:
            self.sample_once()
        ep, self.episode = self.episode, None
        with self.lock:
            samples = list(self.samples)
            freqs = [f for f in self.freq_samples if ep['start'] <= f[0] <= end]
        start = ep['start']
        energy, n = integrate(samples, start, end)

        def stat_of(idx, fn):
            vals = [f[idx] for f in freqs if f[idx] is not None]
            return fn(vals) if vals else None
        mean = lambda v: sum(v) / len(v)
        wall = end - start
        active = None if active_s is None else min(max(float(active_s), 0.0), wall)
        energy_active = integrate(samples, start, start + active)[0] if active else None
        inf_energy, inf_time, inferences = 0.0, 0.0, []
        for s0, s1, key in ep['inferences']:
            e_j, n_in = integrate(samples, s0, s1)
            updates = sensor_updates(samples, s0, s1)
            attributable = e_j is not None and updates >= MIN_UPDATES_PER_INFERENCE
            inferences.append(dict(key, start_s=s0 - start, end_s=s1 - start, latency_s=s1 - s0,
                                   energy_j=e_j if attributable else None,
                                   mean_power_w=e_j / (s1 - s0) if attributable and s1 > s0 else None,
                                   n_power_samples=n_in, n_sensor_updates=updates))
            inf_energy = None if inf_energy is None or e_j is None else inf_energy + e_j
            inf_time += s1 - s0
        window = [(t, p) for t, p in samples if start <= t <= end]
        out = dict(tag=ep['tag'], wall_s=wall, energy_j=energy,
                   mean_power_w=None if energy is None else energy / wall,
                   n_power_samples=n,
                   active_s=active, energy_active_j=energy_active,
                   mean_power_active_w=energy_active / active if energy_active is not None and active else None,
                   sample_rate_hz=len(window) / wall if wall > 0 else None,
                   sensor_update_rate_hz=sensor_updates(samples, start, end) / wall if wall > 0 else None,
                   inference_energy_j=inf_energy,
                   inference_busy_s=inf_time,
                   inference_duty_cycle=inf_time / wall if wall > 0 else None,
                   n_inferences=len(ep['inferences']), inferences=inferences,
                   mean_cpu_khz=stat_of(1, mean), mean_gpu_hz=stat_of(2, mean),
                   min_cpu_khz=stat_of(1, min), max_cpu_khz=stat_of(1, max),
                   min_gpu_hz=stat_of(2, min), max_gpu_hz=stat_of(2, max),
                   n_freq_samples=len(freqs),
                   temps_start_c=ep['temps_start'], temps_end_c=read_temperatures())
        if include_samples:
            out['power_samples'] = [[round(t - start, 6), round(p, 4)] for t, p in window]
        return out


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class DvfsController:
    """Pin CPU/GPU frequencies and restore the original sysfs state on close."""

    def __init__(self, board_config=BOARD_CONFIG):
        cfg = yaml.safe_load(Path(board_config).read_text())['jetson_orin_nx_16g']
        self.cpu_path = Path(cfg['cpu_path'])
        self.gpu_path = Path(cfg['gpu_path'])
        self.emc_cap_path = cfg.get('emc_cap_path')
        self.ina_path = resolve_ina_path(cfg.get('ina_path'))
        self.n_cpus = int(cfg['cpu_cores_num'])
        self.snapshot = None
        self.current: dict[str, Any] | None = None

    def _cpu(self, i, node):
        return self.cpu_path / f'cpu{i}' / 'cpufreq' / node

    def take_snapshot(self):
        cpus = [{n: read_text(self._cpu(i, n)) for n in
                 ('scaling_governor', 'scaling_min_freq', 'scaling_max_freq')}
                for i in range(self.n_cpus)]
        gpu = {n: read_text(self.gpu_path / n) for n in ('governor', 'min_freq', 'max_freq')}
        if any(v is None for c in cpus for v in c.values()) or any(v is None for v in gpu.values()):
            raise RuntimeError('Cannot read CPU/GPU DVFS sysfs nodes; check paths in jetson_orin_nx.yaml')
        self.snapshot = dict(cpus=cpus, gpu=gpu)
        return self.snapshot

    def check_writable(self):
        nodes = [self._cpu(0, 'scaling_governor'), self.gpu_path / 'governor']
        if not all(os.access(p, os.W_OK) for p in nodes):
            raise PermissionError('DVFS sysfs nodes are not writable; run the server with sudo.')

    def capabilities(self) -> dict[str, Any]:
        """Frequencies the device itself lists as available, plus the active power mode.

        CPU: intersection of every core's ``scaling_available_frequencies`` (falls back to
        the cpuinfo min/max). GPU: devfreq ``available_frequencies``.
        """
        per_core, bounds = [], []
        for i in range(self.n_cpus):
            listed = (read_text(self._cpu(i, 'scaling_available_frequencies')) or '').split()
            per_core.append({int(x) for x in listed})
            lo, hi = (_as_int(read_text(self._cpu(i, n))) for n in ('cpuinfo_min_freq', 'cpuinfo_max_freq'))
            if lo is not None and hi is not None:
                bounds.append((lo, hi))
        cpu = sorted(set.intersection(*per_core)) if per_core and all(per_core) else []
        cpu_range = [max(b[0] for b in bounds), min(b[1] for b in bounds)] if bounds else None
        gpu = sorted(int(x) for x in (read_text(self.gpu_path / 'available_frequencies') or '').split())
        ina = self.ina_path
        return dict(cpu_khz_available=cpu, gpu_hz_available=gpu, cpu_khz_range=cpu_range,
                    cpu_khz_source=('scaling_available_frequencies (intersection over cores)' if cpu
                                    else 'cpuinfo_min/max_freq (no frequency list exposed)'),
                    gpu_hz_source='devfreq available_frequencies',
                    power_mode=read_power_mode(),
                    emc_hz=_as_int(read_text(self.emc_cap_path)) if self.emc_cap_path else None,
                    power_sensor=dict(path=ina, channel='in1/curr1',
                                      label=read_text(os.path.join(ina, 'in1_label')) if ina else None,
                                      update_interval_ms=_as_int(read_text(os.path.join(ina, 'update_interval')))
                                      if ina else None))

    def resolve(self, cpu_khz, gpu_hz, caps: dict[str, Any] | None = None) -> tuple[int, int]:
        """Map 'min'/'max' to the device's available frequencies and reject unlisted values."""
        caps = caps or self.capabilities()
        out = []
        for value, listed, rng, unit in ((cpu_khz, caps['cpu_khz_available'], caps.get('cpu_khz_range'), 'CPU kHz'),
                                         (gpu_hz, caps['gpu_hz_available'], None, 'GPU Hz')):
            bounds = [listed[0], listed[-1]] if listed else rng
            if isinstance(value, str):
                if value not in ('min', 'max'):
                    raise ValueError(f'Unknown symbolic frequency {value!r}')
                if not bounds:
                    raise ValueError(f'Device exposes no available {unit} values; cannot resolve {value!r}')
                value = bounds[0] if value == 'min' else bounds[1]
            value = int(value)
            if listed and value not in listed:
                raise ValueError(f'{unit} {value} is not supported by this device; available: {listed}')
            if not listed and bounds and not bounds[0] <= value <= bounds[1]:
                raise ValueError(f'{unit} {value} is outside the device range {bounds}')
            out.append(value)
        return out[0], out[1]

    def apply(self, cpu_khz, gpu_hz, settle_s: float = 1.0) -> dict[str, Any]:
        requested = (cpu_khz, gpu_hz)
        cpu_khz, gpu_hz = self.resolve(cpu_khz, gpu_hz)
        for i in range(self.n_cpus):
            hw_max = read_text(self._cpu(i, 'cpuinfo_max_freq'))
            hw_min = read_text(self._cpu(i, 'cpuinfo_min_freq'))
            # Open the window first so neither bound is rejected, then close it on the target.
            if hw_max:
                write_text(self._cpu(i, 'scaling_max_freq'), hw_max)
            if hw_min:
                write_text(self._cpu(i, 'scaling_min_freq'), hw_min)
            write_text(self._cpu(i, 'scaling_governor'), 'userspace')
            write_text(self._cpu(i, 'scaling_max_freq'), cpu_khz)
            write_text(self._cpu(i, 'scaling_min_freq'), cpu_khz)
            write_text(self._cpu(i, 'scaling_setspeed'), cpu_khz)
        available = (read_text(self.gpu_path / 'available_frequencies') or '').split()
        if available:
            write_text(self.gpu_path / 'max_freq', max(int(x) for x in available))
            write_text(self.gpu_path / 'min_freq', min(int(x) for x in available))
        write_text(self.gpu_path / 'governor', 'userspace')
        write_text(self.gpu_path / 'max_freq', gpu_hz)
        write_text(self.gpu_path / 'min_freq', gpu_hz)
        time.sleep(max(settle_s, 0.0))
        status = self.status()
        cpu_ok = all(f is not None and abs(f - cpu_khz) <= max(20_000, 0.02 * cpu_khz)
                     for f in status['cpu_khz'])
        gpu = status['gpu_hz']
        gpu_ok = gpu is not None and abs(gpu - gpu_hz) <= max(5_000_000, 0.02 * gpu_hz)
        status.update(requested_cpu_khz=cpu_khz, requested_gpu_hz=gpu_hz,
                      requested_symbolic=[str(v) for v in requested],
                      frequency_valid=bool(cpu_ok and gpu_ok))
        self.current = status
        return status

    def apply_governor(self, cpu_governor: str = DEFAULT_GOVERNORS['cpu'],
                       gpu_governor: str = DEFAULT_GOVERNORS['gpu'], settle_s: float = 1.0) -> dict[str, Any]:
        """Hand control back to the given governors over the full hardware frequency range."""
        for i in range(self.n_cpus):
            hw_max = read_text(self._cpu(i, 'cpuinfo_max_freq'))
            hw_min = read_text(self._cpu(i, 'cpuinfo_min_freq'))
            if hw_max:
                write_text(self._cpu(i, 'scaling_max_freq'), hw_max)
            if hw_min:
                write_text(self._cpu(i, 'scaling_min_freq'), hw_min)
            write_text(self._cpu(i, 'scaling_governor'), cpu_governor)
        available = (read_text(self.gpu_path / 'available_frequencies') or '').split()
        if available:
            write_text(self.gpu_path / 'max_freq', max(int(x) for x in available))
            write_text(self.gpu_path / 'min_freq', min(int(x) for x in available))
        write_text(self.gpu_path / 'governor', gpu_governor)
        time.sleep(max(settle_s, 0.0))
        status = self.status()
        ok = (all(g == cpu_governor for g in status['cpu_governors'])
              and status['gpu_governor'] == gpu_governor)
        status.update(requested_cpu_khz=None, requested_gpu_hz=None,
                      requested_cpu_governor=cpu_governor, requested_gpu_governor=gpu_governor,
                      frequency_valid=bool(ok))
        self.current = status
        return status

    def read_freqs(self) -> tuple[float | None, float | None]:
        """(mean current CPU frequency over cores in kHz, current GPU frequency in Hz)."""
        cpus = []
        for i in range(self.n_cpus):
            try:
                cpus.append(int(read_text(self._cpu(i, 'scaling_cur_freq'))))
            except (TypeError, ValueError):
                pass
        try:
            gpu = int(read_text(self.gpu_path / 'cur_freq'))
        except (TypeError, ValueError):
            gpu = None
        return (sum(cpus) / len(cpus) if cpus else None), gpu

    def status(self) -> dict[str, Any]:
        as_int = _as_int
        return dict(power_mode=read_power_mode(),
                    cpu_khz=[as_int(read_text(self._cpu(i, 'scaling_cur_freq'))) for i in range(self.n_cpus)],
                    gpu_hz=as_int(read_text(self.gpu_path / 'cur_freq')),
                    emc_hz=as_int(read_text(self.emc_cap_path)) if self.emc_cap_path else None,
                    cpu_governors=[read_text(self._cpu(i, 'scaling_governor')) for i in range(self.n_cpus)],
                    gpu_governor=read_text(self.gpu_path / 'governor'),
                    temps_c=read_temperatures())

    def restore(self):
        if self.snapshot is None:
            return
        errors = []
        for i, orig in enumerate(self.snapshot['cpus']):
            try:
                hw_max = read_text(self._cpu(i, 'cpuinfo_max_freq'))
                if hw_max:
                    write_text(self._cpu(i, 'scaling_max_freq'), hw_max)
                write_text(self._cpu(i, 'scaling_min_freq'), orig['scaling_min_freq'])
                write_text(self._cpu(i, 'scaling_max_freq'), orig['scaling_max_freq'])
                write_text(self._cpu(i, 'scaling_governor'), orig['scaling_governor'])
            except OSError as exc:
                errors.append(f'cpu{i}: {exc}')
        try:
            g = self.snapshot['gpu']
            available = (read_text(self.gpu_path / 'available_frequencies') or '').split()
            if available:
                write_text(self.gpu_path / 'max_freq', max(int(x) for x in available))
            write_text(self.gpu_path / 'min_freq', g['min_freq'])
            write_text(self.gpu_path / 'max_freq', g['max_freq'])
            write_text(self.gpu_path / 'governor', g['governor'])
        except OSError as exc:
            errors.append(f'gpu: {exc}')
        if errors:
            print('[WARN] DVFS restore incomplete: ' + '; '.join(errors), flush=True)
        else:
            print('[OK] Original DVFS settings restored.', flush=True)
