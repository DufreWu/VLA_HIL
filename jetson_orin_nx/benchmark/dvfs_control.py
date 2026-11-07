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


def load_dvfs_configs(path=CLOSED_LOOP_CONFIG, names=None) -> list[dict[str, Any]]:
    """Return validated configs in file order.

    Fixed configs: ``{name, cpu_khz, gpu_hz}``. Governor configs: ``{name, governor: default}``
    (stock governors) or ``{name, cpu_governor, gpu_governor}``; their cpu_khz/gpu_hz are None.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    configs = data.get('configs')
    if not isinstance(configs, list) or not configs:
        raise ValueError(f'{path}: expected a non-empty "configs" list')
    out, seen = [], set()
    for item in configs:
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
        cpu, gpu = int(item['cpu_khz']), int(item['gpu_hz'])
        if cpu <= 0 or gpu <= 0:
            raise ValueError(f'{name}: frequencies must be positive')
        out.append(dict(item, name=name, cpu_khz=cpu, gpu_hz=gpu))
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
    if end <= start:
        return None, 0
    pts = sorted(samples)
    if len(pts) < 2 or pts[0][0] > start or pts[-1][0] < end:
        return None, sum(start <= t <= end for t, _ in pts)

    def interp(t):
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                return p0 if t1 == t0 else p0 + (p1 - p0) * (t - t0) / (t1 - t0)
        raise AssertionError('unreachable')

    inside = [(t, p) for t, p in pts if start < t < end]
    seq = [(start, interp(start)), *inside, (end, interp(end))]
    energy = sum(0.5 * (p0 + p1) * (t1 - t0) for (t0, p0), (t1, p1) in zip(seq, seq[1:]))
    return float(energy), len(inside)


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
        self.episode = dict(tag=str(tag), start=self.clock(), inferences=[],
                            temps_start=read_temperatures())
        return dict(tag=self.episode['tag'])

    def record_inference(self, start: float, end: float):
        if self.episode is not None:
            self.episode['inferences'].append((float(start), float(end)))

    def end_episode(self, tag: str, settle_s: float | None = None) -> dict[str, Any]:
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
        energy, n = integrate(samples, ep['start'], end)

        def mean_of(idx):
            vals = [f[idx] for f in freqs if f[idx] is not None]
            return sum(vals) / len(vals) if vals else None
        wall = end - ep['start']
        inf_energy, inf_time = 0.0, 0.0
        for s, e in ep['inferences']:
            e_j, _ = integrate(samples, s, e)
            if e_j is None:
                inf_energy = None
                break
            inf_energy += e_j
            inf_time += e - s
        return dict(tag=ep['tag'], wall_s=wall, energy_j=energy,
                    mean_power_w=None if energy is None else energy / wall,
                    n_power_samples=n,
                    inference_energy_j=inf_energy,
                    inference_busy_s=inf_time,
                    inference_duty_cycle=inf_time / wall if wall > 0 else None,
                    n_inferences=len(ep['inferences']),
                    mean_cpu_khz=mean_of(1), mean_gpu_hz=mean_of(2), n_freq_samples=len(freqs),
                    temps_start_c=ep['temps_start'], temps_end_c=read_temperatures())


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

    def apply(self, cpu_khz: int, gpu_hz: int, settle_s: float = 1.0) -> dict[str, Any]:
        cpu_khz, gpu_hz = int(cpu_khz), int(gpu_hz)
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
        def as_int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        return dict(cpu_khz=[as_int(read_text(self._cpu(i, 'scaling_cur_freq'))) for i in range(self.n_cpus)],
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
