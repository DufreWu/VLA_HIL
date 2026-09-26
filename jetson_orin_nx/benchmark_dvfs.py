#!/usr/bin/env python3
import os
import re
import csv
import time
import json
import signal
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

import yaml


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path(__file__).with_name("jetson_orin_nx.yaml")
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

config.setdefault("cpu_freqs", [422400, 1190400, 1984000])
config.setdefault("gpu_freqs", [306000000, 612000000, 918000000])
config.setdefault("warmup_runs", 20)
config.setdefault("test_runs", 100)

CPU_FREQS = config["cpu_freqs"]
GPU_FREQS = config["gpu_freqs"]
WARMUP_RUNS = config["warmup_runs"]
TEST_RUNS = config["test_runs"]


# ============================================================
# DVFS control
# ============================================================

def run_cmd(cmd):
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def set_cpu_frequency(freq_khz):
    """
    Set all online CPU cores to a fixed frequency.

    freq_khz:
        422400  -> 422 MHz
        1190400 -> 1190 MHz
        1984000 -> 1984 MHz
    """

    cpu_root = Path("/sys/devices/system/cpu")

    for cpu_dir in sorted(cpu_root.glob("cpu[0-9]*")):
        cpufreq = cpu_dir / "cpufreq"

        if not cpufreq.exists():
            continue

        governor = cpufreq / "scaling_governor"
        min_freq = cpufreq / "scaling_min_freq"
        max_freq = cpufreq / "scaling_max_freq"

        try:
            if governor.exists():
                governor.write_text("userspace")

            if min_freq.exists():
                min_freq.write_text(str(freq_khz))

            if max_freq.exists():
                max_freq.write_text(str(freq_khz))

            setspeed = cpufreq / "scaling_setspeed"
            if setspeed.exists():
                setspeed.write_text(str(freq_khz))

        except PermissionError:
            raise RuntimeError(
                "Permission denied while setting CPU DVFS. "
                "Run benchmark with sudo."
            )


def find_gpu_devfreq():
    """
    Find NVIDIA GPU devfreq directory.
    """

    candidates = list(Path("/sys/class/devfreq").glob("*"))

    for p in candidates:
        name = p.name.lower()

        if "gpu" in name or "gv11b" in name or "17000000" in name:
            available = p / "available_frequencies"

            if available.exists():
                return p

    raise RuntimeError("Cannot find GPU devfreq path.")


def set_gpu_frequency(freq_hz):
    gpu = find_gpu_devfreq()

    governor = gpu / "governor"
    min_freq = gpu / "min_freq"
    max_freq = gpu / "max_freq"

    try:
        if governor.exists():
            # NVIDIA Jetson normally supports userspace governor
            try:
                governor.write_text("userspace")
            except Exception:
                pass

        min_freq.write_text(str(freq_hz))
        max_freq.write_text(str(freq_hz))

    except PermissionError:
        raise RuntimeError(
            "Permission denied while setting GPU DVFS. "
            "Run benchmark with sudo."
        )


# ============================================================
# Tegrastats
# ============================================================

POWER_PATTERN = re.compile(
    r"VDD_IN\s+(\d+)mW"
)


class TegrastatsMonitor:
    def __init__(self, interval_ms=20):
        self.interval_ms = interval_ms
        self.process = None
        self.samples = []

    def start(self):
        self.samples = []

        self.process = subprocess.Popen(
            [
                "tegrastats",
                "--interval",
                str(self.interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def collect_until(self, end_time):
        """
        Read tegrastats until end_time.
        """

        if self.process is None:
            return

        while time.perf_counter() < end_time:
            line = self.process.stdout.readline()

            if not line:
                continue

            timestamp = time.perf_counter()

            match = POWER_PATTERN.search(line)

            if match:
                power_mw = float(match.group(1))

                self.samples.append(
                    (timestamp, power_mw / 1000.0)
                )

    def stop(self):
        if self.process is not None:
            self.process.send_signal(signal.SIGINT)

            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()

            self.process = None


# ============================================================
# Energy calculation
# ============================================================

def calculate_energy(samples, start_time, end_time):
    """
    Integrate VDD_IN power samples during inference window.

    Returns:
        avg_power_w
        energy_j
    """

    selected = [
        (t, p)
        for t, p in samples
        if start_time <= t <= end_time
    ]

    if len(selected) < 2:
        return np.nan, np.nan

    t = np.array([x[0] for x in selected])
    p = np.array([x[1] for x in selected])

    avg_power = np.mean(p)

    # trapezoidal integration
    energy = np.trapz(p, t)

    return avg_power, energy


# ============================================================
# Model
# ============================================================

def load_policy(model_path, device):
    """
    IMPORTANT:
    Replace this function with the SAME model-loading code
    already used in your policy_server.py.

    Example structure only.
    """

    print(f"Loading model: {model_path}")

    # --------------------------------------------------------
    # COPY YOUR WORKING policy_server.py MODEL LOAD HERE
    # --------------------------------------------------------

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model_path)

    policy.to(device)
    policy.eval()

    return policy


def create_dummy_observation(device):
    """
    Current scene contract:
      state:   [9]
      camera1: [3, 256, 256]
      camera2: [3, 256, 256]
      camera3: [3, 256, 256]

    Batch dimension is included.
    """

    observation = {
        "observation.state":
            torch.zeros(
                1, 9,
                dtype=torch.float32,
                device=device,
            ),

        "observation.images.camera1":
            torch.rand(
                1, 3, 256, 256,
                dtype=torch.float32,
                device=device,
            ),

        "observation.images.camera2":
            torch.rand(
                1, 3, 256, 256,
                dtype=torch.float32,
                device=device,
            ),

        "observation.images.camera3":
            torch.rand(
                1, 3, 256, 256,
                dtype=torch.float32,
                device=device,
            ),
    }

    return observation


@torch.inference_mode()
def inference(policy, observation):
    """
    Replace only this line if policy_server.py uses
    another inference function.
    """

    action = policy.select_action(observation)

    return action


# ============================================================
# Benchmark
# ============================================================

def benchmark_configuration(
    policy,
    observation,
    cpu_freq,
    gpu_freq,
    warmup_runs,
    test_runs,
):

    print()
    print("=" * 70)
    print(
        f"CPU = {cpu_freq / 1000:.0f} MHz | "
        f"GPU = {gpu_freq / 1e6:.0f} MHz"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Set DVFS
    # --------------------------------------------------------

    set_cpu_frequency(cpu_freq)
    set_gpu_frequency(gpu_freq)

    # Let clocks stabilize
    time.sleep(2)

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(f"Warmup: {warmup_runs} runs")

    for _ in range(warmup_runs):

        inference(policy, observation)

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Start power monitor
    # --------------------------------------------------------

    monitor = TegrastatsMonitor(interval_ms=20)
    monitor.start()

    time.sleep(0.5)

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    latencies = []

    benchmark_start = time.perf_counter()

    for i in range(test_runs):

        torch.cuda.synchronize()

        start = time.perf_counter()

        inference(policy, observation)

        torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0

        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(
                f"\rInference {i + 1}/{test_runs}",
                end="",
                flush=True,
            )

    benchmark_end = time.perf_counter()

    print()

    # Give tegrastats one final sample
    time.sleep(0.1)

    monitor.stop()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    latencies = np.asarray(latencies)

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    avg_power, total_energy = calculate_energy(
        monitor.samples,
        benchmark_start,
        benchmark_end,
    )

    if np.isnan(total_energy):
        energy_per_inference = np.nan
    else:
        energy_per_inference = total_energy / test_runs

    result = {
        "cpu_mhz": cpu_freq / 1000,
        "gpu_mhz": gpu_freq / 1e6,

        "latency_mean_ms": mean_latency,
        "latency_std_ms": std_latency,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,

        "avg_power_w": avg_power,
        "total_energy_j": total_energy,
        "energy_per_inference_j": energy_per_inference,
    }

    print(
        f"Latency : {mean_latency:.2f} ± "
        f"{std_latency:.2f} ms"
    )

    print(
        f"P50/P95 : {p50:.2f} / "
        f"{p95:.2f} ms"
    )

    print(
        f"Power   : {avg_power:.2f} W"
    )

    print(
        f"Energy  : {energy_per_inference:.4f} J/inference"
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="outputs/nx_model",
    )

    parser.add_argument(
        "--output",
        default="results/dvfs_vla.csv",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_RUNS,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=TEST_RUNS,
    )

    args = parser.parse_args()

    device = "cuda"

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    policy = load_policy(
        args.model,
        device,
    )

    observation = create_dummy_observation(device)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # --------------------------------------------------------
    # Sweep
    # --------------------------------------------------------

    for cpu_freq in CPU_FREQS:

        for gpu_freq in GPU_FREQS:

            result = benchmark_configuration(
                policy,
                observation,
                cpu_freq,
                gpu_freq,
                args.warmup,
                args.runs,
            )

            results.append(result)

            # Save immediately in case experiment crashes later
            with open(
                output_path,
                "w",
                newline="",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=results[0].keys(),
                )

                writer.writeheader()
                writer.writerows(results)

    print()
    print("=" * 70)
    print("Benchmark finished")
    print(f"Results: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()