#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from common.contract import TASK
from jetson_orin_nx import Jetson_Orin_NX

CONFIG_PATH = ROOT / "jetson_orin_nx" / "benchmark" / "jetson_orin_nx.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

CPU_FREQS = config["jetson_orin_nx_16g"]["cpu_freqs"]
CPU_CORE_NUM = config["jetson_orin_nx_16g"]["cpu_cores_num"]
GPU_FREQS = config["jetson_orin_nx_16g"]["gpu_freqs"]
WARMUP_RUNS = 20
TEST_RUNS = 100


def read_text(path: str | os.PathLike[str]) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return None


def read_float(path: str | os.PathLike[str]) -> float | None:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_nvpmodel_status() -> str | None:
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip() or None
    except Exception:
        return None


def read_vdd_in_power(ina_path: str | None) -> float | None:
    if not ina_path:
        return None
    voltage_v = read_float(os.path.join(ina_path, "in1_input"))
    current_ma = read_float(os.path.join(ina_path, "curr1_input"))
    if voltage_v is None or current_ma is None:
        return None
    # INA3221 exposes voltage in mV and current in mA; convert to watts.
    return (voltage_v / 1000.0) * (current_ma / 1000.0)


def read_thermal_sensors() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    thermal_root = Path("/sys/class/thermal")
    if not thermal_root.exists():
        return entries
    for zone in sorted(thermal_root.glob("thermal_zone*")):
        zone_type = read_text(zone / "type")
        temp = read_float(zone / "temp")
        if zone_type is None or temp is None:
            continue
        entries.append({
            "name": zone.name,
            "type": zone_type,
            "temp_milli_c": temp,
            "temp_c": temp / 1000.0,
        })
    return entries


def load_policy(model_path, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(model_path)
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config)
    policy.to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor


@torch.inference_mode()
def inference(policy, observation, preprocessor=None):
    policy.reset()
    batch = observation if preprocessor is None else preprocessor(observation)
    return policy.select_action(batch)


def create_dummy_observation(device):
    return {
        "observation.state": torch.zeros(1, 9, dtype=torch.float32, device=device),
        "observation.images.camera1": torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device),
        "observation.images.camera2": torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device),
        "observation.images.camera3": torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device),
        "task": TASK,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_config_json(model_path: Path, device: str, collect_energy: bool, warmup: int, runs: int, output_dir: Path) -> dict[str, Any]:
    cfg = {
        "model": {
            "path": str(model_path),
            "name": model_path.name,
            "exists": model_path.exists(),
            "device": device,
            "task": TASK,
        },
        "input": {
            "state_dim": 9,
            "image_size": [256, 256],
            "camera_keys": ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"],
            "task_string": TASK,
        },
        "nvpmodel": {
            "status": read_nvpmodel_status(),
            "power_sensor_path": config["jetson_orin_nx_16g"].get("ina_path"),
        },
        "experiment": {
            "output_dir": str(output_dir),
            "collect_energy": collect_energy,
            "warmup_runs": warmup,
            "test_runs": runs,
            "cpu_freqs": list(CPU_FREQS),
            "gpu_freqs": list(GPU_FREQS),
            "cpu_cores_num": CPU_CORE_NUM,
            "device": device,
            "physics_hz": 60,
            "inference_hz": None,
        },
        "thermal": read_thermal_sensors(),
    }
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "outputs" / "nx_model"))
    parser.add_argument("--output", default=str(ROOT / "jetson_orin_nx" / "benchmark" / "results"), help="Directory for config.json, repetitions.csv, inferences.csv and power_samples.csv")
    parser.add_argument("--collect-energy", action="store_true", help="Collect battery power samples and energy estimates from the INA sensor when available")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=TEST_RUNS)
    parser.add_argument("--sampling-interval-s", type=float, default=0.005, help="Power-sample interval during each inference window")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor = load_policy(str(model_path), device)
    observation = create_dummy_observation(device)

    nvml = None
    if args.collect_energy:
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml = pynvml
        except Exception:
            print("Warning: pynvml not available or NVML init failed; continuing without NVML energy collection")
            nvml = None

    edge_board = Jetson_Orin_NX(config)
    ina_path = config["jetson_orin_nx_16g"].get("ina_path")
    config_json = build_config_json(model_path, device, args.collect_energy, args.warmup, args.runs, output_dir)
    (output_dir / "config.json").write_text(json.dumps(config_json, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    repetition_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    repeated_idx = 0

    for cpu_freq in CPU_FREQS:
        for gpu_freq in GPU_FREQS:
            for cpu_idx in range(CPU_CORE_NUM):
                edge_board.set_cpu_freq(cpu_idx, cpu_freq)
            edge_board.set_gpu_freq(gpu_freq)

            actual_cpu_mhz = float(edge_board.read_cpu_freq(0) / 1000.0)
            actual_gpu_mhz = float(edge_board.read_gpu_freq() / 1e6)

            warmup_start = time.perf_counter()
            for _ in range(args.warmup):
                inference(policy, observation, preprocessor)
                if device == "cuda":
                    torch.cuda.synchronize()
            warmup_elapsed = time.perf_counter() - warmup_start

            latencies = []
            powers = []
            energies = []
            for run_idx in range(args.runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                power_before = read_vdd_in_power(ina_path)
                t0 = time.perf_counter()
                infer_started = time.perf_counter()
                inference(policy, observation, preprocessor)
                if device == "cuda":
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - infer_started) * 1000.0
                latencies.append(infer_ms)

                power_after = read_vdd_in_power(ina_path)
                power_w = None
                if power_before is not None and power_after is not None:
                    power_w = 0.5 * (power_before + power_after)
                elif power_before is not None:
                    power_w = power_before
                elif power_after is not None:
                    power_w = power_after
                if power_w is not None:
                    powers.append(power_w)
                else:
                    powers.append(None)

                energy_j = None
                if power_w is not None:
                    energy_j = power_w * (infer_ms / 1000.0)
                energies.append(energy_j)

                inference_rows.append({
                    "repetition_id": repeated_idx,
                    "run_idx": run_idx,
                    "cpu_mhz_requested": float(cpu_freq / 1000.0),
                    "cpu_mhz_actual": actual_cpu_mhz,
                    "gpu_mhz_requested": float(gpu_freq / 1e6),
                    "gpu_mhz_actual": actual_gpu_mhz,
                    "latency_ms": float(infer_ms),
                    "power_w": power_w,
                    "energy_j": energy_j,
                    "wall_time_s": float(time.perf_counter() - t0),
                    "device": device,
                    "model_name": model_path.name,
                    "collect_energy": bool(args.collect_energy),
                })

                if args.collect_energy and ina_path is not None:
                    sample_start = time.perf_counter()
                    sample_window = max(0.002, min(0.25, (infer_ms / 1000.0) * 2.0))
                    sample_interval = max(args.sampling_interval_s, 0.001)
                    while time.perf_counter() - sample_start < sample_window:
                        power_sample = read_vdd_in_power(ina_path)
                        if power_sample is not None:
                            power_rows.append({
                                "repetition_id": repeated_idx,
                                "run_idx": run_idx,
                                "timestamp_monotonic_s": time.perf_counter(),
                                "power_w": float(power_sample),
                                "source": "VDD_IN",
                                "device": device,
                            })
                        time.sleep(sample_interval)

            mean_latency = float(np.mean(latencies)) if latencies else None
            p95_latency = float(np.percentile(latencies, 95)) if latencies else None
            std_latency = float(np.std(latencies)) if latencies else None
            mean_power = float(np.mean([p for p in powers if p is not None])) if any(p is not None for p in powers) else None
            mean_energy = float(np.mean([e for e in energies if e is not None])) if any(e is not None for e in energies) else None
            total_energy = float(np.sum([e for e in energies if e is not None])) if any(e is not None for e in energies) else None

            repetition_rows.append({
                "repetition_id": repeated_idx,
                "requested_cpu_mhz": float(cpu_freq / 1000.0),
                "actual_cpu_mhz": actual_cpu_mhz,
                "requested_gpu_mhz": float(gpu_freq / 1e6),
                "actual_gpu_mhz": actual_gpu_mhz,
                "model_name": model_path.name,
                "mean_inference_ms": mean_latency,
                "p95_inference_ms": p95_latency,
                "std_inference_ms": std_latency,
                "mean_power_w": mean_power,
                "mean_energy_j": mean_energy,
                "total_energy_j": total_energy,
                "energy_per_inference_j": mean_energy,
                "warmup_elapsed_s": float(warmup_elapsed),
                "device": device,
                "power_mode": read_nvpmodel_status(),
            })
            repeated_idx += 1

    repetitions_path = output_dir / "repetitions.csv"
    repetition_fieldnames = [
        "repetition_id", "requested_cpu_mhz", "actual_cpu_mhz", "requested_gpu_mhz", "actual_gpu_mhz",
        "model_name", "mean_inference_ms", "p95_inference_ms", "std_inference_ms",
        "mean_power_w", "mean_energy_j", "total_energy_j", "energy_per_inference_j",
        "warmup_elapsed_s", "device", "power_mode"
    ]
    write_csv(repetitions_path, repetition_fieldnames, repetition_rows)

    inference_fieldnames = [
        "repetition_id", "run_idx", "cpu_mhz_requested", "cpu_mhz_actual", "gpu_mhz_requested",
        "gpu_mhz_actual", "latency_ms", "power_w", "energy_j", "wall_time_s", "device", "model_name",
        "collect_energy"
    ]
    write_csv(output_dir / "inferences.csv", inference_fieldnames, inference_rows)

    power_fieldnames = ["repetition_id", "run_idx", "timestamp_monotonic_s", "power_w", "source", "device"]
    write_csv(output_dir / "power_samples.csv", power_fieldnames, power_rows)

    legacy_summary_path = output_dir / "smolvla_dvfs.csv"
    legacy_rows = [{
        "repetition_id": row["repetition_id"],
        "requested_cpu_mhz": row["requested_cpu_mhz"],
        "actual_cpu_mhz": row["actual_cpu_mhz"],
        "requested_gpu_mhz": row["requested_gpu_mhz"],
        "actual_gpu_mhz": row["actual_gpu_mhz"],
        "mean_inference_ms": row["mean_inference_ms"],
        "mean_power_w": row["mean_power_w"],
        "mean_energy_j": row["mean_energy_j"],
        "energy_per_inference_j": row["energy_per_inference_j"],
        "power_mode": row["power_mode"],
    } for row in repetition_rows]
    write_csv(legacy_summary_path, list(legacy_rows[0].keys()) if legacy_rows else [], legacy_rows)

    if nvml is not None:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

    print(f"Saved config to {output_dir / 'config.json'}")
    print(f"Saved repetition summary to {repetitions_path}")
    print(f"Saved per-inference samples to {output_dir / 'inferences.csv'}")
    print(f"Saved power samples to {output_dir / 'power_samples.csv'}")


if __name__ == "__main__":
    main()
