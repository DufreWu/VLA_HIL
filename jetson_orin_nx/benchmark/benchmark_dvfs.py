#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from common.contract import IMAGE_KEY, RESOLUTION, TASK, contract, encode_image
from jetson_orin_nx.benchmark.jetson_orin_nx import Jetson_Orin_NX
from jetson_orin_nx.policy_server import Engine

CONFIG_PATH = ROOT / "jetson_orin_nx" / "benchmark" / "jetson_orin_nx.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

CPU_FREQS = config["jetson_orin_nx_16g"]["cpu_freqs"]
CPU_CORE_NUM = config["jetson_orin_nx_16g"]["cpu_cores_num"]
GPU_FREQS = config["jetson_orin_nx_16g"]["gpu_freqs"]
DEFAULT_WARMUP = 20
DEFAULT_SAMPLES = 20
DEFAULT_REPETITIONS = 3
TASKS = {
    "grasping": "Grasp the red cube and lift it above the table.",
    "pushing": "Push the red cube to the green target on the table.",
}


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


def integrate_power_samples(samples: list[dict[str, Any]]) -> float | None:
    if not samples:
        return None
    valid = sorted((s for s in samples if s.get("power_w") is not None), key=lambda x: float(x["timestamp_monotonic_s"]))
    if len(valid) < 2:
        return None
    energy = 0.0
    for prev, curr in zip(valid, valid[1:]):
        dt = float(curr["timestamp_monotonic_s"]) - float(prev["timestamp_monotonic_s"])
        if dt <= 0:
            continue
        energy += 0.5 * (float(prev["power_w"]) + float(curr["power_w"])) * dt
    return float(energy)


def make_task_cycle(task_names: list[str]) -> list[str]:
    if not task_names:
        return ["grasping"]
    return list(task_names)


def build_observation(device: str, task_name: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    rgb = np.fromiter((rng.randrange(256) for _ in range(256 * 256 * 3)), dtype=np.uint8).reshape(256, 256, 3)
    state = np.zeros(9, dtype=np.float32)
    return {
        "observation.state": torch.from_numpy(state.copy()).to(device),
        IMAGE_KEY: torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float().to(device) / 255.0,
        "task": TASKS[task_name],
    }


def validate_frequency_setting(board: Jetson_Orin_NX, cpu_hz: int, gpu_hz: int, cpu_idx: int = 0) -> bool:
    actual_cpu = board.read_cpu_freq(cpu_idx)
    actual_gpu = board.read_gpu_freq()
    if actual_cpu is None or actual_gpu is None:
        return False
    if abs(actual_cpu - cpu_hz) > max(20000, int(cpu_hz * 0.02)):
        return False
    if abs(actual_gpu - gpu_hz) > max(5000000, int(gpu_hz * 0.02)):
        return False
    return True


def collect_power_samples(duration_s: float, sample_interval_s: float, ina_path: str | None, sample_id: str | int) -> list[dict[str, Any]]:
    if ina_path is None:
        return []
    samples: list[dict[str, Any]] = []
    start = time.monotonic()
    deadline = start + max(duration_s, 0.0)
    while time.monotonic() < deadline:
        ts = time.monotonic()
        power_w = read_vdd_in_power(ina_path)
        if power_w is not None:
            samples.append({
                "sample_id": str(sample_id),
                "timestamp_monotonic_s": ts,
                "power_w": power_w,
                "source": "VDD_IN",
            })
        time.sleep(max(sample_interval_s, 0.001))
    return samples


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_config_json(model_path: Path, output_dir: Path, args: argparse.Namespace, device: str, thermal: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": {
            "path": str(model_path),
            "name": model_path.name,
            "exists": model_path.exists(),
            "device": device,
            "task": TASK,
            "contract": contract(),
        },
        "input": {
            "state_dim": 9,
            "image_size": list(RESOLUTION),
            "camera_keys": [IMAGE_KEY],
            "task_instructions": TASKS,
            "task_string": TASK,
        },
        "nvpmodel": {
            "status": read_nvpmodel_status(),
            "power_sensor_path": config["jetson_orin_nx_16g"].get("ina_path"),
        },
        "experiment": {
            "output_dir": str(output_dir),
            "collect_energy": bool(args.collect_energy),
            "warmup_runs": int(args.warmup),
            "samples_per_config": int(args.samples_per_config),
            "repetitions": int(args.repetitions),
            "seed": int(args.seed),
            "cpu_freqs": [int(v) for v in CPU_FREQS],
            "gpu_freqs": [int(v) for v in GPU_FREQS],
            "cpu_cores_num": int(CPU_CORE_NUM),
            "device": device,
            "task_selection": args.task,
            "timeout_s": float(args.timeout_s),
            "sampling_interval_s": float(args.sampling_interval_s),
        },
        "thermal": thermal,
    }


def summarize_repetitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["config_id"],
            row["requested_cpu_mhz"],
            row["requested_gpu_mhz"],
            row["task_name"],
        )
        grouped.setdefault(key, []).append(row)
    for key, items in grouped.items():
        inference_ms = np.asarray([float(x["inference_ms"]) for x in items if x["inference_ms"] is not None], dtype=np.float64)
        power_w = np.asarray([float(x["power_w"]) for x in items if x["power_w"] is not None], dtype=np.float64)
        energy_j = np.asarray([float(x["energy_j"]) for x in items if x["energy_j"] is not None], dtype=np.float64)
        edp = energy_j * (inference_ms / 1000.0) if energy_j.size and inference_ms.size else np.asarray([], dtype=np.float64)
        out.append({
            "config_id": key[0],
            "requested_cpu_mhz": key[1],
            "requested_gpu_mhz": key[2],
            "task_name": key[3],
            "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms.size else None,
            "p95_inference_ms": float(np.percentile(inference_ms, 95)) if inference_ms.size else None,
            "mean_power_w": float(np.mean(power_w)) if power_w.size else None,
            "mean_energy_j": float(np.mean(energy_j)) if energy_j.size else None,
            "energy_delay_product_j_s": float(np.mean(edp)) if edp.size else None,
            "n_samples": int(len(items)),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Jetson Orin NX DVFS benchmark for the VLA-HIL SmolVLA model.")
    parser.add_argument("--model", default=str(ROOT / "outputs" / "nx_multitask_v2"), help="Path to the deployed model export bundle.")
    parser.add_argument("--output", default=str(ROOT / "jetson_orin_nx" / "benchmark" / "results"), help="Directory for config.json, inferences.csv, power_samples.csv, repetitions.csv and summary.csv.")
    parser.add_argument("--task", choices=["grasping", "pushing", "both"], default="both", help="Task instructions to benchmark for the multitask model.")
    parser.add_argument("--samples-per-config", type=int, default=DEFAULT_SAMPLES, help="Inference samples recorded per configuration and repetition.")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help="Independent repetitions per configuration.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations before timing starts.")
    parser.add_argument("--collect-energy", action="store_true", help="Sample VDD_IN power during each inference and integrate energy over time.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for configuration order and input generation.")
    parser.add_argument("--sampling-interval-s", type=float, default=0.002, help="Power sampling interval in seconds.")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Safety timeout for a full DVFS benchmark run.")
    args = parser.parse_args()

    if args.samples_per_config < 1 or args.repetitions < 1:
        raise ValueError("--samples-per-config and --repetitions must be positive integers.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Engine(str(model_path), device)

    thermal = read_thermal_sensors()
    config_json = build_config_json(model_path, output_dir, args, device, thermal)
    (output_dir / "config.json").write_text(json.dumps(config_json, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    task_names = ["grasping", "pushing"] if args.task == "both" else [args.task]
    task_cycle = make_task_cycle(task_names)

    board = Jetson_Orin_NX(config)
    original_cpu_freqs = [board.read_cpu_freq(i) for i in range(CPU_CORE_NUM)]
    original_gpu_freq = board.read_gpu_freq()
    rng = random.Random(args.seed)
    configs = [(cpu_hz, gpu_hz) for cpu_hz in CPU_FREQS for gpu_hz in GPU_FREQS]
    rng.shuffle(configs)

    inference_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    repetition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    try:
        for config_id, (cpu_hz, gpu_hz) in enumerate(configs):
            for cpu_idx in range(CPU_CORE_NUM):
                board.set_cpu_freq(cpu_idx, cpu_hz)
            board.set_gpu_freq(gpu_hz)
            actual_cpu = board.read_cpu_freq(0)
            actual_gpu = board.read_gpu_freq()
            frequency_valid = validate_frequency_setting(board, cpu_hz, gpu_hz)
            if not frequency_valid:
                print(f"Warning: frequency request mismatch for config {config_id}: requested {(cpu_hz, gpu_hz)}, actual {(actual_cpu, actual_gpu)}")

            for repetition in range(args.repetitions):
                for sample_idx in range(args.samples_per_config):
                    task_name = task_cycle[(sample_idx + repetition) % len(task_cycle)]
                    observation = build_observation(device, task_name, seed=args.seed + config_id * 101 + repetition * 17 + sample_idx)
                    request = {
                        "contract": contract(),
                        "episode_id": f"dvfs_{config_id}_{repetition}_{sample_idx}",
                        "obs_id": sample_idx,
                        "state": observation["observation.state"].cpu().numpy().astype(np.float32).reshape(-1).tolist(),
                        "rgb_png": encode_image(observation[IMAGE_KEY].permute(1, 2, 0).cpu().numpy().astype(np.uint8)),
                        "task": observation["task"],
                    }

                    if args.collect_energy:
                        stop_event = threading.Event()
                        power_samples: list[dict[str, Any]] = []
                        def sampler():
                            while not stop_event.is_set():
                                ts = time.monotonic()
                                value = read_vdd_in_power(config["jetson_orin_nx_16g"].get("ina_path"))
                                if value is not None:
                                    power_samples.append({
                                        "config_id": config_id,
                                        "repetition_id": repetition,
                                        "sample_id": sample_idx,
                                        "timestamp_monotonic_s": ts,
                                        "power_w": float(value),
                                        "source": "VDD_IN",
                                        "task_name": task_name,
                                    })
                                time.sleep(max(args.sampling_interval_s, 0.001))
                        thread = threading.Thread(target=sampler, daemon=True)
                        thread.start()
                    else:
                        power_samples = []

                    t0 = time.monotonic()
                    actions, infer_ms = engine.infer(request)
                    t1 = time.monotonic()
                    if args.collect_energy:
                        stop_event.set()
                        thread.join(timeout=5.0)
                    energy_j = integrate_power_samples(power_samples) if args.collect_energy else None
                    if args.collect_energy and energy_j is None:
                        print(f"Warning: invalid power/integration for config {config_id}, repetition {repetition}, sample {sample_idx}")

                    row = {
                        "config_id": config_id,
                        "repetition_id": repetition,
                        "sample_id": sample_idx,
                        "requested_cpu_mhz": float(cpu_hz / 1000.0),
                        "actual_cpu_mhz": float(actual_cpu / 1000.0),
                        "requested_gpu_mhz": float(gpu_hz / 1e6),
                        "actual_gpu_mhz": float(actual_gpu / 1e6),
                        "task_name": task_name,
                        "inference_ms": float(infer_ms),
                        "power_w": float(np.mean([p["power_w"] for p in power_samples])) if power_samples else None,
                        "energy_j": None if energy_j is None else float(energy_j),
                        "wall_duration_s": float(t1 - t0),
                        "start_monotonic_s": t0,
                        "end_monotonic_s": t1,
                        "frequency_valid": bool(frequency_valid),
                        "device": device,
                        "model_name": model_path.name,
                    }
                    inference_rows.append(row)
                    for sample in power_samples:
                        power_rows.append({
                            "config_id": config_id,
                            "repetition_id": repetition,
                            "sample_id": sample_idx,
                            "timestamp_monotonic_s": float(sample["timestamp_monotonic_s"]),
                            "power_w": float(sample["power_w"]),
                            "source": sample["source"],
                            "task_name": task_name,
                            "device": device,
                        })

                    repetition_rows.append({
                        "config_id": config_id,
                        "repetition_id": repetition,
                        "requested_cpu_mhz": float(cpu_hz / 1000.0),
                        "actual_cpu_mhz": float(actual_cpu / 1000.0),
                        "requested_gpu_mhz": float(gpu_hz / 1e6),
                        "actual_gpu_mhz": float(actual_gpu / 1e6),
                        "task_name": task_name,
                        "mean_inference_ms": float(infer_ms),
                        "p95_inference_ms": float(infer_ms),
                        "average_power_w": float(np.mean([p["power_w"] for p in power_samples])) if power_samples else None,
                        "energy_per_inference_j": None if energy_j is None else float(energy_j),
                        "edp_j_s": None if energy_j is None else float(energy_j * (infer_ms / 1000.0)),
                        "frequency_valid": bool(frequency_valid),
                        "thermal_c": float(np.mean([z["temp_c"] for z in thermal])) if thermal else None,
                        "device": device,
                        "model_name": model_path.name,
                    })

                    time.sleep(0.01)
    finally:
        for cpu_idx in range(CPU_CORE_NUM):
            board.set_cpu_freq(cpu_idx, original_cpu_freqs[cpu_idx] if original_cpu_freqs[cpu_idx] is not None else CPU_FREQS[0])
        board.set_gpu_freq(original_gpu_freq if original_gpu_freq is not None else GPU_FREQS[0])

    inference_fieldnames = [
        "config_id", "repetition_id", "sample_id", "requested_cpu_mhz", "actual_cpu_mhz",
        "requested_gpu_mhz", "actual_gpu_mhz", "task_name", "inference_ms", "power_w",
        "energy_j", "wall_duration_s", "start_monotonic_s", "end_monotonic_s",
        "frequency_valid", "device", "model_name"
    ]
    write_csv(output_dir / "inferences.csv", inference_fieldnames, inference_rows)

    power_fieldnames = [
        "config_id", "repetition_id", "sample_id", "timestamp_monotonic_s", "power_w", "source",
        "task_name", "device"
    ]
    write_csv(output_dir / "power_samples.csv", power_fieldnames, power_rows)

    repetition_fieldnames = [
        "config_id", "repetition_id", "requested_cpu_mhz", "actual_cpu_mhz", "requested_gpu_mhz",
        "actual_gpu_mhz", "task_name", "mean_inference_ms", "p95_inference_ms", "average_power_w",
        "energy_per_inference_j", "edp_j_s", "frequency_valid", "thermal_c", "device", "model_name"
    ]
    write_csv(output_dir / "repetitions.csv", repetition_fieldnames, repetition_rows)

    summary = summarize_repetitions(inference_rows)
    summary_fieldnames = [
        "config_id", "requested_cpu_mhz", "requested_gpu_mhz", "task_name", "mean_inference_ms",
        "p95_inference_ms", "mean_power_w", "mean_energy_j", "energy_delay_product_j_s", "n_samples"
    ]
    write_csv(output_dir / "summary.csv", summary_fieldnames, summary)

    print(f"Saved benchmark configuration to {output_dir / 'config.json'}")
    print(f"Saved {len(inference_rows)} inference records to {output_dir / 'inferences.csv'}")
    print(f"Saved {len(power_rows)} power samples to {output_dir / 'power_samples.csv'}")
    print(f"Saved summary to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
