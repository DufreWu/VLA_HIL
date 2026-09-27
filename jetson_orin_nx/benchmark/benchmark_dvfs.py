#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import sys
import time

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "outputs" / "nx_model"))
    parser.add_argument("--output", default=str(ROOT / "jetson_orin_nx" / "benchmark" / "results" / "smolvla_dvfs.csv"))
    parser.add_argument("--collect-energy", action="store_true", help="Collect GPU energy/power using NVML if available")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=TEST_RUNS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor = load_policy(args.model, device)
    observation = create_dummy_observation(device)

    # Try to initialize NVML for energy measurements if requested
    nvml = None
    collect_energy = args.collect_energy
    if collect_energy:
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml = pynvml
        except Exception:
            print("Warning: pynvml not available or NVML init failed; continuing without energy collection")
            nvml = None

    edge_board = Jetson_Orin_NX(config)
    results = []
    for cpu_freq in CPU_FREQS:
        for gpu_freq in GPU_FREQS:
            for cpu_idx in range(CPU_CORE_NUM):
                edge_board.set_cpu_freq(cpu_idx, cpu_freq)
            edge_board.set_gpu_freq(gpu_freq)

            start = time.perf_counter()
            for _ in range(args.warmup):
                inference(policy, observation, preprocessor)
                if device == "cuda":
                    torch.cuda.synchronize()
            latencies = []
            energies = []
            for _ in range(args.runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                # sample power just before and after inference when possible
                p0 = None
                p1 = None
                if nvml is not None:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        util = nvml.nvmlDeviceGetPowerUsage(handle)
                        p0 = float(util) / 1000.0
                    except Exception:
                        p0 = None

                inference(policy, observation, preprocessor)
                torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt_ms)

                if nvml is not None:
                    try:
                        # re-get handle in case of transient errors
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        util = nvml.nvmlDeviceGetPowerUsage(handle)
                        p1 = float(util) / 1000.0
                    except Exception:
                        p1 = None

                # Estimate energy in joules: average power (W) * seconds
                if p0 is not None and p1 is not None:
                    avg_p = 0.5 * (p0 + p1)
                    energy_j = avg_p * (dt_ms / 1000.0)
                    energies.append(energy_j)
                else:
                    energies.append(None)
            mean_latency = float(np.mean(latencies))
            # compute mean energy if collected
            mean_energy = float(np.mean([e for e in energies if e is not None])) if any(e is not None for e in energies) else None

            results.append({
                "cpu_mhz": edge_board.read_cpu_freq(0) / 1000.0,
                "gpu_mhz": edge_board.read_gpu_freq() / 1e6,
                "latency_mean_ms": mean_latency,
                "energy_mean_j": mean_energy,
                "elapsed_s": time.perf_counter() - start,
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        # ensure energy column is present even if None
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # shutdown NVML if initialized
    if nvml is not None:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
