#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "jetson_orin_nx.yaml"

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


def load_policy(model_path, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy.to(device)
    policy.eval()
    return policy


@torch.inference_mode()
def inference(policy, observation):
    return policy.select_action(observation)


def create_dummy_observation(device):
    return {
        "observation.state": torch.zeros(1, 9, dtype=torch.float32, device=device),
        "observation.images.camera1": torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device),
        "observation.images.camera2": torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device),
        "observation.images.camera3": torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "outputs" / "nx_model"))
    parser.add_argument("--output", default=str(ROOT / "benchmark" / "results" / "smolvla_dvfs.csv"))
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=TEST_RUNS)
    args = parser.parse_args()

    device = "cuda"
    policy = load_policy(args.model, device)
    observation = create_dummy_observation(device)

    results = []
    for cpu_freq in CPU_FREQS:
        for gpu_freq in GPU_FREQS:
            start = time.perf_counter()
            for _ in range(args.warmup):
                inference(policy, observation)
                torch.cuda.synchronize()
            latencies = []
            for _ in range(args.runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                inference(policy, observation)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)
            mean_latency = float(np.mean(latencies))
            results.append({
                "cpu_mhz": cpu_freq / 1000.0,
                "gpu_mhz": gpu_freq / 1e6,
                "latency_mean_ms": mean_latency,
                "elapsed_s": time.perf_counter() - start,
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
