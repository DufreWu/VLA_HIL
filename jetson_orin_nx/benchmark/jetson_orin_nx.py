import yaml
import os
import sys
import threading
import time
import numpy as np

# ===============================
# Basic Settings
# ===============================

class Jetson_Orin_NX:
    def __init__(self, cfg):
        self.config = cfg

        board_info = self.config["jetson_orin_nx_16g"]

        self.cpu_path = board_info["cpu_path"]
        self.gpu_path = board_info["gpu_path"]
        self.mem_path = board_info["mem_path"]
        self.ina_path = board_info["ina_path"]

        self.cpu_freqs = board_info.get("cpu_freqs", [])
        self.gpu_freqs = board_info.get("gpu_freqs", [])
        self.mem_freqs = board_info.get("mem_freqs", [])
    
    def set_default_dvfs(self):
        # -------- CPU: schedutil --------
        for cpu in range(os.cpu_count()):
            path = f"{self.cpu_path}/cpu{cpu}/cpufreq/scaling_governor"
            try:
                with open(path, "w") as f:
                    f.write("schedutil")
            except:
                print(f"  [WARN] CPU restore error: {e}")

        # -------- GPU: nvhost_podgov --------
        try:
            with open(f"{self.gpu_path}/governor", "w") as f:
                f.write("nvhost_podgov")
        except:
            print(f"  [WARN] GPU restore error: {e}")

        # -------- Memory (EMC) Default Reset --------
        EMC_CAP_PATH = self.config["jetson_orin_nx_16g"]["emc_cap_path"]
        MAX_VALID_FREQ = self.config["jetson_orin_nx_16g"]["max_mem_freq"]

        try:
            if os.path.exists(EMC_CAP_PATH):
                with open(EMC_CAP_PATH, "w") as f:
                    f.write(MAX_VALID_FREQ)
                print(f"  [OK] MEM cap reset to {MAX_VALID_FREQ}")
        except Exception as e:
            print(f"  [WARN] MEM restore error: {e}")

        print("DVFS set to default (CPU:schedutil, GPU:nvhost_podgov, MEM:simple_ondemand)")

    def set_mem_default_dvfs(self):
        # -------- Memory (EMC) Default Reset --------
        EMC_CAP_PATH = self.config["jetson_orin_nx_16g"]["emc_cap_path"]
        MAX_VALID_FREQ = self.config["jetson_orin_nx_16g"]["max_mem_freq"]

        try:
            if os.path.exists(EMC_CAP_PATH):
                with open(EMC_CAP_PATH, "w") as f:
                    f.write(MAX_VALID_FREQ)
                print(f"  [OK] MEM cap reset to {MAX_VALID_FREQ}")
        except Exception as e:
            print(f"  [WARN] MEM restore error: {e}")

        print("MEM DVFS set to default: simple_ondemand)")
        
    def set_cpu_freq(self, cpu, freq):
        try:
            # first switch to userspace governor
            with open(f"{self.cpu_path}/cpu{cpu}/cpufreq/scaling_governor", "w") as f:
                f.write("userspace")
            with open(f"{self.cpu_path}/cpu{cpu}/cpufreq/scaling_max_freq", "w") as f:
                f.write(str(freq))
            with open(f"{self.cpu_path}/cpu{cpu}/cpufreq/scaling_min_freq", "w") as f:
                f.write(str(freq))
            with open(f"{self.cpu_path}/cpu{cpu}/cpufreq/scaling_setspeed", "w") as f:
                f.write(str(freq))
        except Exception as e:
            print(f"Set CPU freq failed: {e}")

    def set_gpu_freq(self, freq):
        try:
            with open(f"{self.gpu_path}/governor", "w") as f:
                f.write("userspace")
            with open(f"{self.gpu_path}/max_freq", "w") as f:
                f.write(str(freq))
            with open(f"{self.gpu_path}/min_freq", "w") as f:
                f.write(str(freq))
        except Exception as e:
            print(f"Set GPU freq failed: {e}")

    def set_mem_freq(self, freq):
        try:
            val = str(int(freq)).strip()

            lock_path = f"{self.mem_path}/mrq_rate_locked"
            if os.path.exists(lock_path):
                with open(lock_path, "w") as f: f.write("1")

            rate_path = f"{self.mem_path}/rate"
            if os.path.exists(rate_path):
                with open(rate_path, "w") as f: f.write(val)

            state_path = f"{self.mem_path}/state"
            if os.path.exists(state_path):
                with open(state_path, "w") as f: f.write("1")

            for node in ["min_rate", "max_rate"]:
                node_path = f"{self.mem_path}/{node}"
                if os.path.exists(node_path):
                    with open(node_path, "w") as f: f.write(val)
                    
            print(f"  [OK] MEM freq locked at {val}")
        except Exception as e:
            print(f"Set MEM freq {freq} failed: {e}")

    def read_cpu_freq(self, cpu):
        path = f"{self.cpu_path}/cpu{cpu}/cpufreq/scaling_cur_freq"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return int(f.read().strip())
            except Exception as e:
                print(f"Read CPU freq failed: {e}")
    
    def read_gpu_freq(self):
        path = f"{self.gpu_path}/cur_freq"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return int(f.read().strip())
            except Exception as e:
                print(f"Read GPU freq failed: {e}")

    def read_mem_freq(self):
        EMC_CAP_PATH = self.config["jetson_orin_nx_16g"]["emc_cap_path"]
        try:
            target_path = EMC_CAP_PATH if os.path.exists(EMC_CAP_PATH) else f"{self.mem_path}/rate"
            
            if os.path.exists(target_path):
                with open(target_path, "r") as f:
                    content = f.read().strip() # 只读一次并保存结果
                    if not content:
                        return 0
                    freq = int(content)
                    # print(f"[read_mem_freq] Read MEM freq {freq}")
                    return freq
            return 0
        except Exception as e:
            print(f"Read MEM freq failed: {e}")
            return 0

    # ===============================
    # Power Reading
    # ===============================
    def read_power(self):
        try:
            with open(os.path.join(self.ina_path, "in1_input"), "r") as f: v = int(f.read()) / 1000.0
            with open(os.path.join(self.ina_path, "curr1_input"), "r") as f: i = int(f.read()) / 1000.0
            return v * i
        except Exception as e:
            print(f"Read board power failed: {e}")
            return 0.0
