```python
python3 jetson_orin_nx/benchmark/benchmark_dvfs.py
```

Usage examples
```
# Default run (uses workspace model path and writes CSV):
```python
python3 jetson_orin_nx/benchmark/benchmark_dvfs.py
```

# Enable GPU energy sampling (requires NVML / pynvml):
```python
python3 jetson_orin_nx/benchmark/benchmark_dvfs.py --collect-energy
```

# Specify a model and output file:
```python
python3 jetson_orin_nx/benchmark/benchmark_dvfs.py --model /path/to/model --output /path/to/results.csv
```

# Custom warmup / run counts:
```python
python3 jetson_orin_nx/benchmark/benchmark_dvfs.py --warmup 10 --runs 50
```

Notes:
- Install NVML Python bindings for energy collection: `pip install pynvml`
- The script uses CUDA if available; otherwise it runs on CPU.