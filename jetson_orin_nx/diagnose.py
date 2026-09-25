"""Read-only environment report; run before selecting Jetson CUDA wheels."""
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
print('OS:', platform.platform(), 'machine:', platform.machine(), 'Python:', platform.python_version())
p = Path('/etc/nv_tegra_release')
print('L4T:', p.read_text().strip() if p.exists() else 'not a Jetson/L4T installation')
for name in ['torch', 'torchvision', 'lerobot', 'transformers']:
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, 'not installed')
try:
    import torch
    print('CUDA available:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('Memory free/total:', torch.cuda.mem_get_info())
except ImportError:
    pass
