"""Benchmark model loaders."""

from .smolvla import load_policy as load_smolvla
from .tinyvla import load_policy as load_tinyvla

MODEL_LOADERS = {
    "smolvla": load_smolvla,
    "tinyvla": load_tinyvla,
}


def get_model_loader(name: str):
    key = name.lower().strip()
    try:
        return MODEL_LOADERS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported model name: {name!r}") from exc
