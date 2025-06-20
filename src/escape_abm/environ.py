"""Global environment."""

from typing import Any

_ENVIRON: dict[str, Any]


def get(key: str, default: Any = None) -> Any:
    return _ENVIRON.get(key, default)


def set(key: str, value: Any):
    _ENVIRON[key] = value


def remove(key: str):
    if key in _ENVIRON:
        del _ENVIRON[key]


def clear():
    _ENVIRON.clear()
