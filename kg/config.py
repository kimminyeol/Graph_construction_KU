"""Config loading: YAML with `extends:` inheritance, `.env` secrets, run paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env loader. Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent:
        return _deep_merge(_load_yaml((path.parent / parent).resolve()), raw)
    return raw


@dataclass
class Paths:
    raw: Path          # shared across runs: API responses are reused when scaling up
    processed: Path    # per run
    exports: Path      # per run
    logs: Path

    @property
    def openalex_raw(self) -> Path:
        return self.raw / "openalex"


@dataclass
class Config:
    data: dict[str, Any]
    path: Path

    @property
    def run_name(self) -> str:
        return self.data["run_name"]

    @property
    def paths(self) -> Paths:
        p = self.data.get("paths", {})
        data_dir = PROJECT_ROOT / p.get("data_dir", "data")
        exports_dir = PROJECT_ROOT / p.get("exports_dir", "exports")
        return Paths(raw=data_dir / "raw",
                     processed=data_dir / "processed" / self.run_name,
                     exports=exports_dir / self.run_name,
                     logs=data_dir / "logs")

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


_REQUIRED = {"run_name": None, "corpus": ("max_papers", "years", "sources")}


def _check(data: dict, path: Path) -> None:
    for key, sub in _REQUIRED.items():
        if key not in data:
            raise ValueError(f"{path}: missing top-level key {key!r}")
        for s in sub or ():
            if s not in data[key]:
                raise ValueError(f"{path}: missing {key}.{s} (check YAML indentation)")
    for src in data["corpus"]["sources"]:
        for s in ("name", "kind", "share", "items"):
            if s not in src:
                raise ValueError(f"{path}: corpus source {src.get('name', '?')!r} missing {s!r}")


def load_config(path: str | Path) -> Config:
    load_dotenv()
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else PROJECT_ROOT / path
    data = _load_yaml(path.resolve())
    _check(data, path)
    return Config(data=data, path=path.resolve())
