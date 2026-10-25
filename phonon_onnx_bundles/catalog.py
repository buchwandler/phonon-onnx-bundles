from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registry" / "models.json"

class CatalogError(RuntimeError):
    pass

def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CatalogError(f"expected JSON object in {path}")
    return data

def registry() -> dict[str, Any]:
    return read_json(REGISTRY)

def model_spec(key: str) -> dict[str, Any]:
    data = registry()
    try:
        spec = data["models"][key]
    except KeyError as exc:
        raise CatalogError(f"unknown model key {key!r}") from exc
    return dict(spec)

def stable_model_catalog(data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = registry() if data is None else data
    precisions = list(data["precisions"])
    return {
        "schema_version": 1,
        "models": [
            {"key": key, **dict(spec), "precisions": precisions}
            for key, spec in sorted(data["models"].items())
        ],
    }
