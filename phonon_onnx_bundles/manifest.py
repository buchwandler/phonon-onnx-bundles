from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

GRAPH_FILES = (
    "onnx/text_conditioner.onnx",
    "onnx/flow_lm_main.onnx",
    "onnx/flow_lm_flow.onnx",
    "onnx/mimi_decoder.onnx",
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    exclude = exclude or set()
    out=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel=p.relative_to(root).as_posix()
        if rel in exclude:
            continue
        out.append({"path":rel,"bytes":p.stat().st_size,"sha256":sha256(p)})
    return out

def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
