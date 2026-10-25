#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from phonon_onnx_bundles.catalog import stable_model_catalog

def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def main():
    actual=load("catalog/models.json"); expected=stable_model_catalog()
    if actual!=expected: raise SystemExit("catalog/models.json is stale; run python scripts/build_catalog.py")
    for data,schema in [(actual,"schemas/model-catalog.schema.json"),(load("catalog/bundles.json"),"schemas/bundle-catalog.schema.json")]:
        Draft202012Validator(load(schema)).validate(data)
    bundles=load("catalog/bundles.json")["bundles"]
    keys=set();
    for b in bundles:
        k=(b["model_key"],b["precision"],b["source_revision"])
        if k in keys: raise SystemExit(f"duplicate bundle entry: {k}")
        keys.add(k)
    print(f"catalog ok: {len(actual['models'])} models, {len(bundles)} published bundles")
if __name__=="__main__": main()
