#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from phonon_onnx_bundles.catalog import stable_model_catalog


def load(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    actual = load("catalog/models.json")
    expected = stable_model_catalog()
    if actual != expected:
        raise SystemExit("catalog/models.json is stale; run python scripts/build_catalog.py")
    for data, schema_path in (
        (actual, "schemas/model-catalog.schema.json"),
        (load("catalog/bundles.json"), "schemas/bundle-catalog.schema.json"),
    ):
        schema = load(schema_path)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(data)

    bundles = load("catalog/bundles.json")["bundles"]
    variants: set[tuple[str, str]] = set()
    for bundle in bundles:
        variant = (bundle["model_key"], bundle["precision"])
        if variant in variants:
            raise SystemExit(f"duplicate published model+precision variant: {variant}")
        variants.add(variant)
        paths = [artifact["path"] for artifact in bundle["artifacts"]]
        if len(paths) != len(set(paths)):
            raise SystemExit(f"duplicate artifact path in {bundle['tag']}")
    print(f"catalog ok: {len(actual['models'])} models, {len(bundles)} installable bundles")


if __name__ == "__main__":
    main()
