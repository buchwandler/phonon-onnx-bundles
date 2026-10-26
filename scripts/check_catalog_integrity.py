#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from phonon_onnx_bundles.catalog import stable_model_catalog
from phonon_onnx_bundles.release_version import validate_release_version
from scripts.refresh_catalog import expected_variants


def load(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    actual = load("catalog/models.json")
    expected = stable_model_catalog()
    if actual != expected:
        raise SystemExit("catalog/models.json is stale; run python scripts/build_catalog.py")
    bundles_data = load("catalog/bundles.json")
    for data, schema_path in (
        (actual, "schemas/model-catalog.schema.json"),
        (bundles_data, "schemas/bundle-catalog.schema.json"),
    ):
        schema = load(schema_path)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(data)

    bundles = bundles_data["bundles"]
    active_version = bundles_data["release_version"]
    variants: set[tuple[str, str]] = set()
    for bundle in bundles:
        variant = (bundle["model_key"], bundle["precision"])
        if bundle["release_version"] != active_version:
            raise SystemExit(f"bundle {bundle['tag']} is not part of active release {active_version}")
        if variant in variants:
            raise SystemExit(f"duplicate published model+precision variant: {variant}")
        variants.add(variant)
        paths = [artifact["path"] for artifact in bundle["artifacts"]]
        if len(paths) != len(set(paths)):
            raise SystemExit(f"duplicate artifact path in {bundle['tag']}")
    if active_version is None:
        if bundles:
            raise SystemExit("catalog has bundles but no active release_version")
    else:
        validate_release_version(active_version)
        expected_variants_set = expected_variants()
        missing = expected_variants_set - variants
        extra = variants - expected_variants_set
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(f"{model}/{precision}" for model, precision in sorted(missing)))
            if extra:
                details.append("extra: " + ", ".join(f"{model}/{precision}" for model, precision in sorted(extra)))
            raise SystemExit(f"active release {active_version} is incomplete ({'; '.join(details)})")
    print(f"catalog ok: {len(actual['models'])} models, {len(bundles)} installable bundles")


if __name__ == "__main__":
    main()
