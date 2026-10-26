#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import onnx
from jsonschema import Draft202012Validator

from phonon_onnx_bundles.licensing import validate_license_metadata
from phonon_onnx_bundles.manifest import GRAPH_FILES, sha256
from phonon_onnx_bundles.release_assets import flattened_release_asset_name

REQ = {
    "onnx/text_conditioner.onnx": ({"input_ids", "duration_delta"}, {"inputs_embeds", "bos_embeds"}),
    "onnx/flow_lm_main.onnx": ({"inputs_embeds"}, {"hidden", "eos_logit"}),
    "onnx/flow_lm_flow.onnx": ({"hidden", "noise", "duration_delta"}, {"latent", "next_embeds"}),
    "onnx/mimi_decoder.onnx": ({"latent"}, {"audio"}),
}


class VerifyError(RuntimeError):
    pass


def verify(root: Path) -> None:
    manifest = json.loads((root / "release-manifest.json").read_text(encoding="utf-8"))
    schema = json.loads(Path("schemas/release-manifest.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)
    if manifest["distribution"]["kind"] != "individual-files-v1":
        raise VerifyError("unsupported release distribution kind")
    if manifest["runtime"]["graphs"] != list(GRAPH_FILES):
        raise VerifyError("runtime graph inventory does not match the Phonon ABI")

    expected_tag = (
        f"{manifest['model_key']}-onnx-{manifest['precision']}-v{manifest['release_version']}-"
        f"{manifest['source']['resolved_revision'][:8]}-files-v1"
    )
    if manifest["tag"] != expected_tag:
        raise VerifyError("release tag does not match model, precision, release_version, and source revision")

    config_path = root / "bundle" / "config.json"
    try:
        canonical_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerifyError("bundle/config.json is missing or invalid JSON") from exc
    expected_config_keys = {"schema_version", "runtime", "generation", "voices", "default_voice", "languages"}
    if not isinstance(canonical_config, dict) or set(canonical_config) != expected_config_keys:
        raise VerifyError("bundle/config.json does not have the canonical Phonon config shape")
    if type(canonical_config["schema_version"]) is not int or canonical_config["schema_version"] != 1:
        raise VerifyError("bundle/config.json has an unsupported schema_version")
    if canonical_config["runtime"] != manifest["runtime"]:
        raise VerifyError("bundle config runtime differs from release manifest")
    if canonical_config["generation"] != manifest["generation"]:
        raise VerifyError("bundle config generation differs from release manifest")
    for field in ("voices", "default_voice", "languages"):
        if canonical_config[field] != manifest[field]:
            raise VerifyError(f"bundle config {field} differs from release manifest")
    if canonical_config["runtime"].get("kind") != "phonon-streaming-onnx-v1":
        raise VerifyError("bundle config runtime kind is not canonical")
    if canonical_config["runtime"].get("graphs") != list(GRAPH_FILES):
        raise VerifyError("bundle config graph inventory does not match the Phonon ABI")
    records = manifest["assets"]
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        rel = record["path"]
        try:
            flattened_release_asset_name(rel)
        except ValueError as exc:
            raise VerifyError(f"unsafe bundle path {rel!r}") from exc
        if rel in by_path:
            raise VerifyError(f"duplicate bundle asset path {rel!r}")
        by_path[rel] = record
        path = root / "bundle" / rel
        if not path.is_file():
            raise VerifyError(f"missing {rel}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise VerifyError(f"asset mismatch {rel}")

    required = {"config.json", "tokenizer.json", "ATTRIBUTION.md", "export-provenance.json", *GRAPH_FILES}
    required.update(f"voices/{voice['name']}.bin" for voice in manifest["voices"])
    missing = required - set(by_path)
    if missing:
        raise VerifyError(f"manifest is missing required runtime assets: {', '.join(sorted(missing))}")
    voice_names = {voice["name"] for voice in manifest["voices"]}
    if manifest["default_voice"] not in voice_names:
        raise VerifyError("default_voice is not present in the voice inventory")

    try:
        validate_license_metadata(manifest, root / "bundle")
    except (ValueError, KeyError, TypeError) as exc:
        raise VerifyError(str(exc)) from exc

    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise VerifyError("missing SHA256SUMS")
    sums = set(sums_path.read_text(encoding="utf-8").splitlines())
    for record in records:
        expected = f"{record['sha256']}  bundle/{record['path']}"
        if expected not in sums:
            raise VerifyError(f"SHA256SUMS missing {record['path']} checksum")
    manifest_sum = f"{sha256(root / 'release-manifest.json')}  release-manifest.json"
    if manifest_sum not in sums:
        raise VerifyError("SHA256SUMS missing release-manifest.json checksum")

    release_assets = root / "release-assets"
    if not release_assets.is_dir():
        raise VerifyError("missing release-assets directory")
    expected_names: set[str] = set()
    for record in records:
        name = flattened_release_asset_name(record["path"])
        if name in expected_names:
            raise VerifyError(f"flattened release asset collision: {name}")
        expected_names.add(name)
        path = release_assets / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise VerifyError(f"missing or invalid flattened release asset {name}")
    actual_names = {path.name for path in release_assets.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise VerifyError("release-assets inventory differs from the manifest")

    for rel in GRAPH_FILES:
        path = root / "bundle" / rel
        model = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model)
        inputs = {value.name for value in model.graph.input}
        outputs = {value.name for value in model.graph.output}
        required_inputs, required_outputs = REQ[rel]
        if not required_inputs <= inputs:
            raise VerifyError(f"{rel}: missing inputs {sorted(required_inputs - inputs)}")
        if not required_outputs <= outputs:
            raise VerifyError(f"{rel}: missing outputs {sorted(required_outputs - outputs)}")

    archive = root / f"{manifest['tag']}.zip"
    if not archive.is_file():
        raise VerifyError("missing zip")
    with zipfile.ZipFile(archive) as zipped:
        bad = zipped.testzip()
        if bad:
            raise VerifyError(f"corrupt zip member {bad}")
        names = set(zipped.namelist())
        prefix = manifest["tag"] + "/"
        for rel in sorted(required):
            if prefix + rel not in names:
                raise VerifyError(f"zip missing {rel}")
        if prefix + "SHA256SUMS" not in names:
            raise VerifyError("zip missing SHA256SUMS")
    print(f"verified {manifest['tag']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    verify(args.candidate)


if __name__ == "__main__":
    main()
