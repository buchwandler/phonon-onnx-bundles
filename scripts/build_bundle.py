#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from phonon_onnx_bundles.catalog import model_spec
from phonon_onnx_bundles.licensing import license_metadata
from phonon_onnx_bundles.manifest import GRAPH_FILES, inventory, sha256, write_json
from phonon_onnx_bundles.release_assets import flattened_release_asset_name
from phonon_onnx_bundles.release_version import validate_release_version
from scripts.export_phonon import run_export

_RUNTIME_KIND = "phonon-streaming-onnx-v1"
_PRECISIONS = ("fp32", "q8", "q4", "fp16")
_RUNTIME_FIELDS: dict[str, tuple[str, ...]] = {
    "sample_rate": ("sample_rate", "sampling_rate"),
    "frame_samples": ("frame_samples",),
    "frame_rate": ("frame_rate",),
    "d_model": ("d_model", "hidden_size"),
    "latent_dim": ("latent_dim",),
    "num_layers": ("num_layers", "flow_layers", "num_flow_layers"),
    "num_heads": ("num_heads", "heads"),
    "head_dim": ("head_dim",),
    "mimi_state": ("mimi_state",),
}
_GENERATION_FIELDS: dict[str, tuple[str, ...]] = {
    "temperature_default": ("temperature_default", "temperature"),
    "eos_threshold_default": ("eos_threshold_default", "eos_threshold"),
    "duration_delta_default": ("duration_delta_default", "duration_delta"),
    "frames_after_eos": ("frames_after_eos",),
    "min_frames_before_eos": ("min_frames_before_eos",),
    "max_frames": ("max_frames", "max_frames_default"),
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _lookup(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[bool, Any]:
    for key in keys:
        if key in mapping:
            return True, mapping[key]
    return False, None


def _config_value(config: Mapping[str, Any], section: str, aliases: Sequence[str]) -> Any:
    """Read documented runtime, ONNX, Mimi, and Flow-LM locations plus root aliases."""
    onnx_config = _mapping(config.get("onnx"))
    sections = (section, "onnx", "mimi", "flow_lm", "root") if section == "runtime" else (section, "onnx.generation", "root")
    for location in sections:
        if location == "root":
            mapping = config
        elif location == "onnx.generation":
            mapping = _mapping(onnx_config.get("generation"))
        else:
            mapping = _mapping(config.get(location)) if location != "onnx" else onnx_config
        found, value = _lookup(mapping, aliases)
        if found:
            return value
    return None

def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Phonon config {field} must be a positive integer")
    return value


def _finite_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Phonon config {field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"Phonon config {field} is outside its valid range")
    return number


def _derive_sample_rate(config: Mapping[str, Any]) -> int | None:
    """Derive audio sample rate from the exported frame size and frame rate."""
    frame_samples = _config_value(config, "runtime", ("frame_samples",))
    frame_rate = _config_value(config, "runtime", ("frame_rate",))
    if isinstance(frame_samples, bool) or not isinstance(frame_samples, int) or frame_samples <= 0:
        return None
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float)):
        return None
    derived = float(frame_samples) * float(frame_rate)
    if not math.isfinite(derived) or derived <= 0 or derived % 1 != 0:
        return None
    return int(derived)



def _generation_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for normalized, aliases in _GENERATION_FIELDS.items():
        value = _config_value(config, "generation", aliases)
        if value is None:
            result[normalized] = None
        elif normalized in {"frames_after_eos", "min_frames_before_eos", "max_frames"}:
            minimum = 1 if normalized == "max_frames" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Phonon generation {normalized} is invalid")
            result[normalized] = value
        else:
            minimum = 0.0 if normalized == "temperature_default" else None
            result[normalized] = _finite_number(value, normalized, minimum=minimum)
    return result


def _voice_metadata(config: Mapping[str, Any], bundle: Path) -> tuple[list[dict[str, Any]], str, list[str]]:
    raw_voices = config.get("voices")
    voices: list[dict[str, Any]] = []
    if isinstance(raw_voices, Mapping):
        entries = []
        for key, value in raw_voices.items():
            if not isinstance(value, Mapping):
                raise ValueError("Phonon config voice entries must be objects")
            entries.append({"name": key, **value})
    elif isinstance(raw_voices, Sequence) and not isinstance(raw_voices, (str, bytes)):
        entries = list(raw_voices)
    else:
        raise ValueError("Phonon config must declare its bundled voice inventory")

    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Phonon config voice entries must be objects")
        name = entry.get("name") or entry.get("id")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Phonon config contains an unsafe voice name: {name!r}")
        if name in names:
            raise ValueError(f"Phonon config contains duplicate voice {name!r}")
        names.add(name)
        prompt = bundle / "voices" / f"{name}.bin"
        if not prompt.is_file():
            raise ValueError(f"Phonon config voice {name!r} has no bundled prompt file")
        detail: dict[str, Any] = {"name": name}
        for field in ("display_name", "voice_id", "locale", "gender", "frames"):
            value = entry.get(field)
            if value is not None:
                detail[field] = value
        if "frames" in detail:
            detail["frames"] = _positive_int(detail["frames"], f"voices.{name}.frames")
        voices.append(detail)

    default_voice = config.get("default_voice")
    if not isinstance(default_voice, str) or default_voice not in names:
        raise ValueError("Phonon config default_voice must name a bundled voice")
    raw_languages = config.get("languages")
    if raw_languages is None:
        languages = []
    elif isinstance(raw_languages, Sequence) and not isinstance(raw_languages, (str, bytes)):
        if any(not isinstance(language, str) or not language.strip() for language in raw_languages):
            raise ValueError("Phonon config languages must be non-empty strings")
        languages = sorted({language.strip().lower() for language in raw_languages})
    else:
        raise ValueError("Phonon config languages must be a list")
    return voices, default_voice, languages


def normalize_phonon_config(
    raw_config: Mapping[str, Any],
    bundle_dir: Path,
    model_languages: Sequence[str],
) -> dict[str, Any]:
    """Convert exporter metadata into the producer-owned canonical bundle contract."""
    raw_runtime = _mapping(raw_config.get("runtime"))
    if "kind" in raw_runtime:
        kind = raw_runtime["kind"]
    elif "runtime_kind" in raw_config:
        kind = raw_config["runtime_kind"]
    elif "kind" in raw_config:
        kind = raw_config["kind"]
    else:
        kind = _RUNTIME_KIND
    if kind != _RUNTIME_KIND:
        raise ValueError(f"Phonon config runtime kind must be {_RUNTIME_KIND!r}")

    runtime: dict[str, Any] = {"kind": _RUNTIME_KIND, "graphs": list(GRAPH_FILES)}
    for field, aliases in _RUNTIME_FIELDS.items():
        value = _config_value(raw_config, "runtime", aliases)
        if field == "sample_rate" and value is None:
            value = _derive_sample_rate(raw_config)
        if field == "mimi_state":
            if not isinstance(value, (Mapping, list, tuple)) or not value:
                raise ValueError("Phonon config must declare non-empty onnx.mimi_state")
            runtime[field] = value
        elif field == "frame_rate":
            runtime[field] = _finite_number(value, field, minimum=0.0)
            if runtime[field] == 0:
                raise ValueError("Phonon config frame_rate must be positive")
        else:
            runtime[field] = _positive_int(value, field)

    generation = _generation_metadata(raw_config)
    voices, default_voice, config_languages = _voice_metadata(raw_config, bundle_dir)
    languages = config_languages or sorted({str(value).lower() for value in model_languages})
    if not languages:
        raise ValueError("Phonon bundle must declare at least one language")
    return {
        "schema_version": 1,
        "runtime": runtime,
        "generation": generation,
        "voices": voices,
        "default_voice": default_voice,
        "languages": languages,
    }


def runtime_metadata(
    config: Mapping[str, Any], bundle: Path, model_languages: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str, list[str]]:
    """Compatibility view of canonical metadata for existing helper callers."""
    canonical = normalize_phonon_config(config, bundle, model_languages)
    return (
        canonical["runtime"],
        canonical["generation"],
        canonical["voices"],
        canonical["default_voice"],
        canonical["languages"],
    )

def write_candidate_sidecars(final: Path, manifest: dict[str, Any]) -> None:
    """Write checksums, recipient notes, a ZIP, and flat per-file release assets."""
    tag = manifest["tag"]
    write_json(final / "release-manifest.json", manifest)
    sums = [f"{rec['sha256']}  bundle/{rec['path']}" for rec in manifest["assets"]]
    sums.append(f"{sha256(final / 'release-manifest.json')}  release-manifest.json")
    (final / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    release_assets = final / "release-assets"
    if release_assets.exists():
        shutil.rmtree(release_assets)
    release_assets.mkdir(parents=True)
    names: set[str] = set()
    for record in manifest["assets"]:
        name = flattened_release_asset_name(record["path"])
        if name in names:
            raise ValueError(f"flattened release asset collision for {record['path']!r}")
        names.add(name)
        source = final / "bundle" / record["path"]
        if not source.is_file() or source.stat().st_size != record["bytes"] or sha256(source) != record["sha256"]:
            raise ValueError(f"bundle artifact does not match inventory: {record['path']}")
        shutil.copy2(source, release_assets / name)

    source = manifest["source"]
    license_info = manifest["license"]
    license_name = "CC BY-NC 4.0" if license_info["spdx"] == "CC-BY-NC-4.0" else license_info["spdx"]
    notes = (
        f"# {tag}\n\n"
        f"ONNX conversion of `{source['repo_id']}` at `{source['resolved_revision']}`.\n\n"
        f"Release set: {manifest['release_version']}\nSource revision: {source['resolved_revision']}\n\n"
        f"Source:\n{source['uri']}\n\n"
        f"License:\n{license_name} — {license_info['uri']}\n\n"
        "This release contains a technical ONNX conversion of the upstream model. "
        "See `ATTRIBUTION.md` inside the ZIP for retained upstream attribution, "
        "modification information, and the warranty-disclaimer reference. Individual "
        "runtime payload files are also published as `bundle--...` release assets.\n"
    )
    (final / "release-notes.md").write_text(notes, encoding="utf-8")

    archive = final / f"{tag}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted((final / "bundle").rglob("*")):
            if path.is_file():
                output.write(path, arcname=f"{tag}/{path.relative_to(final / 'bundle').as_posix()}")
        output.write(final / "release-manifest.json", arcname=f"{tag}/release-manifest.json")
        output.write(final / "SHA256SUMS", arcname=f"{tag}/SHA256SUMS")


def build(
    model_key: str,
    precision: str,
    dist: Path,
    *,
    allow_noncommercial: bool = False,
    release_version: str,
    revision: str | None = None,
    voices_revision: str | None = None,
    exporter_revision: str | None = None,
) -> Path:
    validate_release_version(release_version)
    if precision not in _PRECISIONS:
        raise ValueError(f"unsupported precision {precision!r}")
    spec = model_spec(model_key)
    with tempfile.TemporaryDirectory(prefix="phonon-candidate-") as temporary:
        payload = Path(temporary) / "payload"
        provenance = run_export(
            model_key,
            precision,
            payload,
            allow_noncommercial=allow_noncommercial,
            revision=revision,
            voices_revision=voices_revision,
            exporter_revision=exporter_revision,
        )
        source_sha = provenance["source"]["resolved_revision"]
        tag = f"{model_key}-onnx-{precision}-v{release_version}-{source_sha[:8]}-files-v1"
        final = dist / tag
        if final.exists():
            raise FileExistsError(f"candidate already exists and will not be replaced: {final}")
        final.mkdir(parents=True)
        shutil.copytree(payload, final / "bundle")
        config_path = final / "bundle" / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("exported bundle has no valid config.json") from exc
        if not isinstance(config, Mapping):
            raise ValueError("exported Phonon config must be a JSON object")
        canonical_config = normalize_phonon_config(config, final / "bundle", spec.get("languages", ()))
        write_json(config_path, canonical_config)
        runtime = canonical_config["runtime"]
        generation = canonical_config["generation"]
        voices = canonical_config["voices"]
        default_voice = canonical_config["default_voice"]
        languages = canonical_config["languages"]

        restrictions = [] if spec["commercial_use"] else ["non-commercial"]
        license_details = license_metadata(spec["license"])
        license_info = {
            "spdx": spec["license"],
            "uri": spec["license_uri"],
            "commercial_use": spec["commercial_use"],
            "restrictions": restrictions,
            "attribution_path": "ATTRIBUTION.md",
        }
        if "legalcode_uri" in license_details:
            license_info["legalcode_uri"] = spec.get("license_legalcode_uri", license_details["legalcode_uri"])
            license_info["warranty_disclaimer_section"] = "Section 5"

        manifest = {
            "schema_version": 3,
            "release_version": release_version,
            "distribution": {"kind": "individual-files-v1"},
            "tag": tag,
            "model_key": model_key,
            "precision": precision,
            "source": provenance["source"],
            "exporter": provenance["exporter"],
            "runtime": runtime,
            "generation": generation,
            "voices": voices,
            "default_voice": default_voice,
            "languages": languages,
            "license": license_info,
            "modifications": provenance["modifications"],
            "assets": inventory(final / "bundle"),
        }
        write_candidate_sidecars(final, manifest)
        return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--precision", default="fp32", choices=_PRECISIONS)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--revision")
    parser.add_argument("--voices-revision")
    parser.add_argument("--exporter-revision")
    parser.add_argument("--release-version", required=True, help="release-set version, e.g. 0.2.0")
    parser.add_argument(
        "--acknowledge-noncommercial-license",
        "--allow-noncommercial",
        dest="allow_noncommercial",
        action="store_true",
        help="Acknowledge the noncommercial model-license condition; this does not grant commercial rights",
    )
    args = parser.parse_args()
    print(build(
        args.model_key,
        args.precision,
        args.dist,
        allow_noncommercial=args.allow_noncommercial,
        revision=args.revision,
        voices_revision=args.voices_revision,
        exporter_revision=args.exporter_revision,
        release_version=args.release_version,
    ))


if __name__ == "__main__":
    main()
