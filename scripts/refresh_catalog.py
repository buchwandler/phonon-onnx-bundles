#!/usr/bin/env python3
"""Rebuild the installable v2 catalog from immutable individual-file releases."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from phonon_onnx_bundles.catalog import registry
from phonon_onnx_bundles.manifest import write_json
from phonon_onnx_bundles.release_assets import (
    flattened_release_asset_name,
    legacy_github_asset_name,
)
from phonon_onnx_bundles.release_version import validate_release_version

ROOT = Path(__file__).resolve().parents[1]


def request_bytes(url: str, token: str | None, accept: str = "application/vnd.github+json") -> bytes:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "phonon-onnx-bundles-catalog",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return response.read()


def _read_manifest(release: Mapping[str, Any], token: str | None) -> dict[str, Any] | None:
    assets = release.get("assets", [])
    manifests = [asset for asset in assets if asset.get("name") == "release-manifest.json"]
    if len(manifests) != 1:
        return None
    manifest_asset = manifests[0]
    payload = json.loads(request_bytes(str(manifest_asset["url"]), token, "application/octet-stream"))
    if not isinstance(payload, dict):
        raise ValueError("release-manifest.json must contain an object")
    return payload


def _catalog_row(release: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    release_assets = release.get("assets", [])
    by_name: dict[str, Mapping[str, Any]] = {}
    for asset in release_assets:
        name = asset.get("name")
        if isinstance(name, str):
            if name in by_name:
                raise ValueError(f"release contains duplicate asset name {name!r}")
            by_name[name] = asset

    records = manifest.get("assets")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{manifest.get('tag')}: manifest has no payload assets")
    artifacts: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("release manifest asset entries must be objects")
        path = record.get("path")
        size = record.get("bytes")
        checksum = record.get("sha256")
        if not isinstance(path, str) or path in seen_paths:
            raise ValueError(f"release manifest has an invalid or duplicate path: {path!r}")
        seen_paths.add(path)
        remote_name = flattened_release_asset_name(path)
        remote_asset = by_name.get(remote_name)
        if remote_asset is None:
            legacy_name = legacy_github_asset_name(path)
            remote_name = legacy_name
            remote_asset = by_name.get(legacy_name)
        if remote_asset is None:
            raise ValueError(f"{manifest.get('tag')}: missing release payload {flattened_release_asset_name(path)!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"{path}: manifest size must be a positive integer")
        if remote_asset.get("size") != size:
            raise ValueError(f"{path}: release asset size does not match the manifest")
        url = remote_asset.get("browser_download_url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"{path}: release asset has no direct HTTPS download URL")
        artifacts.append({"path": path, "bytes": size, "sha256": checksum, "download_url": url})

    artifacts.sort(key=lambda artifact: artifact["path"])
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{manifest.get('tag')}: source metadata is missing")
    return {
        "tag": manifest["tag"],
        "release_version": manifest["release_version"],
        "model_key": manifest["model_key"],
        "precision": manifest["precision"],
        "source_revision": source["resolved_revision"],
        "source_uri": source.get("uri"),
        "runtime": manifest["runtime"],
        "generation": manifest["generation"],
        "license": manifest["license"],
        "voices": manifest["voices"],
        "default_voice": manifest["default_voice"],
        "languages": manifest["languages"],
        "artifacts": artifacts,
        "published_at": release.get("published_at"),
    }


def expected_variants() -> set[tuple[str, str]]:
    """Return the registry-derived model/precision release cross-product."""
    data = registry()
    return {
        (model_key, precision)
        for model_key in data["models"]
        for precision in data["precisions"]
    }


def _all_releases(repository: str, token: str | None):
    page = 1
    while True:
        payload = json.loads(
            request_bytes(
                f"https://api.github.com/repos/{repository}/releases?per_page=100&page={page}",
                token,
            )
        )
        if not isinstance(payload, list):
            raise ValueError("GitHub releases response must be a list")
        yield from payload
        if len(payload) < 100:
            return
        page += 1


def _variant_report(variants: set[tuple[str, str]]) -> str:
    return ", ".join(f"{model}/{precision}" for model, precision in sorted(variants))


def refresh(
    repository: str,
    output: Path,
    token: str | None,
    *,
    release_version: str,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    release_version = validate_release_version(release_version)
    rows: list[dict[str, Any]] = []
    for release in _all_releases(repository, token):
        if release.get("draft") or release.get("prerelease"):
            continue
        manifest = _read_manifest(release, token)
        if manifest is None:
            continue
        distribution = manifest.get("distribution")
        if distribution is None:
            # Existing immutable ZIP-only releases are human-downloadable, not installable.
            continue
        if not isinstance(distribution, Mapping) or distribution.get("kind") != "individual-files-v1":
            raise ValueError(f"{manifest.get('tag')}: unsupported distribution contract")
        # Schema-2 historical releases have no release-set identity and cannot activate a catalog.
        if manifest.get("schema_version") == 2 and "release_version" not in manifest:
            continue
        if manifest.get("release_version") != release_version:
            continue
        row = _catalog_row(release, manifest)
        if row["tag"] != release.get("tag_name"):
            raise ValueError("release tag and manifest tag do not match")
        rows.append(row)

    rows.sort(key=lambda row: (row["model_key"], row["precision"], row["source_revision"]))
    seen: set[tuple[str, str]] = set()
    for row in rows:
        variant = (row["model_key"], row["precision"])
        if variant in seen:
            raise ValueError(f"duplicate published Phonon variant in release {release_version}: {variant}")
        seen.add(variant)

    if require_complete:
        expected = expected_variants()
        missing = expected - seen
        extra = seen - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing variants: {_variant_report(missing)}")
            if extra:
                details.append(f"extra variants: {_variant_report(extra)}")
            raise ValueError(f"release {release_version} is incomplete: {'; '.join(details)}")

    data = {"schema_version": 2, "release_version": release_version, "bundles": rows}
    schema = json.loads((ROOT / "schemas/bundle-catalog.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(data)
    write_json(output, data)
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--output", type=Path, default=Path("catalog/bundles.json"))
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if not args.repository:
        parser.error("--repository or GITHUB_REPOSITORY is required")
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    rows = refresh(args.repository, args.output, token, release_version=args.release_version, require_complete=args.require_complete)
    print(f"wrote {len(rows)} installable bundles to {args.output}")


if __name__ == "__main__":
    main()
