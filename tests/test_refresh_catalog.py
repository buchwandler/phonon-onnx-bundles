from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from phonon_onnx_bundles.release_assets import flattened_release_asset_name
from scripts import refresh_catalog


API = "https://api.github.com/repos/example/bundles/releases?per_page=100&page="


def _manifest(
    *,
    model_key: str = "phonon-31m-en",
    precision: str = "fp32",
    release_version: str = "0.2.0",
    source_sha: str = "a" * 40,
) -> dict:
    tag = f"{model_key}-onnx-{precision}-v{release_version}-{source_sha[:8]}-files-v1"
    return {
        "schema_version": 3,
        "release_version": release_version,
        "distribution": {"kind": "individual-files-v1"},
        "tag": tag,
        "model_key": model_key,
        "precision": precision,
        "source": {
            "resolved_revision": source_sha,
            "uri": f"https://huggingface.co/{model_key}/tree/{source_sha}",
        },
        "runtime": {
            "kind": "phonon-streaming-onnx-v1",
            "graphs": [
                "onnx/text_conditioner.onnx",
                "onnx/flow_lm_main.onnx",
                "onnx/flow_lm_flow.onnx",
                "onnx/mimi_decoder.onnx",
            ],
            "sample_rate": 24000,
            "frame_samples": 1920,
            "frame_rate": 12.5,
            "d_model": 512,
            "latent_dim": 32,
            "num_layers": 6,
            "num_heads": 8,
            "head_dim": 64,
            "mimi_state": [{"name": "cache", "shape": [1], "dtype": "float32"}],
        },
        "generation": {
            "temperature_default": 0.8,
            "eos_threshold_default": None,
            "duration_delta_default": 0.0,
            "frames_after_eos": None,
            "min_frames_before_eos": None,
            "max_frames": None,
        },
        "license": {
            "spdx": "Apache-2.0",
            "uri": "https://www.apache.org/licenses/LICENSE-2.0",
            "commercial_use": True,
            "restrictions": [],
            "attribution_path": "ATTRIBUTION.md",
        },
        "voices": [{"name": "easton", "display_name": "Easton", "locale": "en-US", "gender": "male", "frames": 125}],
        "default_voice": "easton",
        "languages": ["en"],
        "assets": [
            {"path": "config.json", "bytes": 12, "sha256": "b" * 64},
            {"path": "onnx/text_conditioner.onnx", "bytes": 34, "sha256": "c" * 64},
        ],
    }


def _release(manifest: dict, *, index: int = 0, omit: str | None = None, with_manifest: bool = True) -> dict:
    manifest_url = f"https://api.example/manifest-{index}"
    assets = [{"name": "release-manifest.json", "url": manifest_url}] if with_manifest else []
    for record in manifest["assets"]:
        name = flattened_release_asset_name(record["path"])
        if name == omit:
            continue
        assets.append({
            "name": name,
            "size": record["bytes"],
            "browser_download_url": f"https://github.com/example/release/download/{name}",
        })
    return {
        "tag_name": manifest["tag"],
        "published_at": "2026-10-26T00:00:00Z",
        "draft": False,
        "prerelease": False,
        "assets": assets,
        "manifest_url": manifest_url if with_manifest else None,
    }


def _patch_api(monkeypatch, pages: dict[int, list[dict]], manifests: dict[str, dict]) -> list[str]:
    requested: list[str] = []

    def request(url: str, token: str | None, accept: str = "application/vnd.github+json") -> bytes:
        requested.append(url)
        if url.startswith(API):
            page = int(url.rsplit("page=", 1)[1])
            return json.dumps(pages.get(page, [])).encode()
        if url in manifests:
            return json.dumps(manifests[url]).encode()
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr(refresh_catalog, "request_bytes", request)
    return requested




def test_refresh_emits_release_versioned_rows_and_validates_schema(tmp_path: Path, monkeypatch):
    manifest = _manifest()
    release = _release(manifest)
    _patch_api(monkeypatch, {1: [release]}, {release["manifest_url"]: manifest})
    output = tmp_path / "bundles.json"

    rows = refresh_catalog.refresh("example/bundles", output, None, release_version="0.2.0")

    assert len(rows) == 1
    assert rows[0]["model_key"] == "phonon-31m-en"
    assert rows[0]["precision"] == "fp32"
    assert rows[0]["release_version"] == "0.2.0"
    assert rows[0]["artifacts"][0]["path"] == "config.json"
    assert rows[0]["artifacts"][0]["download_url"].endswith("bundle--config.2Ejson")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["release_version"] == "0.2.0"
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/bundle-catalog.schema.json").read_text())
    Draft202012Validator(schema).validate(data)


def test_refresh_skips_legacy_zip_only_release(tmp_path: Path, monkeypatch):
    legacy = {"tag": "legacy", "assets": []}
    release = {
        "tag_name": "legacy",
        "draft": False,
        "prerelease": False,
        "assets": [{"name": "release-manifest.json", "url": "https://api.example/legacy"}],
    }
    _patch_api(monkeypatch, {1: [release]}, {"https://api.example/legacy": legacy})
    assert refresh_catalog.refresh("example/bundles", tmp_path / "bundles.json", None, release_version="0.2.0") == []


def test_refresh_fails_when_manifest_payload_asset_is_missing(tmp_path: Path, monkeypatch):
    manifest = _manifest()
    release = _release(manifest, omit="bundle--config.2Ejson")
    _patch_api(monkeypatch, {1: [release]}, {release["manifest_url"]: manifest})
    with pytest.raises(ValueError, match="missing release payload"):
        refresh_catalog.refresh("example/bundles", tmp_path / "bundles.json", None, release_version="0.2.0")


def test_refresh_filters_other_versions_before_duplicate_detection(tmp_path: Path, monkeypatch):
    old = _manifest(release_version="0.1.0")
    active = _manifest(release_version="0.2.0", source_sha="d" * 40)
    releases = [_release(old, index=1), _release(active, index=2)]
    manifests = {release["manifest_url"]: manifest for release, manifest in zip(releases, [old, active])}
    _patch_api(monkeypatch, {1: releases}, manifests)

    rows = refresh_catalog.refresh("example/bundles", tmp_path / "bundles.json", None, release_version="0.2.0")

    assert len(rows) == 1
    assert rows[0]["release_version"] == "0.2.0"
    assert rows[0]["source_revision"] == "d" * 40


def test_duplicate_variants_within_selected_version_fail(tmp_path: Path, monkeypatch):
    first = _manifest()
    second = _manifest(source_sha="d" * 40)
    releases = [_release(first, index=1), _release(second, index=2)]
    manifests = {releases[0]["manifest_url"]: first, releases[1]["manifest_url"]: second}
    _patch_api(monkeypatch, {1: releases}, manifests)
    with pytest.raises(ValueError, match="duplicate published Phonon variant"):
        refresh_catalog.refresh("example/bundles", tmp_path / "bundles.json", None, release_version="0.2.0")


def test_refresh_paginates_beyond_100_releases(tmp_path: Path, monkeypatch):
    ignored = [{"tag_name": f"legacy-{index}", "draft": False, "prerelease": False, "assets": []} for index in range(100)]
    manifest = _manifest()
    page_two = _release(manifest, index=101)
    requested = _patch_api(monkeypatch, {1: ignored, 2: [page_two]}, {page_two["manifest_url"]: manifest})

    rows = refresh_catalog.refresh("example/bundles", tmp_path / "bundles.json", None, release_version="0.2.0")

    assert len(rows) == 1
    assert API + "1" in requested
    assert API + "2" in requested


def test_expected_variants_derive_current_registry_cross_product():
    variants = refresh_catalog.expected_variants()
    assert len({model for model, _ in variants}) == 7
    assert len({precision for _, precision in variants}) == 4
    assert len(variants) == 28


def _complete_release_fixture():
    manifests = {}
    releases = []
    for index, (model_key, precision) in enumerate(sorted(refresh_catalog.expected_variants()), start=1):
        source_sha = f"{index:040x}"
        manifest = _manifest(model_key=model_key, precision=precision, source_sha=source_sha)
        release = _release(manifest, index=index)
        manifests[release["manifest_url"]] = manifest
        releases.append(release)
    return releases, manifests


def test_complete_release_set_succeeds(tmp_path: Path, monkeypatch):
    releases, manifests = _complete_release_fixture()
    _patch_api(monkeypatch, {1: releases}, manifests)

    rows = refresh_catalog.refresh(
        "example/bundles", tmp_path / "bundles.json", None,
        release_version="0.2.0", require_complete=True,
    )

    assert len(rows) == 28


def test_complete_release_set_reports_missing_variant(tmp_path: Path, monkeypatch):
    releases, manifests = _complete_release_fixture()
    removed = releases.pop()
    manifests.pop(removed["manifest_url"])
    _patch_api(monkeypatch, {1: releases}, manifests)

    with pytest.raises(ValueError, match="missing variants:.*phonon-"):
        refresh_catalog.refresh(
            "example/bundles", tmp_path / "bundles.json", None,
            release_version="0.2.0", require_complete=True,
        )


def test_complete_release_set_reports_extra_variant(tmp_path: Path, monkeypatch):
    releases, manifests = _complete_release_fixture()
    extra = _manifest(model_key="unknown-model", precision="fp32", source_sha="f" * 40)
    extra_release = _release(extra, index=1000)
    releases.append(extra_release)
    manifests[extra_release["manifest_url"]] = extra
    _patch_api(monkeypatch, {1: releases}, manifests)

    with pytest.raises(ValueError, match="extra variants: unknown-model/fp32"):
        refresh_catalog.refresh(
            "example/bundles", tmp_path / "bundles.json", None,
            release_version="0.2.0", require_complete=True,
        )
