from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from phonon_onnx_bundles.release_assets import flattened_release_asset_name
from scripts import refresh_catalog


def _manifest(tag: str = "phonon-31m-en-onnx-fp32-test-files-v1") -> dict:
    return {
        "schema_version": 2,
        "distribution": {"kind": "individual-files-v1"},
        "tag": tag,
        "model_key": "phonon-31m-en",
        "precision": "fp32",
        "source": {
            "resolved_revision": "a" * 40,
            "uri": "https://huggingface.co/gradium/phonon-31m-en/tree/" + "a" * 40,
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


def _release(manifest: dict, *, omit: str | None = None) -> dict:
    assets = [{"name": "release-manifest.json", "url": "https://api.example/manifest"}]
    for record in manifest["assets"]:
        name = flattened_release_asset_name(record["path"])
        if name == omit:
            continue
        assets.append(
            {
                "name": name,
                "size": record["bytes"],
                "browser_download_url": f"https://github.com/example/release/download/{name}",
            }
        )
    return {
        "tag_name": manifest["tag"],
        "published_at": "2026-10-26T00:00:00Z",
        "draft": False,
        "prerelease": False,
        "assets": assets,
    }


def _patch_api(monkeypatch, releases: list[dict], manifest: dict) -> None:
    def request(url: str, token: str | None, accept: str = "application/vnd.github+json") -> bytes:
        if url == "https://api.github.com/repos/example/bundles/releases?per_page=100":
            return json.dumps(releases).encode()
        if url == "https://api.example/manifest":
            return json.dumps(manifest).encode()
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr(refresh_catalog, "request_bytes", request)


def test_refresh_emits_per_file_v2_rows_and_validates_schema(tmp_path: Path, monkeypatch):
    manifest = _manifest()
    _patch_api(monkeypatch, [_release(manifest)], manifest)
    output = tmp_path / "bundles.json"

    rows = refresh_catalog.refresh("example/bundles", output, None)

    assert len(rows) == 1
    assert rows[0]["model_key"] == "phonon-31m-en"
    assert rows[0]["precision"] == "fp32"
    assert rows[0]["artifacts"][0]["path"] == "config.json"
    assert rows[0]["artifacts"][0]["download_url"].endswith("bundle--config.json")
    data = json.loads(output.read_text(encoding="utf-8"))
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/bundle-catalog.schema.json").read_text())
    Draft202012Validator(schema).validate(data)


def test_refresh_skips_legacy_zip_only_release(tmp_path: Path, monkeypatch):
    legacy = {"tag": "legacy", "assets": []}
    release = {
        "tag_name": "legacy",
        "draft": False,
        "prerelease": False,
        "assets": [{"name": "release-manifest.json", "url": "https://api.example/manifest"}],
    }
    _patch_api(monkeypatch, [release], legacy)
    assert refresh_catalog.refresh("example/bundles", tmp_path / "bundles.json", None) == []


def test_refresh_fails_when_manifest_payload_asset_is_missing(tmp_path: Path, monkeypatch):
    manifest = _manifest()
    release = _release(manifest, omit="bundle--config.json")
    _patch_api(monkeypatch, [release], manifest)
    with pytest.raises(ValueError, match="missing release payload"):
        refresh_catalog.refresh("example/bundles", tmp_path / "bundles.json", None)


def test_refresh_rejects_duplicate_model_precision_releases(tmp_path: Path, monkeypatch):
    first = _manifest()
    second = _manifest("phonon-31m-en-onnx-fp32-other-files-v1")
    second["source"]["resolved_revision"] = "d" * 40
    second["source"]["uri"] = "https://huggingface.co/gradium/phonon-31m-en/tree/" + "d" * 40
    releases = [_release(first), _release(second)]
    # Each release has its own manifest endpoint in GitHub; keep this fixture tiny by
    # exposing the current payload based on the request URL.
    def request(url: str, token: str | None, accept: str = "application/vnd.github+json") -> bytes:
        if url.endswith("releases?per_page=100"):
            return json.dumps(releases).encode()
        if url.endswith("/manifest") or url.endswith("/manifest-2"): 
            tag = next(item["tag_name"] for item in releases if item["assets"][0]["url"] == url)
            return json.dumps(first if tag == first["tag"] else second).encode()
        raise AssertionError(url)

    # Give the second release a separate manifest URL for the fixture.
    releases[1]["assets"][0]["url"] = "https://api.example/manifest-2"
    monkeypatch.setattr(refresh_catalog, "request_bytes", request)
    with pytest.raises(ValueError, match="duplicate published Phonon variant"):
        refresh_catalog.refresh("example/bundles", tmp_path / "bundles.json", None)
