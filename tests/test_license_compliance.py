import copy
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from jsonschema import Draft202012Validator

from phonon_onnx_bundles.catalog import model_spec, stable_model_catalog
from phonon_onnx_bundles.licensing import (
    hf_revision_uri,
    license_metadata,
    modification_notice,
    render_attribution,
    validate_license_metadata,
)
from phonon_onnx_bundles.manifest import inventory
from scripts.build_bundle import write_candidate_sidecars
from scripts.export_phonon import preserve_upstream_notices
from scripts import verify_candidate


def _candidate(tmp_path: Path, precision: str = "q4"):
    root = tmp_path / "candidate"
    bundle = root / "bundle"
    (bundle / "third_party/model").mkdir(parents=True)
    (bundle / "third_party/voices").mkdir(parents=True)
    (bundle / "third_party/model/README.md").write_text("upstream model card")
    (bundle / "third_party/voices/README.md").write_text("upstream voice card")

    sha = "0123456789abcdef0123456789abcdef01234567"
    source = {
        "repo_id": "gradium/phonon-40m-en",
        "requested_revision": "main",
        "resolved_revision": sha,
        "uri": hf_revision_uri("gradium/phonon-40m-en", sha),
        "license": "CC-BY-NC-4.0",
        "license_uri": license_metadata("CC-BY-NC-4.0")["uri"],
        "license_legalcode_uri": license_metadata("CC-BY-NC-4.0")["legalcode_uri"],
    }
    voices = {
        "repo_id": "gradium/phonon-voices",
        "resolved_revision": sha,
        "uri": hf_revision_uri("gradium/phonon-voices", sha),
        "license": "Apache-2.0",
        "license_uri": license_metadata("Apache-2.0")["uri"],
    }
    provenance = {"source": source, "voices": voices}
    (bundle / "ATTRIBUTION.md").write_text(render_attribution(provenance, precision))

    tag = f"phonon-40m-en-onnx-{precision}-01234567"
    for graph in verify_candidate.GRAPH_FILES:
        path = bundle / graph
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic ONNX placeholder")

    license_info = {
        "spdx": "CC-BY-NC-4.0",
        "uri": source["license_uri"],
        "legalcode_uri": source["license_legalcode_uri"],
        "commercial_use": False,
        "restrictions": ["non-commercial"],
        "attribution_path": "ATTRIBUTION.md",
        "warranty_disclaimer_section": "Section 5",
    }
    manifest = {
        "schema_version": 2,
        "tag": tag,
        "model_key": "phonon-40m-en",
        "precision": precision,
        "source": source,
        "exporter": {},
        "runtime": {},
        "voices": voices,
        "license": license_info,
        "modifications": {
            "modified": True,
            "modification_notice": modification_notice(precision),
            "precision": precision,
            "quantization": precision if precision in {"q4", "q8"} else None,
        },
        "assets": inventory(bundle),
    }
    write_candidate_sidecars(root, manifest)
    return root, manifest


def _stub_onnx(monkeypatch):
    def fake_load(path, load_external_data=True):
        rel = f"onnx/{Path(path).name}"
        inputs, outputs = verify_candidate.REQ[rel]
        return SimpleNamespace(
            graph=SimpleNamespace(
                input=[SimpleNamespace(name=name) for name in inputs],
                output=[SimpleNamespace(name=name) for name in outputs],
            )
        )

    monkeypatch.setattr(verify_candidate.onnx, "load", fake_load)
    monkeypatch.setattr(verify_candidate.onnx.checker, "check_model", lambda model: None)


def test_40m_registry_and_catalog_have_canonical_cc_metadata():
    spec = model_spec("phonon-40m-en")
    assert spec["license"] == "CC-BY-NC-4.0"
    assert spec["license_uri"] == "https://creativecommons.org/licenses/by-nc/4.0/"
    assert spec["license_legalcode_uri"] == "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en"
    assert spec["commercial_use"] is False
    rows = stable_model_catalog()["models"]
    assert all(row["license_uri"] == spec["license_uri"] for row in rows if row["license"] == spec["license"])


def test_generated_attribution_has_immutable_source_license_and_modifications():
    sha = "a" * 40
    source_uri = hf_revision_uri("gradium/phonon-40m-en", sha)
    provenance = {
        "source": {
            "repo_id": "gradium/phonon-40m-en",
            "resolved_revision": sha,
            "uri": source_uri,
            "license": "CC-BY-NC-4.0",
        },
        "voices": {
            "repo_id": "gradium/phonon-voices",
            "resolved_revision": sha,
            "license": "Apache-2.0",
        },
    }
    attribution = render_attribution(provenance, "q4")
    for fragment in (
        source_uri,
        sha,
        "CC-BY-NC-4.0",
        "https://creativecommons.org/licenses/by-nc/4.0/",
        "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en",
        "Section 5",
        "Converted to the phonon-streaming-onnx-v1",
        "two flow graphs are additionally quantized to q4",
        "not an official Gradium release",
    ):
        assert fragment in attribution
    assert "two flow graphs are additionally quantized to q8" in render_attribution(provenance, "q8")
    assert "fp16 graph variant" in render_attribution(provenance, "fp16")


def test_upstream_readmes_and_notice_files_are_preserved_without_flattening(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "docs").mkdir(parents=True)
    for name in ("README.md", "LICENSE.txt", "NOTICE", "COPYRIGHT", "ATTRIBUTION.md"):
        (snapshot / name).write_text(name)
    (snapshot / "docs/NOTICE-extra").write_text("nested notice")

    destination = tmp_path / "bundle/third_party/model"
    copied = preserve_upstream_notices(snapshot, destination)
    assert set(copied) == {
        "README.md", "LICENSE.txt", "NOTICE", "COPYRIGHT", "ATTRIBUTION.md", "docs/NOTICE-extra"
    }
    assert (destination / "README.md").read_text() == "README.md"
    assert (destination / "docs/NOTICE-extra").read_text() == "nested notice"
    voice_snapshot = tmp_path / "voice-snapshot"
    voice_snapshot.mkdir()
    (voice_snapshot / "README.md").write_text("voice card")
    voice_destination = tmp_path / "bundle/third_party/voices"
    assert preserve_upstream_notices(voice_snapshot, voice_destination) == ["README.md"]
    assert (voice_destination / "README.md").read_text() == "voice card"


def test_license_compliance_helper_accepts_a_complete_candidate(tmp_path: Path):
    root, manifest = _candidate(tmp_path)
    bundle = root / "bundle"
    schema = json.loads(Path("schemas/release-manifest.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    validate_license_metadata(manifest, bundle)


@pytest.mark.parametrize(
    "defect",
    ["license_uri", "source_uri", "modification", "commercial_use", "attribution_path", "attribution_asset", "legalcode", "attribution_file"],
)
def test_license_compliance_helper_rejects_missing_or_invalid_cc_fields(tmp_path: Path, defect: str):
    _, manifest = _candidate(tmp_path)
    damaged = copy.deepcopy(manifest)
    if defect == "license_uri":
        damaged["license"]["uri"] = ""
    elif defect == "source_uri":
        damaged["source"]["uri"] = "https://huggingface.co/gradium/phonon-40m-en/tree/main"
    elif defect == "modification":
        damaged["modifications"].pop("modification_notice")
    elif defect == "commercial_use":
        damaged["license"]["commercial_use"] = True
    elif defect == "attribution_path":
        damaged["license"]["attribution_path"] = "../ATTRIBUTION.md"
    elif defect == "attribution_asset":
        damaged["assets"] = [asset for asset in damaged["assets"] if asset["path"] != "ATTRIBUTION.md"]
    elif defect == "legalcode":
        damaged["license"].pop("legalcode_uri")
    elif defect == "attribution_file":
        (tmp_path / "candidate/bundle/ATTRIBUTION.md").unlink()

    with pytest.raises((ValueError, KeyError, TypeError)):
        validate_license_metadata(damaged, tmp_path / "candidate/bundle")


def test_candidate_verifier_accepts_compliant_synthetic_candidate(tmp_path: Path, monkeypatch):
    root, _ = _candidate(tmp_path)
    _stub_onnx(monkeypatch)
    verify_candidate.verify(root)

def test_read_only_release_auditor_flags_legacy_catalog_metadata(tmp_path: Path):
    from scripts.audit_cc_by_nc_releases import audit_catalog

    catalog = tmp_path / "bundles.json"
    catalog.write_text(json.dumps({"bundles": [{
        "tag": "legacy-40m", "model_key": "phonon-40m-en",
        "license": {"spdx": "CC-BY-NC-4.0", "commercial_use": False},
        "download_url": "https://example.invalid/legacy.zip",
    }]}))
    report = audit_catalog(catalog)
    assert report[0]["status"] == "review-required"
    assert any("source URI missing" in item for item in report[0]["findings"])


def test_read_only_release_auditor_accepts_compliant_local_candidate(tmp_path: Path):
    from scripts.audit_cc_by_nc_releases import audit_candidate

    root, _ = _candidate(tmp_path)
    report = audit_candidate(root)
    assert report["status"] == "pass"
    assert report["findings"] == []



def test_candidate_verifier_rejects_missing_attribution_checksum(tmp_path: Path):
    root, _ = _candidate(tmp_path)
    sums_path = root / "SHA256SUMS"
    sums_path.write_text("\n".join(
        line for line in sums_path.read_text().splitlines()
        if "bundle/ATTRIBUTION.md" not in line
    ) + "\n")

    with pytest.raises(verify_candidate.VerifyError, match="SHA256SUMS missing ATTRIBUTION.md checksum"):
        verify_candidate.verify(root)


def test_candidate_verifier_rejects_zip_without_attribution(tmp_path: Path, monkeypatch):
    root, manifest = _candidate(tmp_path)
    _stub_onnx(monkeypatch)
    archive = root / f"{manifest['tag']}.zip"
    with ZipFile(archive) as source:
        retained = [(name, source.read(name)) for name in source.namelist() if not name.endswith("/ATTRIBUTION.md")]
    with ZipFile(archive, "w") as destination:
        for name, payload in retained:
            destination.writestr(name, payload)

    with pytest.raises(verify_candidate.VerifyError, match="zip missing ATTRIBUTION.md"):
        verify_candidate.verify(root)
