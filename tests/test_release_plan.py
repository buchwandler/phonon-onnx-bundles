from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from phonon_onnx_bundles.catalog import registry
from phonon_onnx_bundles.revisions import is_commit_sha
from scripts import export_phonon
from scripts.resolve_release_plan import resolve_release_plan


class FakeHfApi:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def model_info(self, repo_id: str, *, revision: str):
        self.calls.append((repo_id, revision))
        sha = f"{len(self.calls):040x}"
        return SimpleNamespace(sha=sha)


def test_release_plan_resolves_each_mutable_ref_once_and_derives_28_variants():
    data = registry()
    api = FakeHfApi()
    passed_tokens: list[str | None] = []

    def factory(token: str | None):
        passed_tokens.append(token)
        return api

    plan = resolve_release_plan("0.2.0", "test-token", api_factory=factory)

    assert passed_tokens == ["test-token"]
    assert plan["release_version"] == "0.2.0"
    assert plan["exporter_revision"] == data["exporter"]["revision"]
    assert is_commit_sha(plan["exporter_revision"])
    assert is_commit_sha(plan["voices_revision"])
    assert set(plan["models"]) == set(data["models"])
    assert all(is_commit_sha(revision) for revision in plan["models"].values())
    assert plan["precisions"] == data["precisions"]
    assert len(plan["models"]) == 7
    assert len(plan["precisions"]) == 4
    assert len(plan["models"]) * len(plan["precisions"]) == 28
    assert len(api.calls) == len(data["models"]) + 1
    assert len(api.calls) == len(set(api.calls))


def test_release_plan_requires_hf_token_for_gated_registry_models():
    called = False

    def factory(_token: str | None):
        nonlocal called
        called = True
        raise AssertionError("API must not be contacted without the required token")

    with pytest.raises(ValueError, match="HF_TOKEN is required"):
        resolve_release_plan("0.2.0", None, api_factory=factory)
    assert not called


def test_release_plan_validates_version_before_resolving_revisions():
    called = False

    def factory(_token: str | None):
        nonlocal called
        called = True
        return FakeHfApi()

    with pytest.raises(ValueError, match="release_version"):
        resolve_release_plan("v0.2.0", "test-token", api_factory=factory)
    assert not called


def test_pinned_export_revision_skips_mutable_ref_resolution(monkeypatch):
    sha = "0123456789abcdef0123456789abcdef01234567"

    def unexpected_resolution(*_args, **_kwargs):
        raise AssertionError("an immutable SHA must not be resolved again")

    monkeypatch.setattr(export_phonon, "_resolved_sha", unexpected_resolution)
    assert export_phonon._resolve_or_use_sha("org/model", sha, None) == sha
    assert is_commit_sha(sha)



def test_run_export_uses_and_records_release_plan_pins(tmp_path, monkeypatch):
    data = registry()
    model_key = "phonon-31m-en"
    model_sha = "1" * 40
    voices_sha = "2" * 40
    exporter_sha = data["exporter"]["revision"]
    model_dir = tmp_path / "model"
    recipe_dir = tmp_path / "recipe"
    voices_dir = tmp_path / "voices"
    (recipe_dir / "export").mkdir(parents=True)
    (recipe_dir / data["exporter"]["entrypoint"]).write_text("# pinned recipe")
    for directory in (model_dir, voices_dir):
        directory.mkdir()
    revisions: dict[str, str] = {}

    def snapshot(repo_id: str, revision: str, token: str | None, allow_patterns=None):
        revisions[repo_id] = revision
        return {
            data["models"][model_key]["repo_id"]: model_dir,
            data["exporter"]["repo_id"]: recipe_dir,
            data["voices"]["repo_id"]: voices_dir,
        }[repo_id]

    def run_recipe(command, *, check, env):
        raw = Path(command[command.index("--out") + 1])
        (raw / "onnx").mkdir(parents=True)
        (raw / "voices").mkdir()
        for name in export_phonon.ALL_GRAPHS:
            (raw / "onnx" / f"{name}.onnx").write_bytes(b"graph")
        (raw / "config.json").write_text("{}")
        (raw / "tokenizer.json").write_text("{}")

    monkeypatch.setattr(export_phonon, "_snap", snapshot)
    monkeypatch.setattr(export_phonon.subprocess, "run", run_recipe)
    monkeypatch.setattr(export_phonon, "preserve_upstream_notices", lambda *_args: [])
    monkeypatch.setattr(export_phonon, "_repair_graph_order", lambda _path: None)
    provenance = export_phonon.run_export(
        model_key,
        "fp32",
        tmp_path / "payload",
        revision=model_sha,
        voices_revision=voices_sha,
        exporter_revision=exporter_sha,
    )

    assert revisions == {
        data["models"][model_key]["repo_id"]: model_sha,
        data["exporter"]["repo_id"]: exporter_sha,
        data["voices"]["repo_id"]: voices_sha,
    }
    assert provenance["source"]["resolved_revision"] == model_sha
    assert provenance["exporter"]["resolved_revision"] == exporter_sha
    assert provenance["voices"]["resolved_revision"] == voices_sha
