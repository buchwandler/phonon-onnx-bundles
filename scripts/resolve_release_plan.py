#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from phonon_onnx_bundles.catalog import registry
from phonon_onnx_bundles.manifest import write_json
from phonon_onnx_bundles.release_version import validate_release_version
from phonon_onnx_bundles.revisions import is_commit_sha


def _new_api(token: str | None) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token)


def _resolve_sha(api: Any, repo_id: str, revision: str) -> str:
    info = api.model_info(repo_id, revision=revision)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not is_commit_sha(sha):
        raise ValueError(f"could not resolve immutable commit SHA for {repo_id}@{revision}")
    return sha


def resolve_release_plan(
    release_version: str,
    token: str | None,
    *,
    api_factory: Callable[[str | None], Any] | None = None,
) -> dict[str, Any]:
    """Resolve every mutable upstream ref once and produce an immutable release plan."""
    release_version = validate_release_version(release_version)
    data = registry()
    if any(model.get("gated") for model in data["models"].values()) and not token:
        raise ValueError("HF_TOKEN is required to resolve the full registry release plan")

    exporter_revision = data["exporter"]["revision"]
    if not is_commit_sha(exporter_revision):
        raise ValueError("registry exporter revision must be an immutable 40-character commit SHA")

    api = (api_factory or _new_api)(token)
    voices = data["voices"]
    voices_revision = _resolve_sha(api, voices["repo_id"], voices["revision"])
    model_revisions = {
        model_key: _resolve_sha(api, spec["repo_id"], spec["revision"])
        for model_key, spec in sorted(data["models"].items())
    }
    plan = {
        "release_version": release_version,
        "exporter_revision": exporter_revision,
        "voices_revision": voices_revision,
        "models": model_revisions,
        "precisions": list(data["precisions"]),
    }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve immutable upstream revisions for a release set")
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or None
    plan = resolve_release_plan(args.release_version, token)
    if args.output:
        write_json(args.output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
