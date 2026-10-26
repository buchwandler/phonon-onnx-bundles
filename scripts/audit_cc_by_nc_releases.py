#!/usr/bin/env python3
"""Read-only audit of published catalog entries and local CC BY-NC candidates."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from phonon_onnx_bundles.licensing import validate_license_metadata
from phonon_onnx_bundles.manifest import sha256


def audit_catalog(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    reports = []
    for row in data.get("bundles", []):
        if not str(row.get("model_key", "")).startswith("phonon-40m-"):
            continue
        license_info = row.get("license", {})
        if not isinstance(license_info, dict):
            license_info = {}
        findings = []
        if license_info.get("uri") != "https://creativecommons.org/licenses/by-nc/4.0/":
            findings.append("catalog license URI missing or noncanonical")
        if license_info.get("attribution_path") != "ATTRIBUTION.md":
            findings.append("catalog attribution path missing")
        if row.get("source_uri") is None:
            findings.append("catalog immutable source URI missing")
        if license_info.get("commercial_use") is not False:
            findings.append("catalog does not mark commercial_use false")
        findings.append("downloaded release ZIP still requires direct inspection")
        reports.append({
            "tag": row.get("tag"),
            "model_key": row.get("model_key"),
            "download_url": row.get("download_url"),
            "status": "review-required",
            "findings": findings,
        })
    return reports


def audit_candidate(path: Path) -> dict[str, Any]:
    manifest_path = path / "release-manifest.json"
    report: dict[str, Any] = {"path": str(path), "status": "fail", "findings": []}
    if not manifest_path.is_file():
        report["findings"].append("release-manifest.json missing")
        return report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["findings"].append(f"cannot read release manifest: {exc}")
        return report
    if not str(manifest.get("model_key", "")).startswith("phonon-40m-"):
        report["status"] = "not-applicable"
        return report

    report["tag"] = manifest.get("tag")
    bundle = path / "bundle"
    findings: list[str] = report["findings"]
    for asset in manifest.get("assets", []):
        asset_path = bundle / str(asset.get("path", ""))
        if not asset_path.is_file():
            findings.append(f"asset missing: {asset.get('path')}")
        elif asset_path.stat().st_size != asset.get("bytes") or sha256(asset_path) != asset.get("sha256"):
            findings.append(f"asset checksum/size mismatch: {asset.get('path')}")
    try:
        validate_license_metadata(manifest, bundle)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        findings.append(f"license/attribution verification failed: {exc}")
    license_info = manifest.get("license", {})
    if isinstance(license_info, dict):
        attribution = license_info.get("attribution_path", "ATTRIBUTION.md")
        record = next((asset for asset in manifest.get("assets", []) if asset.get("path") == attribution), None)
        sums_path = path / "SHA256SUMS"
        if record is None or not sums_path.is_file():
            findings.append("SHA256SUMS or ATTRIBUTION.md inventory record missing")
        elif f"{record.get('sha256')}  bundle/{attribution}" not in sums_path.read_text(encoding="utf-8").splitlines():
            findings.append("SHA256SUMS missing ATTRIBUTION.md checksum")

    archive = path / f"{manifest.get('tag', '')}.zip"
    if not archive.is_file():
        sibling_archive = path.parent / f"{manifest.get('tag', '')}.zip"
        if sibling_archive.is_file():
            archive = sibling_archive
    if not archive.is_file():
        findings.append("release ZIP missing")
    else:
        try:
            with zipfile.ZipFile(archive) as zf:
                bad_member = zf.testzip()
                if bad_member:
                    findings.append(f"corrupt ZIP member: {bad_member}")
                attribution = manifest.get("license", {}).get("attribution_path", "ATTRIBUTION.md")
                if f"{manifest.get('tag')}/{attribution}" not in zf.namelist():
                    findings.append("release ZIP is missing ATTRIBUTION.md")
                if f"{manifest.get('tag')}/SHA256SUMS" not in zf.namelist():
                    findings.append("release ZIP is missing SHA256SUMS")
        except (OSError, zipfile.BadZipFile) as exc:
            findings.append(f"cannot inspect release ZIP: {exc}")

    report["status"] = "pass" if not findings else "fail"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="*", type=Path, help="local unpacked release candidate directories")
    parser.add_argument("--catalog", type=Path, default=Path("catalog/bundles.json"))
    args = parser.parse_args()

    result = {"catalog": audit_catalog(args.catalog), "candidates": [audit_candidate(p) for p in args.candidates]}
    print(json.dumps(result, indent=2, sort_keys=True))
    catalog_blocked = any(row["findings"][:-1] for row in result["catalog"])
    candidate_blocked = any(row["status"] == "fail" for row in result["candidates"])
    return 1 if catalog_blocked or candidate_blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
