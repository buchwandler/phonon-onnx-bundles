"""Shared license and attribution metadata for exported model bundles."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import quote


LICENSES: dict[str, dict[str, object]] = {
    "CC-BY-NC-4.0": {
        "uri": "https://creativecommons.org/licenses/by-nc/4.0/",
        "legalcode_uri": "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en",
        "commercial_use": False,
    },
    "Apache-2.0": {
        "uri": "https://www.apache.org/licenses/LICENSE-2.0",
        "commercial_use": True,
    },
}


def license_metadata(spdx: str) -> dict[str, object]:
    """Return canonical metadata for a supported license identifier."""
    try:
        return dict(LICENSES[spdx])
    except KeyError as exc:
        raise ValueError(f"unsupported license identifier: {spdx!r}") from exc


def hf_revision_uri(repo_id: str, revision: str) -> str:
    """Build a URL pinned to an immutable Hugging Face revision."""
    if not repo_id.strip() or not revision.strip():
        raise ValueError("repo_id and resolved revision must be non-empty")
    return f"https://huggingface.co/{quote(repo_id, safe='/')}/tree/{quote(revision, safe='')}"


def modification_notice(precision: str) -> str:
    """Describe the transformations performed by the bundle exporter."""
    if precision in {"q4", "q8"}:
        return (
            "Converted to the phonon-streaming-onnx-v1 four-graph ONNX runtime format. "
            f"The two flow graphs are additionally quantized to {precision}; the text "
            "conditioner and Mimi decoder remain fp32."
        )
    if precision == "fp16":
        return (
            "Converted to the phonon-streaming-onnx-v1 four-graph ONNX runtime format "
            "using the fp16 graph variant; no additional q4/q8 flow quantization is applied."
        )
    if precision == "fp32":
        return (
            "Converted to the phonon-streaming-onnx-v1 four-graph ONNX runtime format "
            "without intentional numeric weight quantization."
        )
    raise ValueError(f"unsupported precision: {precision!r}")


def render_attribution(provenance: dict[str, object], precision: str) -> str:
    """Render recipient-facing attribution from immutable export provenance."""
    source = provenance["source"]
    voices = provenance["voices"]
    if not isinstance(source, dict) or not isinstance(voices, dict):
        raise ValueError("provenance must contain source and voices objects")
    source_uri = source.get("uri") or hf_revision_uri(
        str(source["repo_id"]), str(source["resolved_revision"])
    )
    voices_uri = voices.get("uri") or hf_revision_uri(
        str(voices["repo_id"]), str(voices["resolved_revision"])
    )
    model_license = str(source["license"])
    metadata = license_metadata(model_license)
    legalcode_uri = str(metadata.get("legalcode_uri", ""))
    if source.get("license_uri"):
        metadata["uri"] = source["license_uri"]
    license_uri = str(metadata["uri"])
    modification = modification_notice(precision)

    warranty = ""
    if model_license == "CC-BY-NC-4.0":
        warranty = (
            "The CC BY-NC 4.0 legal code includes its disclaimer of warranties and "
            f"limitation of liability in Section 5: {legalcode_uri}.\n\n"
        )

    return (
        "# Attribution and license information\n\n"
        "This bundle contains upstream model material that has been technically "
        "converted for ONNX runtime use.\n\n"
        "## Upstream model\n\n"
        f"- Source repository: `{source['repo_id']}`\n"
        f"- Exact source revision: `{source['resolved_revision']}`\n"
        f"- Exact source URI: {source_uri}\n"
        "- Upstream publisher: Gradium\n"
        "- Upstream attribution/model-card information: `third_party/model/README.md`\n"
        f"- License: {model_license}\n"
        f"- License URI: {license_uri}\n"
        + (f"- Legal code: {legalcode_uri}\n" if legalcode_uri else "")
        + "\n"
        + warranty
        + "## Modifications made by phonon-onnx-bundles\n\n"
        + modification
        + "\n\n"
        + "This conversion project is independent and is not an official Gradium release. "
        "No individual creator is inferred here; retain the upstream model card above "
        "for publisher-supplied attribution information.\n\n"
        "## Voice assets\n\n"
        f"- Source repository: `{voices['repo_id']}`\n"
        f"- Exact source revision: `{voices['resolved_revision']}`\n"
        f"- Exact source URI: {voices_uri}\n"
        f"- License: {voices['license']}\n"
        "- Preserved upstream notices: `third_party/voices/`\n"
    )



def validate_license_metadata(manifest: dict[str, object], bundle_root: Path) -> None:
    """Fail closed unless manifest and bundled attribution agree on provenance."""
    source = manifest.get("source")
    license_info = manifest.get("license")
    modifications = manifest.get("modifications")
    assets = manifest.get("assets")
    if not isinstance(source, dict) or not isinstance(license_info, dict):
        raise ValueError("manifest must contain source and license objects")
    if not isinstance(modifications, dict) or not isinstance(assets, list):
        raise ValueError("manifest must contain modifications and assets")

    spdx = str(license_info.get("spdx", ""))
    expected = license_metadata(spdx)
    expected_uri = str(expected["uri"])
    if license_info.get("uri") != expected_uri:
        raise ValueError("license metadata is missing the canonical license URI")
    if source.get("license") != spdx or source.get("license_uri") != expected_uri:
        raise ValueError("source license metadata does not match the manifest license")
    if license_info.get("commercial_use") is not expected["commercial_use"]:
        raise ValueError(f"{spdx} commercial_use does not match its canonical license metadata")

    repo_id = str(source.get("repo_id", ""))
    revision = str(source.get("resolved_revision", ""))
    expected_source_uri = hf_revision_uri(repo_id, revision)
    if source.get("uri") != expected_source_uri:
        raise ValueError("source.uri must point to the exact immutable upstream revision")

    precision = str(manifest.get("precision", ""))
    if modifications.get("modified") is not True:
        raise ValueError("modifications.modified must be true")
    expected_notice = modification_notice(precision)
    if modifications.get("precision") != precision or modifications.get("modification_notice") != expected_notice:
        raise ValueError("modification metadata does not describe the selected precision")
    expected_quantization = precision if precision in {"q4", "q8"} else None
    if modifications.get("quantization") != expected_quantization:
        raise ValueError("modification quantization does not match the selected precision")

    attribution_rel = license_info.get("attribution_path")
    if attribution_rel != "ATTRIBUTION.md":
        raise ValueError("license.attribution_path must identify bundled ATTRIBUTION.md")
    rel_path = PurePosixPath(str(attribution_rel))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError("attribution_path must stay inside the bundle")
    if not any(isinstance(asset, dict) and asset.get("path") == attribution_rel for asset in assets):
        raise ValueError("ATTRIBUTION.md is missing from the release asset inventory")
    attribution_path = bundle_root.joinpath(*rel_path.parts)
    if not attribution_path.is_file():
        raise ValueError("bundled ATTRIBUTION.md is missing")
    text = attribution_path.read_text(encoding="utf-8")
    required = [expected_uri, expected_source_uri, "ONNX", expected_notice]

    if spdx == "CC-BY-NC-4.0":
        legalcode_uri = str(expected["legalcode_uri"])
        if license_info.get("legalcode_uri") != legalcode_uri:
            raise ValueError("CC BY-NC 4.0 legal-code URI is missing or incorrect")
        if source.get("license_legalcode_uri") != legalcode_uri:
            raise ValueError("source CC BY-NC 4.0 legal-code URI is missing or incorrect")
        if license_info.get("warranty_disclaimer_section") != "Section 5":
            raise ValueError("CC BY-NC 4.0 warranty disclaimer must refer to Section 5")
        if license_info.get("commercial_use") is not False:
            raise ValueError("CC-BY-NC-4.0 artifact must be marked non-commercial")
        if "non-commercial" not in license_info.get("restrictions", []):
            raise ValueError("CC-BY-NC-4.0 artifact must include the non-commercial restriction")
        required.extend(["CC BY-NC 4.0", legalcode_uri, "Section 5", "not an official Gradium release"])
        for retained in ("third_party/model/README.md", "third_party/voices/README.md"):
            if not (bundle_root / retained).is_file():
                raise ValueError(f"required upstream attribution file is missing: {retained}")

    for fragment in required:
        if fragment not in text:
            raise ValueError(f"ATTRIBUTION.md is missing required information: {fragment!r}")
