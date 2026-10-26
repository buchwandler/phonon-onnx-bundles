#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, tempfile, zipfile
from pathlib import Path
from typing import Any
from phonon_onnx_bundles.catalog import model_spec
from phonon_onnx_bundles.licensing import license_metadata
from phonon_onnx_bundles.manifest import GRAPH_FILES, inventory, sha256, write_json
from scripts.export_phonon import run_export

def write_candidate_sidecars(final:Path, manifest:dict[str,Any])->None:
    """Write the manifest, checksums, recipient notes, and distributable ZIP."""
    tag=manifest["tag"]
    write_json(final/"release-manifest.json",manifest)
    sums=[f"{rec['sha256']}  bundle/{rec['path']}" for rec in manifest["assets"]]
    sums.append(f"{sha256(final/'release-manifest.json')}  release-manifest.json")
    (final/"SHA256SUMS").write_text("\n".join(sums)+"\n",encoding="utf-8")

    source=manifest["source"]
    license_info=manifest["license"]
    license_name="CC BY-NC 4.0" if license_info["spdx"]=="CC-BY-NC-4.0" else license_info["spdx"]
    notes=(
        f"# {tag}\n\n"
        f"ONNX conversion of `{source['repo_id']}` at `{source['resolved_revision']}`.\n\n"
        f"Source:\n{source['uri']}\n\n"
        f"License:\n{license_name} — {license_info['uri']}\n\n"
        "This release contains a technical ONNX conversion of the upstream model. "
        "See `ATTRIBUTION.md` inside the ZIP for retained upstream attribution, "
        "modification information, and the warranty-disclaimer reference.\n"
    )
    (final/"release-notes.md").write_text(notes,encoding="utf-8")

    zpath=final/f"{tag}.zip"
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted((final/"bundle").rglob("*")):
            if path.is_file():
                z.write(path,arcname=f"{tag}/{path.relative_to(final/'bundle').as_posix()}")
        z.write(final/"release-manifest.json",arcname=f"{tag}/release-manifest.json")
        z.write(final/"SHA256SUMS",arcname=f"{tag}/SHA256SUMS")


def build(model_key:str, precision:str, dist:Path, *, allow_noncommercial:bool=False, revision:str|None=None)->Path:
    spec=model_spec(model_key)
    with tempfile.TemporaryDirectory(prefix="phonon-candidate-") as td:
        payload=Path(td)/"payload"
        prov=run_export(model_key,precision,payload,allow_noncommercial=allow_noncommercial,revision=revision)
        source_sha=prov["source"]["resolved_revision"]
        tag=f"{model_key}-onnx-{precision}-{source_sha[:8]}"
        final=dist/tag
        if final.exists(): shutil.rmtree(final)
        final.mkdir(parents=True)
        shutil.copytree(payload,final/"bundle")
        restrictions=[] if spec["commercial_use"] else ["non-commercial"]
        license_details=license_metadata(spec["license"])
        license_info={
          "spdx":spec["license"],"uri":spec["license_uri"],
          "commercial_use":spec["commercial_use"],
          "restrictions":restrictions,"attribution_path":"ATTRIBUTION.md",
        }
        if "legalcode_uri" in license_details:
            license_info["legalcode_uri"]=spec.get("license_legalcode_uri",license_details["legalcode_uri"])
            license_info["warranty_disclaimer_section"]="Section 5"
        manifest={
          "schema_version":2,"tag":tag,"model_key":model_key,"precision":precision,
          "source":prov["source"],"exporter":prov["exporter"],"voices":prov["voices"],
          "runtime":{"kind":"phonon-streaming-onnx-v1","sample_rate":24000,"frame_samples":1920,"graphs":list(GRAPH_FILES)},
          "license":license_info,"modifications":prov["modifications"],
          "assets":inventory(final/"bundle"),
        }
        write_candidate_sidecars(final,manifest)
        return final

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("model_key"); ap.add_argument("--precision",default="fp32"); ap.add_argument("--dist",type=Path,default=Path("dist")); ap.add_argument("--revision"); ap.add_argument("--acknowledge-noncommercial-license","--allow-noncommercial",dest="allow_noncommercial",action="store_true",help="Acknowledge the noncommercial model-license condition; this does not grant commercial rights")
    a=ap.parse_args(); print(build(a.model_key,a.precision,a.dist,allow_noncommercial=a.allow_noncommercial,revision=a.revision))
if __name__=="__main__": main()
