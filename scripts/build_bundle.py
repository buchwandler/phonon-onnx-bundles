#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, tempfile, zipfile
from pathlib import Path
from phonon_onnx_bundles.catalog import model_spec
from phonon_onnx_bundles.manifest import GRAPH_FILES, inventory, sha256, write_json
from scripts.export_phonon import run_export

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
        manifest={
          "schema_version":1,"tag":tag,"model_key":model_key,"precision":precision,
          "source":prov["source"],"exporter":prov["exporter"],"voices":prov["voices"],
          "runtime":{"kind":"phonon-streaming-onnx-v1","sample_rate":24000,"frame_samples":1920,"graphs":list(GRAPH_FILES)},
          "license":{"spdx":spec["license"],"commercial_use":spec["commercial_use"],"restrictions":restrictions},
          "assets":inventory(final/"bundle"),
        }
        write_json(final/"release-manifest.json",manifest)
        sums=[]
        for rec in manifest["assets"]:
            sums.append(f"{rec['sha256']}  bundle/{rec['path']}")
        sums.append(f"{sha256(final/'release-manifest.json')}  release-manifest.json")
        (final/"SHA256SUMS").write_text("\n".join(sums)+"\n", encoding="utf-8")
        notes=(
            f"# {tag}\n\n"
            f"ONNX bundle for `{spec['repo_id']}` at `{source_sha}`.\n\n"
            f"Model license: **{spec['license']}**.\n"
        )
        (final/"release-notes.md").write_text(notes, encoding="utf-8")
        zpath=final/f"{tag}.zip"
        with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted((final/"bundle").rglob("*")):
                if p.is_file(): z.write(p,arcname=f"{tag}/{p.relative_to(final/'bundle').as_posix()}")
            z.write(final/"release-manifest.json",arcname=f"{tag}/release-manifest.json")
            z.write(final/"SHA256SUMS",arcname=f"{tag}/SHA256SUMS")
        return final

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("model_key"); ap.add_argument("--precision",default="fp32"); ap.add_argument("--dist",type=Path,default=Path("dist")); ap.add_argument("--revision"); ap.add_argument("--allow-noncommercial",action="store_true")
    a=ap.parse_args(); print(build(a.model_key,a.precision,a.dist,allow_noncommercial=a.allow_noncommercial,revision=a.revision))
if __name__=="__main__": main()
