#!/usr/bin/env python3
"""Bootstrap exporter for the Phonon streaming ONNX bundle contract.

The MVP deliberately keeps the large model-specific exporter out of this repo. It
pins an Apache-2.0 reference export recipe, downloads it from Hugging Face, runs
that recipe against the selected Gradium checkpoint, and then selects one stable
precision layout. This makes provenance explicit and lets a future native exporter
replace only this adapter without changing bundle/catalog contracts.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import Any
from phonon_onnx_bundles.catalog import registry, model_spec
from phonon_onnx_bundles.manifest import GRAPH_FILES, write_json
from phonon_onnx_bundles.licensing import hf_revision_uri, modification_notice, render_attribution

PRECISION_SUFFIX={"fp32":"","q8":"_quantized","q4":"_q4","fp16":"_fp16"}
FLOW_GRAPHS={"flow_lm_main","flow_lm_flow"}
ALL_GRAPHS=("text_conditioner","flow_lm_main","flow_lm_flow","mimi_decoder")

NOTICE_PREFIXES=("README", "LICENSE", "NOTICE", "COPYRIGHT", "ATTRIBUTION")
UPSTREAM_NOTICE_PATTERNS=[f"{prefix}*" for prefix in NOTICE_PREFIXES] + [f"**/{prefix}*" for prefix in NOTICE_PREFIXES]


def preserve_upstream_notices(snapshot:Path, destination:Path)->list[str]:
    """Copy README and notice-bearing upstream files without flattening paths."""
    copied=[]
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file() and p.name.upper().startswith(NOTICE_PREFIXES)):
        relative=path.relative_to(snapshot)
        target=destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target)
        copied.append(relative.as_posix())
    return copied

class ExportError(RuntimeError): pass

def _snap(repo_id:str, revision:str, token:str|None, allow_patterns:list[str]|None=None)->Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, revision=revision, token=token, allow_patterns=allow_patterns))

def _resolved_sha(repo_id:str, revision:str, token:str|None)->str:
    from huggingface_hub import HfApi

    info=HfApi(token=token).model_info(repo_id, revision=revision)
    if not info.sha: raise ExportError(f"could not resolve immutable revision for {repo_id}@{revision}")
    return info.sha

def _select_graph(raw_onnx:Path, name:str, precision:str)->Path:
    suffix=PRECISION_SUFFIX[precision]
    if precision in {"q8","q4"} and name not in FLOW_GRAPHS:
        suffix=""
    p=raw_onnx/f"{name}{suffix}.onnx"
    if not p.exists(): raise ExportError(f"export recipe did not produce {p.name}")
    return p

def run_export(model_key:str, precision:str, output:Path, *, allow_noncommercial:bool=False, revision:str|None=None)->dict[str,Any]:
    reg=registry(); spec=model_spec(model_key)
    if precision not in reg["precisions"]: raise ExportError(f"unsupported precision {precision}")
    if not spec["commercial_use"] and not allow_noncommercial:
        raise ExportError(f"{model_key} is {spec['license']}; pass --acknowledge-noncommercial-license (or legacy --allow-noncommercial) to acknowledge the non-commercial model license; acknowledgement does not grant commercial rights")
    token=os.environ.get("HF_TOKEN") or None
    if spec.get("gated") and not token:
        raise ExportError(f"{model_key} is gated; set HF_TOKEN after accepting its Hugging Face terms")
    source_rev=revision or spec["revision"]
    source_sha=_resolved_sha(spec["repo_id"], source_rev, token)
    exp=reg["exporter"]; exp_sha=_resolved_sha(exp["repo_id"], exp["revision"], token=None)
    voices=reg["voices"]; voices_sha=_resolved_sha(voices["repo_id"], voices["revision"], token=None)
    with tempfile.TemporaryDirectory(prefix="phonon-onnx-") as td:
        td=Path(td)
        model_dir=_snap(spec["repo_id"], source_sha, token, ["config.json","model.safetensors","tokenizer.json","default-voice.safetensors",*UPSTREAM_NOTICE_PATTERNS])
        recipe_dir=_snap(exp["repo_id"], exp_sha, None, ["export/*"])
        voices_dir=_snap(voices["repo_id"], voices_sha, None, ["voices.jsonl","voices/*.safetensors",*UPSTREAM_NOTICE_PATTERNS])
        recipe=recipe_dir/exp["entrypoint"]
        if not recipe.exists(): raise ExportError(f"missing pinned exporter entrypoint {recipe}")
        raw=td/"raw"
        cmd=[sys.executable,str(recipe),"--model",str(model_dir),"--voices",str(voices_dir),"--out",str(raw)]
        subprocess.run(cmd, check=True, env={**os.environ,"PYTHONHASHSEED":"0"})
        output.mkdir(parents=True, exist_ok=True); (output/"onnx").mkdir(exist_ok=True)
        for name in ALL_GRAPHS:
            shutil.copy2(_select_graph(raw/"onnx",name,precision), output/"onnx"/f"{name}.onnx")
        for name in ("config.json","tokenizer.json"):
            shutil.copy2(raw/name, output/name)
        if (raw/"voices").exists(): shutil.copytree(raw/"voices", output/"voices", dirs_exist_ok=True)
        preserve_upstream_notices(model_dir,output/"third_party"/"model")
        preserve_upstream_notices(voices_dir,output/"third_party"/"voices")
        provenance={
          "schema_version":1,
          "model_key":model_key,"precision":precision,
          "source":{
            "repo_id":spec["repo_id"],"requested_revision":source_rev,"resolved_revision":source_sha,
            "uri":hf_revision_uri(spec["repo_id"],source_sha),"license":spec["license"],
            "license_uri":spec["license_uri"],"license_legalcode_uri":spec.get("license_legalcode_uri"),
          },
          "exporter":{"repo_id":exp["repo_id"],"resolved_revision":exp_sha,"entrypoint":exp["entrypoint"],"license":exp["license"]},
          "voices":{
            "repo_id":voices["repo_id"],"resolved_revision":voices_sha,"uri":hf_revision_uri(voices["repo_id"],voices_sha),
            "license":voices["license"],"license_uri":voices["license_uri"],"scope":"en-bootstrap",
          },
          "modifications":{
            "modified":True,"modification_notice":modification_notice(precision),
            "precision":precision,"quantization":precision if precision in {"q4","q8"} else None,
          },
        }
        write_json(output/"export-provenance.json",provenance)
        (output/"ATTRIBUTION.md").write_text(render_attribution(provenance,precision),encoding="utf-8")
    for rel in GRAPH_FILES:
        if not (output/rel).exists(): raise ExportError(f"missing required graph {rel}")
    return provenance

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("model_key"); ap.add_argument("--precision",default="fp32",choices=PRECISION_SUFFIX); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--revision"); ap.add_argument("--acknowledge-noncommercial-license","--allow-noncommercial",dest="allow_noncommercial",action="store_true",help="Acknowledge the noncommercial model-license condition; this does not grant commercial rights")
    a=ap.parse_args(); p=run_export(a.model_key,a.precision,a.output,allow_noncommercial=a.allow_noncommercial,revision=a.revision); print(json.dumps(p,indent=2))
if __name__=="__main__": main()
