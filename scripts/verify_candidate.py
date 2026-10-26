#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import onnx
from jsonschema import Draft202012Validator
from phonon_onnx_bundles.manifest import GRAPH_FILES, sha256
from phonon_onnx_bundles.licensing import validate_license_metadata

REQ={
 "onnx/text_conditioner.onnx":({"input_ids","duration_delta"},{"inputs_embeds","bos_embeds"}),
 "onnx/flow_lm_main.onnx":({"inputs_embeds"},{"hidden","eos_logit"}),
 "onnx/flow_lm_flow.onnx":({"hidden","noise","duration_delta"},{"latent","next_embeds"}),
 "onnx/mimi_decoder.onnx":({"latent"},{"audio"}),
}
class VerifyError(RuntimeError): pass

def verify(root:Path)->None:
    man=json.loads((root/"release-manifest.json").read_text())
    schema=json.loads(Path("schemas/release-manifest.schema.json").read_text())
    Draft202012Validator(schema).validate(man)
    for rec in man["assets"]:
        p=root/"bundle"/rec["path"]
        if not p.is_file(): raise VerifyError(f"missing {rec['path']}")
        if p.stat().st_size!=rec["bytes"] or sha256(p)!=rec["sha256"]: raise VerifyError(f"asset mismatch {rec['path']}")
    try:
        validate_license_metadata(man,root/"bundle")
    except (ValueError,KeyError,TypeError) as exc:
        raise VerifyError(str(exc)) from exc
    attribution=man["license"]["attribution_path"]
    attribution_record=next(rec for rec in man["assets"] if rec["path"]==attribution)
    sums_path=root/"SHA256SUMS"
    if not sums_path.is_file(): raise VerifyError("missing SHA256SUMS")
    expected_sum=f"{attribution_record['sha256']}  bundle/{attribution}"
    if expected_sum not in sums_path.read_text(encoding="utf-8").splitlines():
        raise VerifyError("SHA256SUMS missing ATTRIBUTION.md checksum")
    for rel in GRAPH_FILES:
        p=root/"bundle"/rel; model=onnx.load(str(p),load_external_data=True); onnx.checker.check_model(model)
        ins={x.name for x in model.graph.input}; outs={x.name for x in model.graph.output}; ri,ro=REQ[rel]
        if not ri<=ins: raise VerifyError(f"{rel}: missing inputs {sorted(ri-ins)}")
        if not ro<=outs: raise VerifyError(f"{rel}: missing outputs {sorted(ro-outs)}")
    z=root/f"{man['tag']}.zip"
    if not z.is_file(): raise VerifyError("missing zip")
    with zipfile.ZipFile(z) as f:
        bad=f.testzip()
        if bad: raise VerifyError(f"corrupt zip member {bad}")
        names=set(f.namelist()); prefix=man['tag']+'/'
        for rel in GRAPH_FILES:
            if prefix+rel not in names: raise VerifyError(f"zip missing {rel}")
        attribution=man["license"]["attribution_path"]
        if prefix+attribution not in names: raise VerifyError(f"zip missing {attribution}")
        if prefix+"SHA256SUMS" not in names: raise VerifyError("zip missing SHA256SUMS")
    print(f"verified {man['tag']}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("candidate",type=Path); a=ap.parse_args(); verify(a.candidate)
if __name__=="__main__": main()
