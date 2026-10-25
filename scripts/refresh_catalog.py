#!/usr/bin/env python3
"""Rebuild catalog/bundles.json from immutable GitHub release manifests."""
from __future__ import annotations
import argparse, json, os, urllib.request
from pathlib import Path
from phonon_onnx_bundles.manifest import write_json

def req(url:str, token:str|None, accept:str="application/vnd.github+json"):
    headers={"Accept":accept,"X-GitHub-Api-Version":"2022-11-28","User-Agent":"phonon-onnx-bundles-catalog"}
    if token: headers["Authorization"]=f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers)) as r: return r.read()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repository",default=os.getenv("GITHUB_REPOSITORY")); ap.add_argument("--output",type=Path,default=Path("catalog/bundles.json")); a=ap.parse_args()
    if not a.repository: ap.error("--repository or GITHUB_REPOSITORY is required")
    token=os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    releases=json.loads(req(f"https://api.github.com/repos/{a.repository}/releases?per_page=100",token))
    rows=[]
    for rel in releases:
        if rel.get("draft") or rel.get("prerelease"): continue
        asset=next((x for x in rel.get("assets",[]) if x["name"]=="release-manifest.json"),None)
        if not asset: continue
        man=json.loads(req(asset["url"],token,"application/octet-stream"))
        zip_asset=next((x for x in rel.get("assets",[]) if x["name"]==f"{man['tag']}.zip"),None)
        if not zip_asset: continue
        rows.append({
          "tag":man["tag"],"model_key":man["model_key"],"precision":man["precision"],
          "source_revision":man["source"]["resolved_revision"],"license":man["license"],
          "download_url":zip_asset["browser_download_url"],"bytes":zip_asset["size"],
          "published_at":rel.get("published_at"),
        })
    rows.sort(key=lambda x:(x["model_key"],x["precision"],x["source_revision"]))
    write_json(a.output,{"schema_version":1,"bundles":rows})
    print(f"wrote {len(rows)} bundles to {a.output}")
if __name__=="__main__": main()
