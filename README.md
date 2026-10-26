# phonon-onnx-bundles (MVP)

Build, verify, release, and catalog streaming ONNX bundles for the Gradium Phonon family.

## Covered checkpoints

The registry contains all current Phonon variants: `phonon-31m-en`, the five monolingual 40m checkpoints (`en`, `fr`, `de`, `es`, `pt`), and `phonon-40m-multilingual`. Every bundle uses the same four-graph runtime ABI documented in `docs/EXPORT_BACKEND.md`.

**License boundary:** `phonon-31m-en` is Apache-2.0. The 40m checkpoints are CC BY-NC 4.0 and gated; builds require an accepted Hugging Face account token and a noncommercial-license acknowledgement. That acknowledgement does not grant commercial-use rights.


## Third-party attribution

The repository's source-code license is separate from each upstream model and voice-asset license. Every generated bundle carries `ATTRIBUTION.md`, preserves upstream attribution/notice files under `third_party/model/` and `third_party/voices/`, and records immutable Hugging Face source-revision URLs in its provenance and release manifest. Inspect these files in the downloaded ZIP rather than relying on a release page.

For CC BY-NC 4.0 checkpoints, `--acknowledge-noncommercial-license` (the legacy `--allow-noncommercial` alias remains supported) records acknowledgement only; it does not grant commercial rights or determine whether a use is NonCommercial. See [`MODEL_LICENSES.md`](MODEL_LICENSES.md) and [`docs/ATTRIBUTION_REMEDIATION.md`](docs/ATTRIBUTION_REMEDIATION.md).

## Build locally

```bash
python -m pip install -e '.[build,test]'
python scripts/build_bundle.py phonon-31m-en --precision q4
python scripts/verify_candidate.py "$(find dist -mindepth 1 -maxdepth 1 -type d -print -quit)"
```

For a gated 40m checkpoint:

```bash
export HF_TOKEN=hf_...   # account must have accepted the model terms
python scripts/build_bundle.py phonon-40m-fr --precision q4 --acknowledge-noncommercial-license
```

The build resolves `main` to an immutable source commit before export. The release tag includes the source SHA prefix, and `export-provenance.json` records the full model, exporter, and voice revisions.

## GitHub Actions

- **checks**: tests, schema/catalog integrity, deterministic source-catalog rebuild; no model download.
- **build-bundle**: manual single-model export, ONNX verification, artifact upload, optional immutable GitHub Release and catalog update.
- **build-all-models**: manual matrix over all seven model variants.
- **refresh-catalog**: scheduled/manual deterministic rebuild of the source catalog plus discovery of published release manifests.

Add `HF_TOKEN` as a repository Actions secret before building 40m checkpoints. For public 31m builds it can be absent.

## Catalogs

- `catalog/models.json` is generated deterministically from `registry/models.json` and describes buildable source models.
- `catalog/bundles.json` is schema v2 and lists only actually published individual-file releases with bundle-relative paths, direct URLs, checksums, runtime/generation defaults, bundled voices, and license metadata; legacy ZIP-only releases remain manual downloads and are omitted.

Run `python scripts/check_catalog_integrity.py` before committing catalog changes.

## Export provenance

The MVP intentionally does not vendor the large ONNX converter. It pins the Apache-2.0 export recipe from `ai-ecoverse/phonon-31m-en-ONNX` and executes that revision against each selected Gradium checkpoint. This keeps the first version small and auditable while preserving a clean path to a native exporter later.
