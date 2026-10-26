# phonon-onnx-bundles

Build, verify, release, and catalog streaming ONNX bundles for the Gradium Phonon family. This repository defines the published Phonon runtime ABI; exporter `config.json` is treated as input and normalized into a producer-owned canonical bundle config.

## Supported checkpoints and license boundary

The registry contains seven Phonon variants: `phonon-31m-en`, five monolingual 40m checkpoints (`en`, `fr`, `de`, `es`, `pt`), and `phonon-40m-multilingual`. Each bundle uses the same four-graph runtime ABI documented in [`docs/EXPORT_BACKEND.md`](docs/EXPORT_BACKEND.md).

**License boundary:** `phonon-31m-en` is Apache-2.0. The 40m checkpoints are CC BY-NC 4.0 and gated; builds require an accepted Hugging Face account token and a noncommercial-license acknowledgement. That acknowledgement does not grant commercial-use rights.

Every generated bundle carries `ATTRIBUTION.md`, preserves upstream attribution/notice files under `third_party/model/` and `third_party/voices/`, and records immutable Hugging Face source-revision URLs in its provenance and release manifest. Inspect these files in the downloaded ZIP rather than relying on a release page. See [`MODEL_LICENSES.md`](MODEL_LICENSES.md) and [`docs/ATTRIBUTION_REMEDIATION.md`](docs/ATTRIBUTION_REMEDIATION.md).

## Canonical bundle config

The published `bundle/config.json` has a stable Phonon-owned shape containing `schema_version`, `runtime`, `generation`, `voices`, `default_voice`, and `languages`. Runtime dimensions, state, generation defaults, voice inventory, and languages are derived from exporter output; family constants are not inferred. If the exporter omits a runtime kind, the builder injects `phonon-streaming-onnx-v1`; an explicit conflicting kind fails validation. The four graph declarations and all other runtime, voice, generation, and license checks remain strict.

## Release versions and immutable tags

Every candidate requires a SemVer-like `--release-version` without a leading `v`, for example `0.2.0` or `1.0.0-rc.1`. New release manifests use schema 3 and include this release-set identity. Tags have the form:

```text
<model_key>-onnx-<precision>-v<release_version>-<source_sha8>-files-v1
```

A GitHub Release tag is immutable: an existing tag is never replaced. Publish a new `release_version` for a new release set. Historical release sets remain available on GitHub, while `catalog/bundles.json` contains exactly one active release version and at most one row per model/precision. The complete registry cross-product is currently 7 models × 4 precisions = 28 variants.

## Build locally

Install dependencies and build a candidate; the build prints the candidate directory to verify:

```bash
python -m pip install -e '.[build,test]'
set -o pipefail
candidate="$(python scripts/build_bundle.py phonon-31m-en --precision q4 --release-version 0.2.0 | tail -n 1)"
python scripts/verify_candidate.py "$candidate"
```

For a gated 40m checkpoint, use an account that has accepted the upstream model terms and acknowledge the noncommercial condition:

```bash
export HF_TOKEN=hf_...  # account must have accepted the model terms
set -o pipefail
candidate="$(python scripts/build_bundle.py phonon-40m-fr --precision q4 \
  --release-version 0.2.0 --acknowledge-noncommercial-license | tail -n 1)"
python scripts/verify_candidate.py "$candidate"
```

The build resolves model and voice revisions to immutable commits. `scripts/resolve_release_plan.py` freezes those revisions once for a full set; model, voice, and exporter revisions are recorded in export provenance. The Apache-2.0 exporter recipe remains pinned in `registry/models.json`.

## GitHub Actions

- **checks** runs tests, schema/catalog integrity, and a deterministic source-catalog rebuild without downloading model weights.
- **release-all-phonon-bundles** is the canonical manual full release. Supply `release_version`, `allow_noncommercial=true`, and `HF_TOKEN`; one dispatch builds every registry model at every registry precision, resolves upstream revisions once, and can publish all 28 immutable GitHub Releases. The catalog is refreshed and committed once, only after all candidates and releases are complete. `publish=false` builds and verifies without publishing.
- **build-bundle** is for diagnostic/ad-hoc single-model builds. `release_version` is required. Publishing is optional; catalog activation defaults off and requires an explicit `activate_catalog=true` plus a complete 28-variant set.
- **refresh-catalog** (scheduled or manual) refreshes only the release version already active in `catalog/bundles.json`; it does not switch the catalog to the newest GitHub release.

A full release always includes CC BY-NC 4.0 models, so its acknowledgement and `HF_TOKEN` gates apply even when the selected models could otherwise be built individually. The acknowledgement does not change the license terms or authorize commercial use.

## Catalogs

- `catalog/models.json` is generated deterministically from `registry/models.json` and describes buildable source models.
- `catalog/bundles.json` is schema v2. It records the active `release_version` at the top level and on every row, and lists only that release set's published individual-file bundles with paths, direct URLs, checksums, runtime/generation defaults, bundled voices, and license metadata. Historical versions are ignored during active catalog generation; legacy ZIP-only releases remain manual downloads and are omitted. The checked-in empty catalog uses a null `release_version` until a full set is activated.

Run `python scripts/check_catalog_integrity.py` before committing catalog changes. Canonical full-release refresh uses `--require-complete` to fail with a readable missing/extra variant report instead of activating a partial set.
