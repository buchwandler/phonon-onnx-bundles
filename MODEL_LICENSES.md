# Model licenses

This repository's source code is Apache-2.0. Generated bundles are derivative model artifacts and follow the upstream checkpoint license recorded in each `release-manifest.json`.

| Model family | Upstream license | Commercial use in this project |
| --- | --- | --- |
| `phonon-31m-en` | Apache-2.0 | allowed by the upstream license |
| `phonon-40m-*` | CC-BY-NC-4.0 | **not allowed for commercial use** |

The 40m models are gated on Hugging Face. A build requires `HF_TOKEN` for an account that has accepted the upstream terms. CI additionally requires the explicit `allow_noncommercial` acknowledgement.

Voice assets are sourced from `gradium/phonon-voices` and their source/license revision is recorded in provenance.
