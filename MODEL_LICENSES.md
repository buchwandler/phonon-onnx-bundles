# Model licenses

This repository's source code is Apache-2.0. Model checkpoints and generated bundles have separate license terms. Generated bundles redistribute upstream model material in a technically converted ONNX format and retain the upstream checkpoint license recorded in each `release-manifest.json`; this wording does not make a legal determination about whether a conversion is Adapted Material.

| Model family | Upstream license | Commercial use in this project |
| --- | --- | --- |
| `phonon-31m-en` | Apache-2.0 | allowed by the upstream license |
| `phonon-40m-*` | CC-BY-NC-4.0 | **not allowed for commercial use** |

Canonical license references:

- CC BY-NC 4.0: <https://creativecommons.org/licenses/by-nc/4.0/>
- CC BY-NC 4.0 legal code: <https://creativecommons.org/licenses/by-nc/4.0/legalcode.en>
- Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0>

Each bundle contains `ATTRIBUTION.md`, immutable source-revision URLs, and preserved upstream model/voice README and notice files under `third_party/`. The 40m models are gated on Hugging Face; a build requires `HF_TOKEN` for an account that accepted the upstream terms.


The separately fetched voice catalog is `gradium/phonon-voices` and is Apache-2.0 (<https://www.apache.org/licenses/LICENSE-2.0>); its immutable source revision and preserved notices are recorded separately from the 40m checkpoint license.

`--allow-noncommercial` (also available as `--acknowledge-noncommercial-license`) is only an acknowledgement that the model license is noncommercial. It does **not** grant commercial-use rights or make a commercial build/distribution noncommercial. The purpose and context of a use must independently comply with CC BY-NC 4.0.

A licensor removal request or discovered defect may require remediation notwithstanding ordinary release immutability. Follow [`docs/ATTRIBUTION_REMEDIATION.md`](docs/ATTRIBUTION_REMEDIATION.md); the audit command reports affected catalog entries and checks local unpacked candidates without modifying releases.
