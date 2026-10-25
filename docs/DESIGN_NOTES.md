# MVP design notes

This repository deliberately separates three concerns:

1. **Source registry** — `registry/models.json` is the human-maintained list of buildable Gradium checkpoints, their licenses, gating status, and supported precisions.
2. **Build/release** — a selected checkpoint is resolved to an immutable Hugging Face commit, exported into the stable four-graph Phonon ONNX ABI, verified, hashed, zipped, and optionally published as an immutable GitHub Release.
3. **Distribution catalog** — `catalog/models.json` is deterministic from the registry; `catalog/bundles.json` is deterministic from published release manifests.

The split is intentional: source model discovery can change without mutating already-published bundle metadata.

## MVP exporter boundary

The first version executes a pinned public Apache-2.0 export recipe instead of copying it into this repository. This keeps the project small, preserves provenance, and makes the native-exporter follow-up narrow: replace `scripts/export_phonon.py` while keeping graph names, verification, manifest, release, and catalog code unchanged.

The pinned recipe is already proven on the public 31m checkpoint. The 40m models advertise the same runtime/file layout upstream, but are gated and CC-BY-NC-4.0; their first successful CI artifacts should be treated as the acceptance run before enabling automated publication for those entries.

## Voice scope

The bootstrap exporter currently packages the English subset of the public Phonon voice catalog for every model because that is the pinned recipe's tested path. This does not change the language capability of the acoustic checkpoint. A follow-up can make voice selection locale-aware or publish voices as a separate catalog without changing the ONNX graph ABI.
