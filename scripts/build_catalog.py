#!/usr/bin/env python3
from pathlib import Path
from phonon_onnx_bundles.catalog import stable_model_catalog
from phonon_onnx_bundles.manifest import write_json

if __name__ == "__main__":
    write_json(Path("catalog/models.json"), stable_model_catalog())
