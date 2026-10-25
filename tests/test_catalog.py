import json
from pathlib import Path
from phonon_onnx_bundles.catalog import stable_model_catalog

def test_catalog_is_generated_from_registry():
    assert json.loads(Path("catalog/models.json").read_text()) == stable_model_catalog()

def test_all_phonon_variants_present():
    keys={x["key"] for x in stable_model_catalog()["models"]}
    assert keys == {"phonon-31m-en","phonon-40m-en","phonon-40m-fr","phonon-40m-de","phonon-40m-es","phonon-40m-pt","phonon-40m-multilingual"}
