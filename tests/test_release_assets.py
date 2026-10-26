from __future__ import annotations

import pytest

from phonon_onnx_bundles.release_assets import (
    bundle_path_from_release_asset,
    flattened_release_asset_name,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("config.json", "bundle--config.2Ejson"),
        ("tokenizer.json", "bundle--tokenizer.2Ejson"),
        ("onnx/text_conditioner.onnx", "bundle--onnx--text_conditioner.2Eonnx"),
        ("voices/easton.bin", "bundle--voices--easton.2Ebin"),
        ("bundle/voice-name--x.bin", "bundle--bundle--voice.2Dname.2D.2Dx.2Ebin"),
        ("third_party/model/NOTICE~.txt", "bundle--third_party--model--NOTICE.7E.2Etxt"),
    ],
)
def test_flat_asset_name_is_deterministic_and_reversible(path: str, expected: str):
    name = flattened_release_asset_name(path)
    assert name == expected
    assert bundle_path_from_release_asset(name) == path


def test_distinct_paths_cannot_collide():
    paths = [
        "a/b-c",
        "a-b/c",
        "a/b--c",
        "a--b/c",
        "a/~2D",
    ]
    names = [flattened_release_asset_name(path) for path in paths]
    assert len(names) == len(set(names))
    assert [bundle_path_from_release_asset(name) for name in names] == paths


@pytest.mark.parametrize("path", ["", "/absolute", "../escape", "a/../b", "a//b", r"a\\b", "a/."])
def test_unsafe_bundle_paths_are_rejected(path: str):
    with pytest.raises(ValueError):
        flattened_release_asset_name(path)


@pytest.mark.parametrize("name", ["other--config.json", "bundle--", "bundle--a~~", "bundle--a--b--"])
def test_invalid_flat_asset_names_are_rejected(name: str):
    with pytest.raises(ValueError):
        bundle_path_from_release_asset(name)
