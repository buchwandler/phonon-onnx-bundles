from __future__ import annotations

import pytest

from scripts.build_bundle import runtime_metadata


def _config() -> dict:
    return {
        "runtime": {
            "kind": "phonon-streaming-onnx-v1",
            "sample_rate": 22050,
            "frame_samples": 1764,
            "frame_rate": 12.5,
            "d_model": 512,
            "latent_dim": 32,
            "num_layers": 6,
            "num_heads": 8,
            "head_dim": 64,
        },
        "onnx": {"mimi_state": [{"name": "cache", "shape": [1, 2], "dtype": "float32"}]},
        "generation": {
            "temperature": 0.7,
            "duration_delta": 0.25,
            "frames_after_eos": 3,
            "max_frames": 900,
        },
        "voices": [
            {
                "name": "easton",
                "display_name": "Easton",
                "voice_id": "upstream-easton",
                "locale": "en-US",
                "gender": "male",
                "frames": 125,
            }
        ],
        "default_voice": "easton",
        "languages": ["en"],
    }


def test_runtime_metadata_is_normalized_from_emitted_config(tmp_path):
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices/easton.bin").write_bytes(b"prompt")
    runtime, generation, voices, default_voice, languages = runtime_metadata(_config(), tmp_path, ["xx"])

    assert runtime["kind"] == "phonon-streaming-onnx-v1"
    assert runtime["sample_rate"] == 22050
    assert runtime["frame_samples"] == 1764
    assert runtime["frame_rate"] == 12.5
    assert runtime["d_model"] == 512
    assert runtime["mimi_state"][0]["name"] == "cache"
    assert generation == {
        "temperature_default": 0.7,
        "eos_threshold_default": None,
        "duration_delta_default": 0.25,
        "frames_after_eos": 3,
        "min_frames_before_eos": None,
        "max_frames": 900,
    }
    assert voices == [
        {
            "name": "easton",
            "display_name": "Easton",
            "voice_id": "upstream-easton",
            "locale": "en-US",
            "gender": "male",
            "frames": 125,
        }
    ]
    assert default_voice == "easton"
    assert languages == ["en"]


def test_runtime_metadata_uses_model_languages_if_config_omits_them(tmp_path):
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices/easton.bin").write_bytes(b"prompt")
    config = _config()
    config.pop("languages")
    _, _, _, _, languages = runtime_metadata(config, tmp_path, ["fr", "en"])
    assert languages == ["en", "fr"]


def test_runtime_metadata_rejects_missing_voice_prompt_and_default(tmp_path):
    (tmp_path / "voices").mkdir()
    with pytest.raises(ValueError, match="no bundled prompt"):
        runtime_metadata(_config(), tmp_path, ["en"])
    (tmp_path / "voices/easton.bin").write_bytes(b"prompt")

    config = _config()
    config["default_voice"] = "unknown"
    with pytest.raises(ValueError, match="default_voice"):
        runtime_metadata(config, tmp_path, ["en"])


def test_runtime_metadata_rejects_invalid_contract_values(tmp_path):
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices/easton.bin").write_bytes(b"prompt")
    config = _config()
    config["runtime"]["sample_rate"] = 0
    with pytest.raises(ValueError, match="sample_rate"):
        runtime_metadata(config, tmp_path, ["en"])
