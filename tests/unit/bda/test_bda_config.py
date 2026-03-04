import pytest
import json

from src.bda.bda_config import (
    load_bda_config,
    validate_bda_config
)

VALID_CONFIG = {
    "decorr_factor": 0.95,
    "lambda_ref":    0.1,
    "fov":           0.01,
}


def _write_config(tmp_path, data):
    path = tmp_path / "bda_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# Test for load_bda_config

def test_loads_valid_config(tmp_path):
    path   = _write_config(tmp_path, VALID_CONFIG)
    result = load_bda_config(path)
    assert result["decorr_factor"] == VALID_CONFIG["decorr_factor"]
    assert result["lambda_ref"]    == VALID_CONFIG["lambda_ref"]
    assert result["fov"]           == VALID_CONFIG["fov"]

def test_raises_if_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="BDA config file not found"):
        load_bda_config(tmp_path / "missing.json")

def test_raises_if_json_is_invalid(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json }", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON in BDA config file"):
        load_bda_config(bad)

def test_raises_if_root_is_not_dict(tmp_path):
    path = _write_config(tmp_path, [VALID_CONFIG])
    with pytest.raises(ValueError, match="BDA config must be a JSON object"):
        load_bda_config(path)

def test_raises_if_required_field_missing(tmp_path):
    config = {k: v for k, v in VALID_CONFIG.items() if k != "decorr_factor"}
    path   = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="decorr_factor"):
        load_bda_config(path)


# Test for validate_bda_config

def test_returns_validated_config():
    result = validate_bda_config(VALID_CONFIG.copy())
    assert result["decorr_factor"] == VALID_CONFIG["decorr_factor"]
    assert result["fov"]           == VALID_CONFIG["fov"]

def test_raises_if_decorr_factor_out_of_range():
    with pytest.raises(ValueError, match="decorr_factor"):
        validate_bda_config({**VALID_CONFIG, "decorr_factor": 0.0})

def test_raises_if_lambda_ref_not_positive():
    with pytest.raises(ValueError, match="lambda_ref"):
        validate_bda_config({**VALID_CONFIG, "lambda_ref": -1.0})

def test_raises_if_fov_not_positive():
    with pytest.raises(ValueError, match="fov"):
        validate_bda_config({**VALID_CONFIG, "fov": 0.0})

def test_missing_lambda_ref_raises_key_error():
    with pytest.raises(KeyError):
        validate_bda_config({"decorr_factor": 0.95, "fov": 0.01})