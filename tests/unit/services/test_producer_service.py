import pytest
import json
import numpy as np
import astropy.units as u
from unittest.mock import Mock, patch

from services.producer_service import (
    load_simulation_config,
    update_bda_config,
    update_grid_config,
    stream_kafka,
    run_producer,
)

MODULE = "services.producer_service"
DEFAULT_TOPIC = "visibility-stream"

# ==============================================================================
# Tests for load_simulation_config
# ==============================================================================

def test_load_simulation_config_raises_on_none():
    """Test that None config_path raises ValueError"""
    with pytest.raises(ValueError, match="config_path cannot be None"):
        load_simulation_config(None)


def test_load_simulation_config_raises_on_nonexistent_file(tmp_path):
    """Test that non-existent file raises FileNotFoundError"""
    missing_path = tmp_path / "missing.json"
    
    with pytest.raises(FileNotFoundError, match="Simulation config file not found"):
        load_simulation_config(str(missing_path))


def test_load_simulation_config_raises_on_invalid_json(tmp_path):
    """Test that invalid JSON raises ValueError"""
    config_file = tmp_path / "bad_config.json"
    config_file.write_text("{ invalid json")
    
    with pytest.raises(ValueError, match="Invalid JSON in simulation config file"):
        load_simulation_config(str(config_file))


def test_load_simulation_config_raises_on_non_dict_json(tmp_path):
    """Test that non-dictionary JSON raises ValueError"""
    config_file = tmp_path / "array_config.json"
    config_file.write_text('["not", "a", "dict"]')
    
    with pytest.raises(ValueError, match="Configuration file must contain a JSON object"):
        load_simulation_config(str(config_file))


def test_load_simulation_config_success(tmp_path, capsys):
    """Test successful loading of simulation config"""
    config_file = tmp_path / "sim_config.json"
    expected_config = {
        "interferometer": "VLA",
        "freq_min": 1.0,
        "freq_max": 2.0,
        "n_chans": 4
    }
    config_file.write_text(json.dumps(expected_config))
    
    result = load_simulation_config(str(config_file))
    
    assert result == expected_config
    
    captured = capsys.readouterr()
    assert f"Loaded simulation config from {str(config_file)}" in captured.out


def test_load_simulation_config_io_error(tmp_path):
    """Test that IOError during file read is caught and re-raised"""
    config_file = tmp_path / "sim_config.json"
    config_file.write_text('{"test": "data"}')
    
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        with pytest.raises(IOError, match="Error reading config file"):
            load_simulation_config(str(config_file))


# ==============================================================================
# Tests for update_bda_config - Input Validation
# ==============================================================================

def test_update_bda_config_raises_on_none_path():
    """Test that None config_path raises ValueError"""
    with pytest.raises(ValueError, match="config_path cannot be None"):
        update_bda_config(None, 0.21, 25.0, 100.0)


def test_update_bda_config_raises_on_empty_path():
    """Test that empty config_path raises ValueError"""
    with pytest.raises(ValueError, match="config_path cannot be None"):
        update_bda_config("", 0.21, 25.0, 100.0)


def test_update_bda_config_raises_on_invalid_lambda_ref_type():
    """Test that non-numeric lambda_ref raises ValueError"""
    with pytest.raises(ValueError, match="lambda_ref must be a number"):
        update_bda_config("path.json", "invalid", 25.0, 100.0)


def test_update_bda_config_raises_on_invalid_max_diameter_type():
    """Test that non-numeric max_diameter raises ValueError"""
    with pytest.raises(ValueError, match="max_diameter must be a positive number"):
        update_bda_config("path.json", 0.21, "invalid", 100.0)


def test_update_bda_config_raises_on_negative_max_diameter():
    """Test that negative max_diameter raises ValueError"""
    with pytest.raises(ValueError, match="max_diameter must be a positive number"):
        update_bda_config("path.json", 0.21, -25.0, 100.0)


def test_update_bda_config_raises_on_zero_max_diameter():
    """Test that zero max_diameter raises ValueError"""
    with pytest.raises(ValueError, match="max_diameter must be a positive number"):
        update_bda_config("path.json", 0.21, 0.0, 100.0)


def test_update_bda_config_raises_on_invalid_threshold_type():
    """Test that non-numeric threshold raises ValueError"""
    with pytest.raises(ValueError, match="threshold must be a positive number"):
        update_bda_config("path.json", 0.21, 25.0, "invalid")


def test_update_bda_config_raises_on_negative_threshold():
    """Test that negative threshold raises ValueError"""
    with pytest.raises(ValueError, match="threshold must be a positive number"):
        update_bda_config("path.json", 0.21, 25.0, -100.0)


def test_update_bda_config_raises_on_zero_threshold():
    """Test that zero threshold raises ValueError"""
    with pytest.raises(ValueError, match="threshold must be a positive number"):
        update_bda_config("path.json", 0.21, 25.0, 0.0)


def test_update_bda_config_raises_on_nonexistent_file(tmp_path):
    """Test that non-existent file raises FileNotFoundError"""
    missing_path = tmp_path / "missing_bda.json"
    
    with pytest.raises(FileNotFoundError, match="BDA config file not found"):
        update_bda_config(str(missing_path), 0.21, 25.0, 100.0)


# ==============================================================================
# Tests for update_bda_config - File Format Validation
# ==============================================================================

def test_update_bda_config_raises_on_invalid_json(tmp_path):
    """Test that invalid JSON raises ValueError"""
    config_file = tmp_path / "bda_config.json"
    config_file.write_text("{ invalid json")
    
    with pytest.raises(ValueError, match="Invalid JSON in BDA config file"):
        update_bda_config(str(config_file), 0.21, 25.0, 100.0)


def test_update_bda_config_raises_on_non_dict_json(tmp_path):
    """Test that non-dictionary JSON raises ValueError"""
    config_file = tmp_path / "bda_config.json"
    config_file.write_text('["not", "a", "dict"]')
    
    with pytest.raises(ValueError, match="BDA configuration file must contain a JSON object"):
        update_bda_config(str(config_file), 0.21, 25.0, 100.0)


def test_update_bda_config_raises_on_missing_fov_strategy(tmp_path):
    """Test that missing fov_strategy raises ValueError"""
    config_file = tmp_path / "bda_config.json"
    config_file.write_text('{"threshold_strategy": "FIXED"}')
    
    with pytest.raises(ValueError, match="BDA config missing required field: 'fov_strategy'"):
        update_bda_config(str(config_file), 0.21, 25.0, 100.0)


def test_update_bda_config_raises_on_invalid_fov_strategy(tmp_path):
    """Test that invalid fov_strategy raises ValueError"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "INVALID",
        "threshold_strategy": "FIXED"
    }
    config_file.write_text(json.dumps(config))
    
    with pytest.raises(ValueError, match="Invalid fov_strategy INVALID"):
        update_bda_config(str(config_file), 0.21, 25.0, 100.0)


def test_update_bda_config_raises_on_missing_threshold_strategy(tmp_path):
    """Test that missing threshold_strategy raises ValueError"""
    config_file = tmp_path / "bda_config.json"
    config = {"fov_strategy": "FIXED", "fov": 0.001}
    config_file.write_text(json.dumps(config))
    
    with pytest.raises(ValueError, match="BDA config missing required field: 'threshold_strategy'"):
        update_bda_config(str(config_file), 0.21, 25.0, 100.0)


def test_update_bda_config_raises_on_invalid_threshold_strategy(tmp_path):
    """Test that invalid threshold_strategy raises ValueError"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "FIXED",
        "fov": 0.001,
        "threshold_strategy": "UNKNOWN"
    }
    config_file.write_text(json.dumps(config))
    
    with pytest.raises(ValueError, match="Invalid threshold_strategy 'UNKNOWN'"):
        update_bda_config(str(config_file), 0.21, 25.0, 100.0)


# ==============================================================================
# Tests for update_bda_config - DERIVED Strategy
# ==============================================================================

def test_update_bda_config_derived_fov_strategy(tmp_path):
    """Test DERIVED fov_strategy calculates FOV correctly"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "DERIVED",
        "threshold_strategy": "DERIVED"
    }
    config_file.write_text(json.dumps(config))
    
    lambda_ref = 0.21
    max_diameter = 25.0
    threshold = 100.0
    
    update_bda_config(str(config_file), lambda_ref, max_diameter, threshold)
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    expected_fov = 0.55 * lambda_ref / (np.pi * max_diameter)
    
    assert result["lambda_ref"] == lambda_ref
    assert result["fov"] == pytest.approx(expected_fov)
    assert result["threshold"] == threshold


def test_update_bda_config_derived_threshold_strategy(tmp_path):
    """Test DERIVED threshold_strategy sets threshold correctly"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "FIXED",
        "fov": 0.001,
        "threshold_strategy": "DERIVED"
    }
    config_file.write_text(json.dumps(config))
    
    threshold = 150.0
    
    update_bda_config(str(config_file), 0.21, 25.0, threshold)
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    assert result["threshold"] == threshold


# ==============================================================================
# Tests for update_bda_config - FIXED Strategy
# ==============================================================================

def test_update_bda_config_fixed_fov_strategy_with_existing_fov(tmp_path):
    """Test FIXED fov_strategy preserves existing FOV"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "FIXED",
        "fov": 0.002,
        "threshold_strategy": "FIXED",
        "threshold": 200.0
    }
    config_file.write_text(json.dumps(config))
    
    update_bda_config(str(config_file), 0.21, 25.0, 100.0)
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    assert result["fov"] == 0.002  # Should preserve existing value


def test_update_bda_config_fixed_fov_strategy_without_fov(tmp_path):
    """Test FIXED fov_strategy sets default FOV if missing"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "FIXED",
        "threshold_strategy": "FIXED",
        "threshold": 200.0
    }
    config_file.write_text(json.dumps(config))
    
    update_bda_config(str(config_file), 0.21, 25.0, 100.0)
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    assert result["fov"] == 0.001  # Default value


def test_update_bda_config_fixed_threshold_strategy_with_existing_threshold(tmp_path):
    """Test FIXED threshold_strategy preserves existing threshold"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "FIXED",
        "fov": 0.001,
        "threshold_strategy": "FIXED",
        "threshold": 300.0
    }
    config_file.write_text(json.dumps(config))
    
    update_bda_config(str(config_file), 0.21, 25.0, 100.0)
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    assert result["threshold"] == 300.0  # Should preserve existing value


def test_update_bda_config_fixed_threshold_strategy_without_threshold(tmp_path):
    """Test FIXED threshold_strategy sets default threshold if missing"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "FIXED",
        "fov": 0.001,
        "threshold_strategy": "FIXED"
    }
    config_file.write_text(json.dumps(config))
    
    update_bda_config(str(config_file), 0.21, 25.0, 100.0)
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    assert result["threshold"] == 5000.0  # Default value


# ==============================================================================
# Tests for update_bda_config - File I/O Errors
# ==============================================================================

def test_update_bda_config_io_error_on_write(tmp_path):
    """Test that IOError during write is caught and re-raised as OSError"""
    config_file = tmp_path / "bda_config.json"
    config = {
        "fov_strategy": "FIXED",
        "fov": 0.001,
        "threshold_strategy": "FIXED"
    }
    config_file.write_text(json.dumps(config))
    
    with patch("builtins.open", side_effect=IOError("Disk full")):
        with pytest.raises(OSError, match="Failed to read/write BDA config file"):
            update_bda_config(str(config_file), 0.21, 25.0, 100.0)


# ==============================================================================
# Tests for update_grid_config - Input Validation
# ==============================================================================

def test_update_grid_config_raises_on_none_path():
    """Test that None config_path raises ValueError"""
    with pytest.raises(ValueError, match="config_path cannot be None or empty"):
        update_grid_config(None, 1e-5, "XX,YY", [1.4e9])


def test_update_grid_config_raises_on_empty_path():
    """Test that empty config_path raises ValueError"""
    with pytest.raises(ValueError, match="config_path cannot be None or empty"):
        update_grid_config("", 1e-5, "XX,YY", [1.4e9])


def test_update_grid_config_raises_on_invalid_theo_resolution_type():
    """Test that non-numeric theo_resolution raises ValueError"""
    with pytest.raises(ValueError, match="theo_resolution must be a positive number"):
        update_grid_config("path.json", "invalid", "XX,YY", [1.4e9])


def test_update_grid_config_raises_on_negative_theo_resolution():
    """Test that negative theo_resolution raises ValueError"""
    with pytest.raises(ValueError, match="theo_resolution must be a positive number"):
        update_grid_config("path.json", -1e-5, "XX,YY", [1.4e9])


def test_update_grid_config_raises_on_zero_theo_resolution():
    """Test that zero theo_resolution raises ValueError"""
    with pytest.raises(ValueError, match="theo_resolution must be a positive number"):
        update_grid_config("path.json", 0.0, "XX,YY", [1.4e9])


def test_update_grid_config_raises_on_none_corrs_string():
    """Test that None corrs_string raises ValueError"""
    with pytest.raises(ValueError, match="corrs_string cannot be None"):
        update_grid_config("path.json", 1e-5, None, [1.4e9])


def test_update_grid_config_raises_on_none_chan_freq():
    """Test that None chan_freq raises ValueError"""
    with pytest.raises(ValueError, match="chan_freq must be a non-empty list"):
        update_grid_config("path.json", 1e-5, "XX,YY", None)


def test_update_grid_config_raises_on_non_list_chan_freq():
    """Test that non-list chan_freq raises ValueError"""
    with pytest.raises(ValueError, match="chan_freq must be a non-empty list"):
        update_grid_config("path.json", 1e-5, "XX,YY", "not a list")


def test_update_grid_config_raises_on_empty_chan_freq():
    """Test that empty chan_freq raises ValueError"""
    with pytest.raises(ValueError, match="chan_freq cannot be empty"):
        update_grid_config("path.json", 1e-5, "XX,YY", [])


def test_update_grid_config_raises_on_nonexistent_file(tmp_path):
    """Test that non-existent file raises FileNotFoundError"""
    missing_path = tmp_path / "missing_grid.json"
    
    with pytest.raises(FileNotFoundError, match="Grid config file not found"):
        update_grid_config(str(missing_path), 1e-5, "XX,YY", [1.4e9])


# ==============================================================================
# Tests for update_grid_config - File Format Validation
# ==============================================================================

def test_update_grid_config_raises_on_invalid_json(tmp_path):
    """Test that invalid JSON raises ValueError"""
    config_file = tmp_path / "grid_config.json"
    config_file.write_text("{ invalid json")
    
    with pytest.raises(ValueError, match="Invalid JSON in grid config file"):
        update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])


def test_update_grid_config_raises_on_non_dict_json(tmp_path):
    """Test that non-dictionary JSON raises ValueError"""
    config_file = tmp_path / "grid_config.json"
    config_file.write_text('["not", "a", "dict"]')
    
    with pytest.raises(ValueError, match="Grid config file must contain a JSON object"):
        update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])


def test_update_grid_config_raises_on_missing_cellsize_strategy(tmp_path):
    """Test that missing cellsize_strategy raises ValueError"""
    config_file = tmp_path / "grid_config.json"
    config_file.write_text('{"img_size": 512}')
    
    with pytest.raises(ValueError, match="Grid config missing required field: 'cellsize_strategy'"):
        update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])


def test_update_grid_config_raises_on_invalid_cellsize_strategy(tmp_path):
    """Test that invalid cellsize_strategy raises ValueError"""
    config_file = tmp_path / "grid_config.json"
    config = {"cellsize_strategy": "UNKNOWN"}
    config_file.write_text(json.dumps(config))
    
    with pytest.raises(ValueError, match="Invalid cellsize_strategy 'UNKNOWN'"):
        update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])


# ==============================================================================
# Tests for update_grid_config - DERIVED Strategy
# ==============================================================================

def test_update_grid_config_derived_cellsize_strategy(tmp_path):
    """Test DERIVED cellsize_strategy calculates cellsize correctly"""
    config_file = tmp_path / "grid_config.json"
    config = {"cellsize_strategy": "DERIVED"}
    config_file.write_text(json.dumps(config))
    
    theo_resolution = 1e-5
    
    update_grid_config(str(config_file), theo_resolution, "XX,YY", [1.4e9, 1.5e9])
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    expected_cellsize = theo_resolution / 7.0
    
    assert result["cellsize"] == pytest.approx(expected_cellsize)
    assert result["corrs_string"] == "XX,YY"
    assert result["chan_freq"] == [1.4e9, 1.5e9]


# ==============================================================================
# Tests for update_grid_config - FIXED Strategy
# ==============================================================================

def test_update_grid_config_fixed_cellsize_strategy_first_run(tmp_path):
    """Test FIXED cellsize_strategy converts arcsec to radians on first run"""
    config_file = tmp_path / "grid_config.json"
    config = {
        "cellsize_strategy": "FIXED",
        "cellsize": 0.5,  # arcseconds
        "cellsize_flag": True
    }
    config_file.write_text(json.dumps(config))
    
    update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    expected_cellsize_rad = (0.5 * u.arcsec).to(u.rad).value
    
    assert result["cellsize"] == pytest.approx(expected_cellsize_rad)
    assert result["cellsize_flag"] is False


def test_update_grid_config_fixed_cellsize_strategy_subsequent_run(tmp_path):
    """Test FIXED cellsize_strategy preserves value on subsequent runs"""
    config_file = tmp_path / "grid_config.json"
    cellsize_rad = 2.4242e-6
    config = {
        "cellsize_strategy": "FIXED",
        "cellsize": cellsize_rad,
        "cellsize_flag": False
    }
    config_file.write_text(json.dumps(config))
    
    update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    assert result["cellsize"] == cellsize_rad
    assert result["cellsize_flag"] is False


def test_update_grid_config_fixed_cellsize_strategy_missing_cellsize(tmp_path):
    """Test FIXED strategy with missing cellsize raises ValueError"""
    config_file = tmp_path / "grid_config.json"
    config = {
        "cellsize_strategy": "FIXED",
        "cellsize_flag": True
    }
    config_file.write_text(json.dumps(config))
    
    with pytest.raises(ValueError, match="Grid config with FIXED strategy missing 'cellsize' field"):
        update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])


def test_update_grid_config_fixed_cellsize_strategy_default_flag(tmp_path):
    """Test FIXED strategy uses default cellsize_flag=True if missing"""
    config_file = tmp_path / "grid_config.json"
    config = {
        "cellsize_strategy": "FIXED",
        "cellsize": 0.5
        # cellsize_flag not provided
    }
    config_file.write_text(json.dumps(config))
    
    update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])
    
    with open(config_file, 'r') as f:
        result = json.load(f)
    
    assert result["cellsize_flag"] is False


# ==============================================================================
# Tests for update_grid_config - File I/O Errors
# ==============================================================================

def test_update_grid_config_io_error_on_write(tmp_path):
    """Test that IOError during write is caught and re-raised as OSError"""
    config_file = tmp_path / "grid_config.json"
    config = {"cellsize_strategy": "DERIVED"}
    config_file.write_text(json.dumps(config))
    
    with patch("builtins.open", side_effect=IOError("Disk full")):
        with pytest.raises(OSError, match="Failed to read/write grid config file"):
            update_grid_config(str(config_file), 1e-5, "XX,YY", [1.4e9])


# ==============================================================================
# Tests for stream_kafka - Input Validation
# ==============================================================================

def test_stream_kafka_raises_on_none_dataset():
    """Test that None dataset raises ValueError"""
    with pytest.raises(ValueError, match="dataset cannot be None"):
        stream_kafka(None, "test-topic")


def test_stream_kafka_raises_on_missing_ms_list():
    """Test that dataset without ms_list raises ValueError"""
    mock_dataset_producer = Mock(spec=[])  # No ms_list attribute
    
    with pytest.raises(ValueError, match="dataset must have 'ms_list' attribute"):
        stream_kafka(mock_dataset_producer, "test-topic")


def test_stream_kafka_raises_on_none_ms_list():
    """Test that dataset with None ms_list raises ValueError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = None
    
    with pytest.raises(ValueError, match="dataset.ms_list cannot be None"):
        stream_kafka(mock_dataset_producer, "test-topic")


def test_stream_kafka_raises_on_none_topic():
    """Test that None topic raises ValueError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = []
    
    with pytest.raises(ValueError, match="topic cannot be None or empty"):
        stream_kafka(mock_dataset_producer, None)


def test_stream_kafka_raises_on_empty_topic():
    """Test that empty topic raises ValueError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = []
    
    with pytest.raises(ValueError, match="topic cannot be None or empty"):
        stream_kafka(mock_dataset_producer, "")


# ==============================================================================
# Tests for stream_kafka - Streaming Behavior
# ==============================================================================

def test_stream_kafka_empty_ms_list(capsys):
    """Test streaming with empty ms_list"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = []
    
    with patch(f"{MODULE}.stream_dataset") as mock_stream:
        stream_kafka(mock_dataset_producer, "test-topic")
        
        mock_stream.assert_not_called()
        
        captured = capsys.readouterr()
        assert "Starting to stream dataset" in captured.out
        assert "Streamed 0 measurement sets" in captured.out


def test_stream_kafka_skips_none_subms(capsys):
    """Test that None measurement sets are skipped"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = [None, None]
    
    with patch(f"{MODULE}.stream_dataset") as mock_stream:
        stream_kafka(mock_dataset_producer, "test-topic")
        
        mock_stream.assert_not_called()
        
        captured = capsys.readouterr()
        assert "Warning: Skipping None measurement set at index 0" in captured.out
        assert "Warning: Skipping None measurement set at index 1" in captured.out


def test_stream_kafka_skips_subms_without_visibilities(capsys):
    """Test that measurement sets without visibilities are skipped"""
    mock_subms = Mock(spec=[])  # No visibilities attribute
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = [mock_subms]
    
    with patch(f"{MODULE}.stream_dataset") as mock_stream:
        stream_kafka(mock_dataset_producer, "test-topic")
        
        mock_stream.assert_not_called()
        
        captured = capsys.readouterr()
        assert "Warning: Skipping measurement set 0 with no visibilities" in captured.out


def test_stream_kafka_skips_subms_with_none_visibilities(capsys):
    """Test that measurement sets with None visibilities are skipped"""
    mock_subms = Mock()
    mock_subms.visibilities = None
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = [mock_subms]
    
    with patch(f"{MODULE}.stream_dataset") as mock_stream:
        stream_kafka(mock_dataset_producer, "test-topic")
        
        mock_stream.assert_not_called()
        
        captured = capsys.readouterr()
        assert "Warning: Skipping measurement set 0 with no visibilities" in captured.out


def test_stream_kafka_successful_streaming(capsys):
    """Test successful streaming of valid measurement sets"""
    mock_vis1 = Mock()
    mock_vis1.__len__ = Mock(return_value=100)
    
    mock_vis2 = Mock()
    mock_vis2.__len__ = Mock(return_value=150)
    
    mock_subms1 = Mock()
    mock_subms1.visibilities = mock_vis1
    
    mock_subms2 = Mock()
    mock_subms2.visibilities = mock_vis2
    
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = [mock_subms1, mock_subms2]
    
    with patch(f"{MODULE}.stream_dataset") as mock_stream:
        stream_kafka(mock_dataset_producer, "test-topic")
        
        assert mock_stream.call_count == 2
        mock_stream.assert_any_call(mock_vis1, mock_subms1, "test-topic")
        mock_stream.assert_any_call(mock_vis2, mock_subms2, "test-topic")
        
        captured = capsys.readouterr()
        assert "Streamed 2 measurement sets" in captured.out
        assert "Total visibilities streamed: 250" in captured.out


def test_stream_kafka_mixed_valid_and_invalid(capsys):
    """Test streaming with mix of valid and invalid measurement sets"""
    mock_vis = Mock()
    mock_vis.__len__ = Mock(return_value=50)
    
    mock_subms_valid = Mock()
    mock_subms_valid.visibilities = mock_vis
    
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = [None, mock_subms_valid, None]
    
    with patch(f"{MODULE}.stream_dataset") as mock_stream:
        stream_kafka(mock_dataset_producer, "test-topic")
        
        mock_stream.assert_called_once_with(mock_vis, mock_subms_valid, "test-topic")
        
        captured = capsys.readouterr()
        assert "Streamed 1 measurement sets" in captured.out
        assert "Total visibilities streamed: 50" in captured.out


def test_stream_kafka_propagates_streaming_error(capsys):
    """Test that exceptions during streaming are caught and re-raised"""
    mock_vis = Mock()
    mock_subms = Mock()
    mock_subms.visibilities = mock_vis
    
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = [mock_subms]
    
    with patch(f"{MODULE}.stream_dataset", side_effect=RuntimeError("Kafka connection failed")):
        with pytest.raises(RuntimeError, match="Failed to stream dataset to Kafka"):
            stream_kafka(mock_dataset_producer, "test-topic")
        
        captured = capsys.readouterr()
        assert "Error during streaming: Kafka connection failed" in captured.out


def test_stream_kafka_handles_visibilities_without_len():
    """Test streaming when visibilities don't have __len__"""
    mock_vis = Mock(spec=[])  # No __len__
    mock_subms = Mock()
    mock_subms.visibilities = mock_vis
    
    mock_dataset_producer = Mock()
    mock_dataset_producer.ms_list = [mock_subms]
    
    with patch(f"{MODULE}.stream_dataset") as mock_stream:
        stream_kafka(mock_dataset_producer, "test-topic")
        
        mock_stream.assert_called_once()

# ==============================================================================
# Tests for run_producer - Input Validation
# ==============================================================================

def test_run_producer_raises_on_none_antenna_config():
    """Test that None antenna_config_path raises ValueError"""
    with pytest.raises(ValueError, match="Antenna configuration path is required"):
        run_producer(None, "sim_config.json", "test-topic")


def test_run_producer_raises_on_empty_antenna_config():
    """Test that empty antenna_config_path raises ValueError"""
    with pytest.raises(ValueError, match="Antenna configuration path is required"):
        run_producer("", "sim_config.json", "test-topic")


def test_run_producer_raises_on_nonexistent_antenna_config(tmp_path):
    """Test that non-existent antenna_config_path raises FileNotFoundError"""
    missing_path = tmp_path / "missing.cfg"
    
    with pytest.raises(FileNotFoundError, match="Antenna configuration file not found"):
        run_producer(str(missing_path), "sim_config.json", "test-topic")


def test_run_producer_uses_default_topic_when_none(tmp_antenna_config, tmp_sim_config, capsys):
    """Test that None topic uses DEFAULT_TOPIC"""
    with patch(f"{MODULE}.load_simulation_config") as mock_load, \
         patch(f"{MODULE}.generate_dataset", side_effect=RuntimeError("Stop early")):
        
        mock_load.return_value = {}
        
        with pytest.raises(RuntimeError):
            run_producer(tmp_antenna_config, tmp_sim_config, None)
        
        captured = capsys.readouterr()
        assert f"Using default Kafka topic: {DEFAULT_TOPIC}" in captured.out


# ==============================================================================
# Tests for run_producer - Step 1: Load Simulation Config
# ==============================================================================

def test_run_producer_step1_load_config_failure(tmp_antenna_config):
    """Test failure in load_simulation_config propagates"""
    with patch(f"{MODULE}.load_simulation_config", side_effect=ValueError("Invalid JSON")):
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            run_producer(tmp_antenna_config, "bad_config.json", "test-topic")


# ==============================================================================
# Tests for run_producer - Step 2: Generate Dataset
# ==============================================================================

def test_run_producer_step2_dataset_none(tmp_antenna_config, tmp_sim_config):
    """Test that None dataset raises RuntimeError"""
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(None, Mock())):
        
        with pytest.raises(RuntimeError, match="Dataset generation returned None"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step2_interferometer_none(tmp_antenna_config, tmp_sim_config):
    """Test that None interferometer raises RuntimeError"""
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(Mock(), None)):
        
        with pytest.raises(RuntimeError, match="Interferometer generation returned None"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


# ==============================================================================
# Tests for run_producer - Step 3: Update BDA Config
# ==============================================================================

def test_run_producer_step3_dataset_missing_spws(tmp_antenna_config, tmp_sim_config):
    """Test dataset without spws attribute raises RuntimeError"""
    mock_dataset_producer = Mock(spec=[])  # No spws attribute
    mock_interferometer_producer = Mock()
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)):
        
        with pytest.raises(RuntimeError, match="Dataset missing 'spws' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step3_dataset_spws_none(tmp_antenna_config, tmp_sim_config):
    """Test dataset with None spws raises RuntimeError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.spws = None
    mock_interferometer_producer = Mock()
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)):
        
        with pytest.raises(RuntimeError, match="Dataset missing 'spws' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step3_spws_missing_lambda_ref(tmp_antenna_config, tmp_sim_config):
    """Test dataset.spws without lambda_ref raises RuntimeError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.spws = Mock(spec=[])  # No lambda_ref
    mock_interferometer_producer = Mock()
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)):
        
        with pytest.raises(RuntimeError, match="Dataset.spws missing 'lambda_ref' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step3_interferometer_missing_antenna_array(
    tmp_antenna_config, tmp_sim_config, mock_dataset_producer
):
    """Test interferometer without antenna_array raises RuntimeError"""
    mock_interferometer_producer = Mock(spec=[])  # No antenna_array
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)):
        
        with pytest.raises(RuntimeError, match="Interferometer missing 'antenna_array' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step3_update_bda_config_failure(
    tmp_antenna_config, tmp_sim_config, mock_dataset_producer, mock_interferometer_producer
):
    """Test failure in update_bda_config propagates"""
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config", side_effect=OSError("Write failed")):
        
        with pytest.raises(OSError, match="Write failed"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


# ==============================================================================
# Tests for run_producer - Step 4: Update Grid Config
# ==============================================================================

def test_run_producer_step4_dataset_missing_theo_resolution(
    tmp_antenna_config, tmp_sim_config, mock_interferometer_producer, tmp_bda_config
):
    """Test dataset without theo_resolution raises RuntimeError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.spws = Mock()
    mock_dataset_producer.spws.lambda_ref = 0.21
    del mock_dataset_producer.theo_resolution  # Remove attribute
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"):
        
        with pytest.raises(RuntimeError, match="Dataset missing 'theo_resolution' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step4_dataset_missing_polarization(
    tmp_antenna_config, tmp_sim_config, mock_interferometer_producer, tmp_bda_config
):
    """Test dataset without polarization raises RuntimeError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.spws = Mock()
    mock_dataset_producer.spws.lambda_ref = 0.21
    mock_dataset_producer.theo_resolution = 1e-5
    del mock_dataset_producer.polarization  # Remove attribute
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"):
        
        with pytest.raises(RuntimeError, match="Dataset missing 'polarization' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step4_dataset_polarization_none(
    tmp_antenna_config, tmp_sim_config, mock_interferometer_producer, tmp_bda_config
):
    """Test dataset with None polarization raises RuntimeError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.spws = Mock()
    mock_dataset_producer.spws.lambda_ref = 0.21
    mock_dataset_producer.theo_resolution = 1e-5
    mock_dataset_producer.polarization = None
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"):
        
        with pytest.raises(RuntimeError, match="Dataset missing 'polarization' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step4_polarization_missing_corrs_string(
    tmp_antenna_config, tmp_sim_config, mock_interferometer_producer, tmp_bda_config
):
    """Test polarization without corrs_string raises RuntimeError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.spws = Mock()
    mock_dataset_producer.spws.lambda_ref = 0.21
    mock_dataset_producer.theo_resolution = 1e-5
    mock_dataset_producer.polarization = Mock(spec=[])  # No corrs_string
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"):
        
        with pytest.raises(RuntimeError, match="Dataset.polarization missing 'corrs_string' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step4_empty_spws_dataset(
    tmp_antenna_config, tmp_sim_config, mock_interferometer_producer    , tmp_bda_config
):
    """Test empty spws.dataset raises RuntimeError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.spws = Mock()
    mock_dataset_producer.spws.lambda_ref = 0.21
    mock_dataset_producer.spws.dataset = []  # Empty
    mock_dataset_producer.theo_resolution = 1e-5
    mock_dataset_producer.polarization = Mock()
    mock_dataset_producer.polarization.corrs_string = "XX,YY"
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"):
        
        with pytest.raises(RuntimeError, match="Dataset.spws.dataset is empty"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step4_spw_missing_chan_freq(
    tmp_antenna_config, tmp_sim_config, mock_interferometer_producer, tmp_bda_config
):
    """Test SPW without CHAN_FREQ raises RuntimeError"""
    mock_dataset_producer = Mock()
    mock_dataset_producer.spws = Mock()
    mock_dataset_producer.spws.lambda_ref = 0.21
    mock_dataset_producer.spws.dataset = [Mock(spec=[])]  # No CHAN_FREQ
    mock_dataset_producer.theo_resolution = 1e-5
    mock_dataset_producer.polarization = Mock()
    mock_dataset_producer.polarization.corrs_string = "XX,YY"
    
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"):
        
        with pytest.raises(RuntimeError, match="SPW dataset missing 'CHAN_FREQ' attribute"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


def test_run_producer_step4_update_grid_config_failure(
    tmp_antenna_config, tmp_sim_config, mock_dataset_producer, mock_interferometer_producer, tmp_bda_config
):
    """Test failure in update_grid_config propagates"""
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"), \
         patch(f"{MODULE}.update_grid_config", side_effect=ValueError("Invalid strategy")):
        
        with pytest.raises(ValueError, match="Invalid strategy"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


# ==============================================================================
# Tests for run_producer - Step 5: Stream to Kafka
# ==============================================================================

def test_run_producer_step5_stream_kafka_failure(
    tmp_antenna_config, tmp_sim_config, mock_dataset_producer, mock_interferometer_producer,
    tmp_bda_config, tmp_grid_config
):
    """Test failure in stream_kafka propagates"""
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"), \
         patch(f"{MODULE}.update_grid_config"), \
         patch(f"{MODULE}.stream_kafka", side_effect=RuntimeError("Kafka down")):
        
        with pytest.raises(RuntimeError, match="Kafka down"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")


# ==============================================================================
# Tests for run_producer - Successful Execution
# ==============================================================================

def test_run_producer_successful_execution(
    tmp_antenna_config, tmp_sim_config, mock_dataset_producer, mock_interferometer_producer,
    tmp_bda_config, tmp_grid_config, capsys
):
    """Test complete successful execution of run_producer"""
    with patch(f"{MODULE}.load_simulation_config", return_value={}) as mock_load, \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)) as mock_gen, \
         patch(f"{MODULE}.update_bda_config") as mock_bda, \
         patch(f"{MODULE}.update_grid_config") as mock_grid, \
         patch(f"{MODULE}.stream_kafka") as mock_stream:
        
        result = run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")
        
        # Verify return value
        assert result is True
        
        # Verify all functions were called
        mock_load.assert_called_once_with(tmp_sim_config)
        mock_gen.assert_called_once()
        mock_bda.assert_called_once()
        mock_grid.assert_called_once()
        mock_stream.assert_called_once_with(mock_dataset_producer, "test-topic")
        
        # Verify output messages
        captured = capsys.readouterr()
        assert "Step 1/5: Loading simulation configuration" in captured.out
        assert "Step 2/5: Generating dataset" in captured.out
        assert "Step 3/5: Updating BDA configuration" in captured.out
        assert "Step 4/5: Updating grid configuration" in captured.out
        assert "Step 5/5: Streaming to Kafka" in captured.out
        assert "Producer pipeline completed successfully" in captured.out


def test_run_producer_calls_update_bda_config_with_correct_params(
    tmp_antenna_config, tmp_sim_config, mock_dataset_producer, mock_interferometer_producer,
    tmp_bda_config, tmp_grid_config
):
    """Test that update_bda_config is called with correct parameters"""
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config") as mock_bda, \
         patch(f"{MODULE}.update_grid_config"), \
         patch(f"{MODULE}.stream_kafka"):
        
        run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")
        
        # Verify update_bda_config was called with correct arguments
        mock_bda.assert_called_once_with(
            config_path="./configs/bda_config.json",
            lambda_ref=0.21,
            max_diameter=25.0,
            threshold=100.0
        )


def test_run_producer_calls_update_grid_config_with_correct_params(
    tmp_antenna_config, tmp_sim_config, mock_dataset_producer, mock_interferometer_producer,
    tmp_bda_config, tmp_grid_config
):
    """Test that update_grid_config is called with correct parameters"""
    with patch(f"{MODULE}.load_simulation_config", return_value={}), \
         patch(f"{MODULE}.generate_dataset", return_value=(mock_dataset_producer, mock_interferometer_producer)), \
         patch(f"{MODULE}.update_bda_config"), \
         patch(f"{MODULE}.update_grid_config") as mock_grid, \
         patch(f"{MODULE}.stream_kafka"):
        
        run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")
        
        # Verify update_grid_config was called with correct arguments
        mock_grid.assert_called_once_with(
            config_path="./configs/grid_config.json",
            theo_resolution=1e-5,
            corrs_string="XX,YY",
            chan_freq=[1.4e9, 1.5e9, 1.6e9]
        )


def test_run_producer_exception_handling_prints_error(
    tmp_antenna_config, tmp_sim_config, capsys
):
    """Test that exceptions are caught and printed with traceback"""
    with patch(f"{MODULE}.load_simulation_config", side_effect=RuntimeError("Unexpected error")):
        
        with pytest.raises(RuntimeError, match="Unexpected error"):
            run_producer(tmp_antenna_config, tmp_sim_config, "test-topic")
        
        captured = capsys.readouterr()
        assert "Error in producer service: Unexpected error" in captured.out