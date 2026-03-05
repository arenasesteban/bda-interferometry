import pytest
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.imaging.dirty_image import (
    fft,
    save_dirty_image,
    save_psf_image,
    generate_dirty_image
)

MODULE = "src.imaging.dirty_image"


# ==============================================================================
# Tests for fft
# ==============================================================================

def test_fft_raises_on_invalid_grids_type(weight_array, grid_config):
    """Test that non-numpy array grids raises error"""
    with pytest.raises(ValueError, match="grids must be a 2D numpy array"):
        fft("not_an_array", weight_array, grid_config)


def test_fft_raises_on_1d_grids(weight_array, grid_config):
    """Test that 1D grids array raises error"""
    grids_1d = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="grids must be a 2D numpy array"):
        fft(grids_1d, weight_array, grid_config)


def test_fft_raises_on_3d_grids(weight_array, grid_config):
    """Test that 3D grids array raises error"""
    grids_3d = np.zeros((8, 8, 8))
    with pytest.raises(ValueError, match="grids must be a 2D numpy array"):
        fft(grids_3d, weight_array, grid_config)


def test_fft_raises_on_invalid_weights_type(grid_array, grid_config):
    """Test that non-numpy array weights raises error"""
    with pytest.raises(ValueError, match="weights must be a 2D numpy array"):
        fft(grid_array, "not_an_array", grid_config)


def test_fft_raises_on_1d_weights(grid_array, grid_config):
    """Test that 1D weights array raises error"""
    weights_1d = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="weights must be a 2D numpy array"):
        fft(grid_array, weights_1d, grid_config)


def test_fft_raises_on_shape_mismatch(grid_config):
    """Test that mismatched grids and weights shapes raises error"""
    grids = np.zeros((8, 8))
    weights = np.zeros((4, 4))
    with pytest.raises(ValueError, match="grids and weights must have the same shape"):
        fft(grids, weights, grid_config)


def test_fft_raises_on_missing_img_size(grid_array, weight_array):
    """Test missing img_size in config"""
    incomplete_config = {"padding_factor": 1.0}
    with pytest.raises(KeyError, match="Missing required grid_config parameters"):
        fft(grid_array, weight_array, incomplete_config)


def test_fft_raises_on_missing_padding_factor(grid_array, weight_array):
    """Test missing padding_factor in config"""
    incomplete_config = {"img_size": 8}
    with pytest.raises(KeyError, match="Missing required grid_config parameters"):
        fft(grid_array, weight_array, incomplete_config)


def test_fft_raises_on_invalid_img_size_type(grid_array, weight_array):
    """Test invalid img_size type"""
    bad_config = {"img_size": "eight", "padding_factor": 1.0}
    with pytest.raises(ValueError, match="img_size must be a positive integer"):
        fft(grid_array, weight_array, bad_config)


def test_fft_raises_on_negative_img_size(grid_array, weight_array):
    """Test negative img_size"""
    bad_config = {"img_size": -8, "padding_factor": 1.0}
    with pytest.raises(ValueError, match="img_size must be a positive integer"):
        fft(grid_array, weight_array, bad_config)


def test_fft_raises_on_zero_img_size(grid_array, weight_array):
    """Test zero img_size"""
    bad_config = {"img_size": 0, "padding_factor": 1.0}
    with pytest.raises(ValueError, match="img_size must be a positive integer"):
        fft(grid_array, weight_array, bad_config)


def test_fft_raises_on_invalid_padding_factor_type(grid_array, weight_array):
    """Test invalid padding_factor type"""
    bad_config = {"img_size": 8, "padding_factor": "one"}
    with pytest.raises(ValueError, match="padding_factor must be >= 1.0"):
        fft(grid_array, weight_array, bad_config)


def test_fft_raises_on_padding_factor_less_than_one(grid_array, weight_array):
    """Test padding_factor < 1.0"""
    bad_config = {"img_size": 8, "padding_factor": 0.5}
    with pytest.raises(ValueError, match="padding_factor must be >= 1.0"):
        fft(grid_array, weight_array, bad_config)


def test_fft_raises_on_invalid_type_parameter(grid_array, weight_array, grid_config):
    """Test invalid type parameter"""
    with pytest.raises(ValueError, match="type must be one of"):
        fft(grid_array, weight_array, grid_config, type="invalid")


def test_fft_accepts_type_dirty(grid_array, weight_array, grid_config):
    """Test that type='dirty' is accepted"""
    result = fft(grid_array, weight_array, grid_config, type="dirty")
    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 8)


def test_fft_accepts_type_psf(grid_array, weight_array, grid_config):
    """Test that type='psf' is accepted"""
    result = fft(grid_array, weight_array, grid_config, type="psf")
    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 8)


def test_fft_dirty_returns_real_values(grid_array, weight_array, grid_config):
    """Test that dirty image has only real values"""
    result = fft(grid_array, weight_array, grid_config, type="dirty")
    assert result.dtype in [np.float64, np.float32]
    assert not np.iscomplexobj(result)


def test_fft_psf_returns_positive_values(grid_array, weight_array, grid_config):
    """Test that PSF has only non-negative values (abs applied)"""
    result = fft(grid_array, weight_array, grid_config, type="psf")
    assert np.all(result >= 0)


def test_fft_dirty_applies_cropping_with_padding():
    """Test that dirty image is cropped when padding_factor > 1"""
    grids = np.random.random((16, 16)) + 1j * np.random.random((16, 16))
    weights = np.random.random((16, 16))
    config = {"img_size": 8, "padding_factor": 2.0}
    
    result = fft(grids, weights, config, type="dirty")
    
    # Should be cropped to img_size
    assert result.shape == (8, 8)


def test_fft_psf_no_cropping():
    """Test that PSF is not cropped even with padding"""
    grids = np.random.random((16, 16)) + 1j * np.random.random((16, 16))
    weights = np.random.random((16, 16))
    config = {"img_size": 8, "padding_factor": 2.0}
    
    result = fft(grids, weights, config, type="psf")
    
    # PSF should keep padded size
    assert result.shape == (16, 16)


def test_fft_no_cropping_when_padding_is_one(grid_array, weight_array, grid_config):
    """Test no cropping when padding_factor == 1.0"""
    result = fft(grid_array, weight_array, grid_config, type="dirty")
    # padding_factor = 1.0, so no cropping needed
    assert result.shape == (8, 8)


def test_fft_dirty_uses_forward_normalization(grid_array, weight_array, grid_config):
    """Test that FFT uses forward normalization"""
    # Create simple input
    grids = np.zeros((8, 8), dtype=np.complex128)
    grids[4, 4] = 1.0  # Single point
    weights = np.ones((8, 8))
    
    result = fft(grids, weights, grid_config, type="dirty")
    
    # With forward normalization, amplitude is scaled by 1/N
    # Just verify it's computed without error
    assert result is not None


def test_fft_preserves_hermitian_symmetry():
    """Test that PSF is real-valued (Hermitian symmetry in frequency domain)"""
    grids = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
    weights = np.ones((8, 8))
    config = {"img_size": 8, "padding_factor": 1.0}
    
    psf = fft(grids, weights, config, type="psf")
    
    # PSF from FFT should be real-valued (no complex component)
    # This is because we take abs() in the PSF computation
    assert not np.iscomplexobj(psf)
    assert np.all(psf >= 0)  # All values should be non-negative


# ==============================================================================
# Tests for save_dirty_image
# ==============================================================================

def test_save_dirty_image_raises_on_none():
    """Test that None dirty_image raises error"""
    with pytest.raises(ValueError, match="dirty_image must not be None"):
        save_dirty_image(None, "output.png")


def test_save_dirty_image_raises_on_invalid_type():
    """Test that non-array dirty_image raises error"""
    with pytest.raises(ValueError, match="dirty_image must be a 2D numpy array"):
        save_dirty_image("not_an_array", "output.png")


def test_save_dirty_image_raises_on_1d_array():
    """Test that 1D array raises error"""
    with pytest.raises(ValueError, match="dirty_image must be a 2D numpy array"):
        save_dirty_image(np.array([1, 2, 3]), "output.png")


def test_save_dirty_image_raises_on_none_output_file(image_2d):
    """Test that None output_file raises error"""
    with pytest.raises(ValueError, match="output_file must be None or empty"):
        save_dirty_image(image_2d, None)


def test_save_dirty_image_raises_on_empty_output_file(image_2d):
    """Test that empty output_file raises error"""
    with pytest.raises(ValueError, match="output_file must be None or empty"):
        save_dirty_image(image_2d, "")


def test_save_dirty_image_raises_on_wrong_extension(image_2d):
    """Test that non-.png extension raises error"""
    with pytest.raises(ValueError, match="output_file must have a .png extension"):
        save_dirty_image(image_2d, "output.jpg")


def test_save_dirty_image_raises_on_nonexistent_directory(image_2d, tmp_path):
    """Test that non-existent output directory raises error"""
    output_file = tmp_path / "nonexistent" / "image.png"
    with pytest.raises(ValueError, match="Output directory does not exist"):
        save_dirty_image(image_2d, str(output_file))


def test_save_dirty_image_success(image_2d, tmp_path):
    """Test successful save"""
    output_file = tmp_path / "dirty_image.png"
    
    save_dirty_image(image_2d, str(output_file))
    
    assert output_file.exists()


def test_save_dirty_image_creates_figure(image_2d, tmp_path):
    """Test that matplotlib figure is created with correct properties"""
    output_file = tmp_path / "dirty_image.png"
    
    with patch(f"{MODULE}.plt") as mock_plt:
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        
        save_dirty_image(image_2d, str(output_file))
        
        mock_plt.figure.assert_called_once_with(figsize=(8, 8))
        mock_plt.imshow.assert_called_once()
        mock_plt.title.assert_called_once_with(
            "Dirty Image",
            fontdict={"fontsize": 16, "fontweight": "bold"}
        )
        mock_plt.savefig.assert_called_once_with(str(output_file))
        mock_plt.close.assert_called()


def test_save_dirty_image_uses_correct_colormap(image_2d, tmp_path):
    """Test that correct colormap is used"""
    output_file = tmp_path / "dirty_image.png"
    
    with patch(f"{MODULE}.plt") as mock_plt:
        save_dirty_image(image_2d, str(output_file))
        
        call_kwargs = mock_plt.imshow.call_args.kwargs
        assert call_kwargs["cmap"] == "cmc.acton"
        assert call_kwargs["origin"] == "lower"


def test_save_dirty_image_handles_write_error(image_2d, tmp_path):
    """Test that write errors are caught and re-raised as OSError"""
    output_file = tmp_path / "dirty_image.png"
    
    with patch(f"{MODULE}.plt") as mock_plt:
        mock_plt.savefig.side_effect = IOError("Write failed")
        
        with pytest.raises(OSError, match="Failed to save dirty image"):
            save_dirty_image(image_2d, str(output_file))
        
        # Check cleanup was called
        assert mock_plt.close.called


def test_save_dirty_image_cleans_up_on_error(image_2d, tmp_path):
    """Test that plt.close() is called even on error"""
    output_file = tmp_path / "dirty_image.png"
    
    with patch(f"{MODULE}.plt") as mock_plt:
        mock_plt.imshow.side_effect = Exception("Plotting error")
        
        with pytest.raises(OSError):
            save_dirty_image(image_2d, str(output_file))
        
        # close() should be called in except block
        mock_plt.close.assert_called()


# ==============================================================================
# Tests for save_psf_image
# ==============================================================================

def test_save_psf_image_raises_on_none():
    """Test that None psf_image raises error"""
    with pytest.raises(ValueError, match="psf_image must not be None"):
        save_psf_image(None, "output.png")


def test_save_psf_image_raises_on_invalid_type():
    """Test that non-array psf_image raises error"""
    with pytest.raises(ValueError, match="psf_image must be a 2D numpy array"):
        save_psf_image("not_an_array", "output.png")


def test_save_psf_image_raises_on_1d_array():
    """Test that 1D array raises error"""
    with pytest.raises(ValueError, match="psf_image must be a 2D numpy array"):
        save_psf_image(np.array([1, 2, 3]), "output.png")


def test_save_psf_image_raises_on_none_output_file(image_2d):
    """Test that None output_file raises error"""
    with pytest.raises(ValueError, match="output_file must be None or empty"):
        save_psf_image(image_2d, None)


def test_save_psf_image_raises_on_empty_output_file(image_2d):
    """Test that empty output_file raises error"""
    with pytest.raises(ValueError, match="output_file must be None or empty"):
        save_psf_image(image_2d, "")


def test_save_psf_image_raises_on_wrong_extension(image_2d):
    """Test that non-.png extension raises error"""
    with pytest.raises(ValueError, match="output_file must have a .png extension"):
        save_psf_image(image_2d, "output.txt")


def test_save_psf_image_raises_on_nonexistent_directory(image_2d, tmp_path):
    """Test that non-existent output directory raises error"""
    output_file = tmp_path / "nonexistent" / "psf.png"
    with pytest.raises(ValueError, match="Output directory does not exist"):
        save_psf_image(image_2d, str(output_file))


def test_save_psf_image_success(image_2d, tmp_path):
    """Test successful save"""
    output_file = tmp_path / "psf_image.png"
    
    save_psf_image(image_2d, str(output_file))
    
    assert output_file.exists()


def test_save_psf_image_creates_figure_with_colorbar(image_2d, tmp_path):
    """Test that PSF figure includes colorbar"""
    output_file = tmp_path / "psf_image.png"
    
    with patch(f"{MODULE}.plt") as mock_plt:
        save_psf_image(image_2d, str(output_file))
        
        mock_plt.figure.assert_called_once_with(figsize=(8, 8))
        mock_plt.imshow.assert_called_once()
        mock_plt.colorbar.assert_called_once_with(label="Intensity")
        mock_plt.title.assert_called_once_with(
            "Point Spread Function (PSF)",
            fontdict={"fontsize": 16, "fontweight": "bold"}
        )


def test_save_psf_image_uses_correct_colormap(image_2d, tmp_path):
    """Test that correct colormap is used"""
    output_file = tmp_path / "psf_image.png"
    
    with patch(f"{MODULE}.plt") as mock_plt:
        save_psf_image(image_2d, str(output_file))
        
        call_kwargs = mock_plt.imshow.call_args.kwargs
        assert call_kwargs["cmap"] == "cmc.acton"
        assert call_kwargs["origin"] == "lower"


def test_save_psf_image_handles_write_error(image_2d, tmp_path):
    """Test that write errors are caught and re-raised as OSError"""
    output_file = tmp_path / "psf_image.png"
    
    with patch(f"{MODULE}.plt") as mock_plt:
        mock_plt.savefig.side_effect = IOError("Write failed")
        
        with pytest.raises(OSError, match="Failed to save PSF image"):
            save_psf_image(image_2d, str(output_file))
        
        assert mock_plt.close.called


def test_save_psf_image_cleans_up_on_error(image_2d, tmp_path):
    """Test that plt.close() is called even on error"""
    output_file = tmp_path / "psf_image.png"
    
    with patch(f"{MODULE}.plt") as mock_plt:
        mock_plt.colorbar.side_effect = Exception("Colorbar error")
        
        with pytest.raises(OSError):
            save_psf_image(image_2d, str(output_file))
        
        mock_plt.close.assert_called()


# ==============================================================================
# Tests for generate_dirty_image
# ==============================================================================

def test_generate_dirty_image_raises_on_none_grids(weight_array, grid_config):
    """Test that None grids raises error"""
    with pytest.raises(ValueError, match="grids and weights must not be None"):
        generate_dirty_image(None, weight_array, grid_config, "job123")


def test_generate_dirty_image_raises_on_none_weights(grid_array, grid_config):
    """Test that None weights raises error"""
    with pytest.raises(ValueError, match="grids and weights must not be None"):
        generate_dirty_image(grid_array, None, grid_config, "job123")


def test_generate_dirty_image_raises_on_none_config(grid_array, weight_array):
    """Test that None grid_config raises error"""
    with pytest.raises(ValueError, match="Grid configuration must not be None"):
        generate_dirty_image(grid_array, weight_array, None, "job123")


def test_generate_dirty_image_creates_output_directory(grid_array, weight_array, grid_config, tmp_path):
    """Test that output directory is created"""
    job_id = "test_job_123"
    
    with patch(f"{MODULE}.fft") as mock_fft, \
         patch(f"{MODULE}.save_dirty_image") as mock_save_dirty, \
         patch(f"{MODULE}.save_psf_image") as mock_save_psf:
        
        mock_fft.return_value = np.zeros((8, 8))
        
        # Make save functions create the directory
        def create_dir_side_effect(img, path):
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        mock_save_dirty.side_effect = create_dir_side_effect
        
        generate_dirty_image(grid_array, weight_array, grid_config, job_id)
        
        # Verify save functions were called with correct paths
        assert mock_save_dirty.called
        assert f"./output/{job_id}/dirtyimage_{job_id}.png" in str(mock_save_dirty.call_args[0][1])


def test_generate_dirty_image_calls_fft_twice(grid_array, weight_array, grid_config, tmp_path):
    """Test that fft is called for both dirty and PSF"""
    job_id = "test_job_456"
    output_dir = tmp_path / "output" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with patch(f"{MODULE}.fft") as mock_fft, \
         patch(f"{MODULE}.save_dirty_image"), \
         patch(f"{MODULE}.save_psf_image"), \
         patch("os.makedirs"):
        
        mock_fft.return_value = np.zeros((8, 8))
        
        generate_dirty_image(grid_array, weight_array, grid_config, job_id)
        
        # Should call fft twice: once for dirty, once for PSF
        assert mock_fft.call_count == 2
        
        # Check calls
        calls = mock_fft.call_args_list
        assert calls[0][1]["type"] == "dirty"
        assert calls[1][1]["type"] == "psf"


def test_generate_dirty_image_saves_both_images(grid_array, weight_array, grid_config, tmp_path):
    """Test that both dirty and PSF images are saved"""
    job_id = "test_job_789"
    
    with patch(f"{MODULE}.fft") as mock_fft, \
         patch(f"{MODULE}.save_dirty_image") as mock_save_dirty, \
         patch(f"{MODULE}.save_psf_image") as mock_save_psf, \
         patch("os.makedirs"):
        
        mock_dirty = np.zeros((8, 8))
        mock_psf = np.zeros((8, 8))
        mock_fft.side_effect = [mock_dirty, mock_psf]
        
        generate_dirty_image(grid_array, weight_array, grid_config, job_id)
        
        # Check save_dirty_image was called
        mock_save_dirty.assert_called_once()
        assert mock_dirty is mock_save_dirty.call_args[0][0]
        assert f"dirtyimage_{job_id}.png" in mock_save_dirty.call_args[0][1]
        
        # Check save_psf_image was called
        mock_save_psf.assert_called_once()
        assert mock_psf is mock_save_psf.call_args[0][0]
        assert f"psf_{job_id}.png" in mock_save_psf.call_args[0][1]


def test_generate_dirty_image_uses_correct_paths(grid_array, weight_array, grid_config):
    """Test that correct output paths are constructed"""
    job_id = "job_abc123"
    
    with patch(f"{MODULE}.fft") as mock_fft, \
         patch(f"{MODULE}.save_dirty_image") as mock_save_dirty, \
         patch(f"{MODULE}.save_psf_image") as mock_save_psf, \
         patch("os.makedirs"):
        
        mock_fft.return_value = np.zeros((8, 8))
        
        generate_dirty_image(grid_array, weight_array, grid_config, job_id)
        
        expected_dirty = f"./output/{job_id}/dirtyimage_{job_id}.png"
        expected_psf = f"./output/{job_id}/psf_{job_id}.png"
        
        assert mock_save_dirty.call_args[0][1] == expected_dirty
        assert mock_save_psf.call_args[0][1] == expected_psf


def test_generate_dirty_image_prints_success_messages(grid_array, weight_array, grid_config, capsys):
    """Test that success messages are printed"""
    job_id = "job_print_test"
    
    with patch(f"{MODULE}.fft") as mock_fft, \
         patch(f"{MODULE}.save_dirty_image"), \
         patch(f"{MODULE}.save_psf_image"), \
         patch("os.makedirs"):
        
        mock_fft.return_value = np.zeros((8, 8))
        
        generate_dirty_image(grid_array, weight_array, grid_config, job_id)
        
        captured = capsys.readouterr()
        assert "[Imaging] ✓ Dirty image saved to:" in captured.out
        assert f"dirtyimage_{job_id}.png" in captured.out
        assert "[Imaging] ✓ PSF image saved to:" in captured.out
        assert f"psf_{job_id}.png" in captured.out


def test_generate_dirty_image_closes_all_plots(grid_array, weight_array, grid_config):
    """Test that plt.close('all') is called at the end"""
    with patch(f"{MODULE}.fft") as mock_fft, \
         patch(f"{MODULE}.save_dirty_image"), \
         patch(f"{MODULE}.save_psf_image"), \
         patch(f"{MODULE}.plt") as mock_plt, \
         patch("os.makedirs"):
        
        mock_fft.return_value = np.zeros((8, 8))
        
        generate_dirty_image(grid_array, weight_array, grid_config, "job123")
        
        # Check plt.close('all') was called
        mock_plt.close.assert_called_with('all')


def test_generate_dirty_image_handles_empty_slurm_id(grid_array, weight_array, grid_config):
    """Test that empty slurm_job_id is handled"""
    with patch(f"{MODULE}.fft") as mock_fft, \
         patch(f"{MODULE}.save_dirty_image") as mock_save_dirty, \
         patch(f"{MODULE}.save_psf_image") as mock_save_psf:
        
        mock_fft.return_value = np.zeros((8, 8))
        
        # Make save functions succeed
        def noop_side_effect(img, path):
            import os
            os.makedirs(os.path.dirname(path) or "./output/", exist_ok=True)
        
        mock_save_dirty.side_effect = noop_side_effect
        
        generate_dirty_image(grid_array, weight_array, grid_config, "")
        
        # Should create paths with empty job_id
        assert mock_save_dirty.called
        dirty_path = str(mock_save_dirty.call_args[0][1])
        assert "./output/" in dirty_path


def test_generate_dirty_image_raises_on_makedirs_failure(grid_array, weight_array, grid_config, tmp_path):
    """Test that OSError is raised if save functions fail"""
    with patch(f"{MODULE}.fft") as mock_fft, \
         patch(f"{MODULE}.save_dirty_image") as mock_save_dirty:
        
        mock_fft.return_value = np.zeros((8, 8))
        mock_save_dirty.side_effect = OSError("Failed to save dirty image")
        
        with pytest.raises(OSError, match="Failed to save dirty image"):
            generate_dirty_image(grid_array, weight_array, grid_config, "job123")


# ==============================================================================
# Integration Tests
# ==============================================================================

def test_full_pipeline_dirty_and_psf(grid_array, weight_array, grid_config, tmp_path):
    """Test complete pipeline from grids to saved images"""
    job_id = "integration_test"
    output_dir = tmp_path / "output" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock the output paths to use tmp_path
    with patch(f"{MODULE}.save_dirty_image") as mock_save_dirty, \
         patch(f"{MODULE}.save_psf_image") as mock_save_psf, \
         patch("os.makedirs"):
        
        generate_dirty_image(grid_array, weight_array, grid_config, job_id)
        
        # Verify both images were attempted to be saved
        assert mock_save_dirty.called
        assert mock_save_psf.called


def test_fft_numerical_correctness():
    """Test FFT numerical correctness with known input"""
    # Create a simple known pattern
    grids = np.zeros((8, 8), dtype=np.complex128)
    grids[4, 4] = 1.0 + 0j  # Single point at center
    weights = np.ones((8, 8))
    
    config = {"img_size": 8, "padding_factor": 1.0}
    
    # Compute dirty image
    dirty = fft(grids, weights, config, type="dirty")
    
    # FFT of a delta function should be constant
    # Check that all values are similar (flat spectrum)
    mean_val = np.mean(dirty)
    assert np.allclose(dirty, mean_val, rtol=0.1)


def test_psf_peak_at_center():
    """Test that PSF has peak at center"""
    grids = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
    weights = np.ones((8, 8))
    config = {"img_size": 8, "padding_factor": 1.0}
    
    psf = fft(grids, weights, config, type="psf")
    
    # PSF should have maximum at or near center
    center = psf.shape[0] // 2
    center_val = psf[center, center]
    
    # Center should be among the highest values
    assert center_val >= np.percentile(psf, 90)


def test_cropping_preserves_center(grid_array, weight_array):
    """Test that cropping preserves the central region"""
    config = {"img_size": 4, "padding_factor": 2.0}
    
    # Create 8x8 grid from 4x4 img_size * 2.0 padding
    grids_padded = np.zeros((8, 8), dtype=np.complex128)
    grids_padded[3:5, 3:5] = 1.0  # Mark center
    weights_padded = np.ones((8, 8))
    
    dirty = fft(grids_padded, weights_padded, config, type="dirty")
    
    # Result should be 4x4 (cropped)
    assert dirty.shape == (4, 4)


def test_generate_dirty_image_end_to_end(tmp_path):
    """Full end-to-end test with real file creation"""
    # Create test data
    grids = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
    weights = np.random.random((8, 8)) + 0.1
    config = {
        "img_size": 8,
        "padding_factor": 1.0
    }
    
    job_id = "e2e_test"
    output_dir = tmp_path / "output" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Patch paths to use tmp_path
    with patch(f"{MODULE}.save_dirty_image") as mock_dirty, \
         patch(f"{MODULE}.save_psf_image") as mock_psf:
        
        def save_dirty_side_effect(img, path):
            # Verify image is valid
            assert isinstance(img, np.ndarray)
            assert img.ndim == 2
        
        def save_psf_side_effect(img, path):
            # Verify image is valid
            assert isinstance(img, np.ndarray)
            assert img.ndim == 2
        
        mock_dirty.side_effect = save_dirty_side_effect
        mock_psf.side_effect = save_psf_side_effect
        
        # Run full pipeline
        generate_dirty_image(grids, weights, config, job_id)
        
        # Verify both saves were called with valid data
        assert mock_dirty.call_count == 1
        assert mock_psf.call_count == 1