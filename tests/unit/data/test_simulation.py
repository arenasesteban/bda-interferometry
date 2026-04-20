import pytest
import numpy as np
import astropy.units as u
from unittest.mock import MagicMock, patch

from data.simulation import (
    load_antenna_configuration,
    filter_antenna_configuration,
    configure_observation,
    generate_point_sources,
    simulate_dataset,
    generate_dataset
)

MODULE = "src.data.simulation"


# Test cases for load_antenna_configuration function

def test_raises_if_path_is_none():
    with pytest.raises(ValueError, match="cannot be None"):
        load_antenna_configuration(None)


def test_raises_if_path_is_empty_string():
    with pytest.raises(ValueError, match="cannot be None"):
        load_antenna_configuration("")

def test_raises_if_path_does_not_exist(tmp_path):
    missing_path = tmp_path / "nonexistent.cfg"
    with pytest.raises(FileNotFoundError, match="Antenna config not found"):
        load_antenna_configuration(str(missing_path))


def test_returns_interferometer_on_success(tmp_path):
    config_file = tmp_path / "antenna.cfg"
    config_file.write_text("dummy", encoding="utf-8")

    mock_interferometer = MagicMock()
    mock_io = MagicMock()
    mock_io.read.return_value = mock_interferometer

    with patch(f"{MODULE}.AntennaConfigurationIo", return_value=mock_io) as mock_cls:
        result = load_antenna_configuration(str(config_file))
        mock_cls.assert_called_once_with(input_name=str(config_file))
        mock_io.read.assert_called_once()
        assert result is mock_interferometer


# Test cases for filter_antenna_configuration funtion

def test_raises_if_interferometer_is_none():
    with pytest.raises(ValueError, match="Interferometer cannot be None"):
        filter_antenna_configuration(None, "MID", "AA2")


def test_raises_if_assembly_is_none():
    with pytest.raises(ValueError, match="Assembly cannot be None or empty"):
        filter_antenna_configuration(MagicMock(), "MID", None)


def test_raises_if_assembly_is_empty():
    with pytest.raises(ValueError, match="Assembly cannot be None or empty"):
        filter_antenna_configuration(MagicMock(), "MID", "")


def test_aa4_skips_filtering(mock_interferometer):
    with patch(f"{MODULE}.get_ska_station_list") as mock_get:
        filter_antenna_configuration(mock_interferometer, "MID", "AA4")
        mock_get.assert_not_called()
        mock_interferometer.antenna_array.filter_by_ids.assert_not_called()


def test_non_aa4_calls_filter(mock_interferometer):
    fake_ids = [0, 1, 2]
    with patch(f"{MODULE}.get_ska_station_list", return_value=fake_ids) as mock_get:
        filter_antenna_configuration(mock_interferometer, "MID", "AA2")
        mock_get.assert_called_once_with("MID", "AA2")
        mock_interferometer.antenna_array.filter_by_ids.assert_called_once_with(
            fake_ids, inplace=True
        )


# Test cases for configure_observation function

def test_raises_if_interferometer_is_none(base_sim_config):
    with pytest.raises(ValueError, match="Interferometer cannot be None"):
        configure_observation(None, base_sim_config)


def test_raises_if_sim_config_is_none(mock_interferometer):
    with pytest.raises(ValueError, match="Simulation configuration cannot be None"):
        configure_observation(mock_interferometer, None)


def test_raises_if_required_param_missing(mock_interferometer, base_sim_config):
    del base_sim_config["freq_min"]
    with pytest.raises(KeyError, match="freq_min"):
        configure_observation(mock_interferometer, base_sim_config)


def test_ska_branch_calls_configure_with_frequencies(mock_interferometer, ska_sim_config):
    with patch(f"{MODULE}.Time"), patch(f"{MODULE}.Angle"):
        configure_observation(mock_interferometer, ska_sim_config)
    call_kwargs = mock_interferometer.configure_observation.call_args.kwargs
    assert "frequencies"    in call_kwargs
    assert "reference_time" in call_kwargs
    assert call_kwargs["frequency_step"] is None


def test_non_ska_branch_calls_configure_with_min_max(mock_interferometer, base_sim_config):
    with patch(f"{MODULE}.Angle"):
        configure_observation(mock_interferometer, base_sim_config)
    call_kwargs = mock_interferometer.configure_observation.call_args.kwargs
    assert "min_frequency"      in call_kwargs
    assert "max_frequency"      in call_kwargs
    assert call_kwargs["hour_angle"] == "transit"


def test_returns_freq_and_ref_freq(mock_interferometer, base_sim_config):
    with patch(f"{MODULE}.Angle"):
        freq, ref_freq = configure_observation(mock_interferometer, base_sim_config)
    assert freq is not None
    assert ref_freq is not None


# Test cases for generate_point_sources function

def _freq_ghz():
    return np.linspace(1.0, 2.0, 4) * u.GHz


def _freq_mhz():
    return np.linspace(350.0, 1050.0, 4) * u.MHz


def test_raises_if_ref_freq_is_none(base_sim_config, mock_interferometer):
    with pytest.raises(ValueError, match="Reference frequency cannot be None"):
        generate_point_sources(None, _freq_ghz(), base_sim_config, mock_interferometer)


def test_raises_if_freq_is_none(base_sim_config, mock_interferometer):
    with pytest.raises(ValueError, match="Frequency array cannot be None"):
        generate_point_sources(1.5 * u.GHz, None, base_sim_config, mock_interferometer)


def test_raises_if_sim_config_is_none(mock_interferometer):
    with pytest.raises(ValueError, match="Simulation configuration cannot be None"):
        generate_point_sources(1.5 * u.GHz, _freq_ghz(), None, mock_interferometer)


def test_raises_if_interferometer_is_none(base_sim_config):
    with pytest.raises(ValueError, match="Interferometer cannot be None"):
        generate_point_sources(1.5 * u.GHz, _freq_ghz(), base_sim_config, None)


def test_ska_raises_if_source_path_missing(ska_sim_config):
    del ska_sim_config["source_path"]
    with pytest.raises(KeyError, match="source_path"):
        generate_point_sources(700 * u.MHz, _freq_mhz(), ska_sim_config, MagicMock())


def test_ska_raises_if_source_file_not_found(ska_sim_config):
    ska_sim_config["source_path"] = "/nonexistent/source.fits"
    with pytest.raises(FileNotFoundError, match="Source file not found"):
        generate_point_sources(700 * u.MHz, _freq_mhz(), ska_sim_config, MagicMock())


def test_ska_returns_sources_list(tmp_path, ska_sim_config):
    fake_fits_path = str(tmp_path / "source.fits")
    open(fake_fits_path, "w").close()
    ska_sim_config["source_path"] = fake_fits_path

    mock_image = MagicMock()
    mock_image.cellsize = [None, MagicMock()]
    mock_image.cellsize[1].to.return_value.value = 1e-5
    mock_image.shape = (1, 1, 64, 64)
    
    mock_fits = MagicMock()
    mock_fits.read.return_value = mock_image
    
    mock_nps = MagicMock()
    mock_ps = MagicMock()

    n_sources = 3
    
    with patch(f"{MODULE}.FITS", return_value=mock_fits), \
         patch(f"{MODULE}.NonParametricSource", return_value=mock_nps), \
         patch(f"{MODULE}.PointSource", return_value=mock_ps):
        
        with patch("numpy.random.randint") as mock_randint:

            mock_randint.side_effect = [
                np.array([n_sources]),           # n_sources
                np.array([10, 20, -15]),         # pixel_coords_l
                np.array([-5, 15, 10])           # pixel_coords_m
            ]
            
            sources = generate_point_sources(
                700 * u.MHz, _freq_mhz(), ska_sim_config, MagicMock()
            )

    assert len(sources) == n_sources + 1
    assert sources[0] is mock_nps
    assert all(sources[i] is mock_ps for i in range(1, len(sources)))



def test_non_ska_raises_if_flux_density_missing(base_sim_config, mock_interferometer):
    del base_sim_config["flux_density"]
    with pytest.raises(KeyError, match="flux_density"):
        generate_point_sources(
            1.5 * u.GHz, _freq_ghz(), base_sim_config, mock_interferometer
        )

def test_non_ska_returns_point_sources(base_sim_config, mock_interferometer):
    mock_ps = MagicMock()

    with patch(f"{MODULE}.PointSource", return_value=mock_ps), \
            patch("numpy.random.randint", return_value=np.array([2])), \
            patch("numpy.random.uniform", return_value=np.array([1e-5, 2e-5])):
        sources = generate_point_sources(
            1.5 * u.GHz, _freq_ghz(), base_sim_config, mock_interferometer
        )

    assert len(sources) == 2
    assert all(s is mock_ps for s in sources)


# Test cases for simulate_dataset function

def test_raises_if_interferometer_is_none():
    with pytest.raises(ValueError, match="Interferometer cannot be None"):
        simulate_dataset(None, MagicMock())

def test_raises_if_sources_is_none():
    with pytest.raises(ValueError, match="Sources cannot be None"):
        simulate_dataset(MagicMock(), None)

def test_returns_dataset_on_success():
    mock_dataset = MagicMock()
    mock_sim     = MagicMock()
    mock_sim.simulate.return_value = mock_dataset

    with patch(f"{MODULE}.Simulator", return_value=mock_sim) as mock_cls:
        result = simulate_dataset(interferometer := MagicMock(), sources := MagicMock())
        mock_cls.assert_called_once_with(interferometer=interferometer, sources=sources)
        mock_sim.simulate.assert_called_once()
        assert result is mock_dataset


# Test cases for generate_dataset function

def _patch_all(interferometer, freq, ref_freq, sources, dataset):
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch(f"{MODULE}.load_antenna_configuration",  return_value=interferometer))
    stack.enter_context(patch(f"{MODULE}.filter_antenna_configuration"))
    stack.enter_context(patch(f"{MODULE}.configure_observation",       return_value=(freq, ref_freq)))
    stack.enter_context(patch(f"{MODULE}.generate_point_sources",      return_value=sources))
    stack.enter_context(patch(f"{MODULE}.CompositeSource",             return_value=MagicMock()))
    stack.enter_context(patch(f"{MODULE}.simulate_dataset",            return_value=dataset))
    return stack


def test_non_ska_full_pipeline(mock_interferometer, base_sim_config):
    freq    = np.linspace(1.0, 2.0, 4) * u.GHz
    dataset = MagicMock()

    with _patch_all(mock_interferometer, freq, 1.5 * u.GHz, [MagicMock()], dataset):
        result_dataset, result_interf = generate_dataset("/fake/path.cfg", base_sim_config)

    assert result_dataset is dataset
    assert result_interf  is mock_interferometer


def test_ska_raises_if_array_type_missing(mock_interferometer, ska_sim_config):
    del ska_sim_config["array_type"]

    with patch(f"{MODULE}.load_antenna_configuration", return_value=mock_interferometer):
        with pytest.raises(KeyError, match="array_type"):
            generate_dataset("/fake/path.cfg", ska_sim_config)


def test_ska_full_pipeline(mock_interferometer, ska_sim_config):
    freq    = np.linspace(350.0, 1050.0, 4) * u.MHz
    dataset = MagicMock()

    with _patch_all(mock_interferometer, freq, 700 * u.MHz, [MagicMock()], dataset):
        with patch(f"{MODULE}.filter_antenna_configuration") as mock_filter:
            result_dataset, result_interf = generate_dataset("/fake/path.cfg", ska_sim_config)
            mock_filter.assert_called_once_with(
                mock_interferometer,
                ska_sim_config["array_type"],
                ska_sim_config["assembly"]
            )

    assert result_dataset is dataset
    assert result_interf  is mock_interferometer