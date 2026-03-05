import pytest
import numpy as np
from unittest.mock import MagicMock, Mock
import json
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, ArrayType
)

@pytest.fixture(scope="session")
def spark():
    session = SparkSession.builder \
        .master("local[2]") \
        .appName("BDA Interferometry Tests") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.default.parallelism", "2") \
        .getOrCreate()
    
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()

@pytest.fixture(scope="session")
def visibility_schema():
    return StructType([
        StructField("subms_id", IntegerType(), True),
        StructField("field_id", IntegerType(), True),
        StructField("spw_id", IntegerType(), True),
        StructField("polarization_id", IntegerType(), True),
        StructField("n_channels", IntegerType(), True),
        StructField("n_correlations", IntegerType(), True),
        StructField("antenna1", IntegerType(), True),
        StructField("antenna2", IntegerType(), True),
        StructField("baseline_key", StringType(), True),
        StructField("scan_number", IntegerType(), True),
        StructField("time", DoubleType(), True),
        StructField("interval", DoubleType(), True),
        StructField("exposure", DoubleType(), True),
        StructField("u", DoubleType(), True),
        StructField("v", DoubleType(), True),
        StructField("visibility", ArrayType(ArrayType(ArrayType(DoubleType()))), True),
        StructField("weight", ArrayType(ArrayType(DoubleType())), True),
        StructField("flag", ArrayType(ArrayType(IntegerType())), True),
    ])


@pytest.fixture
def gridded_schema():
    """Schema for gridded visibilities DataFrame"""
    return StructType([
        StructField("u_pix", IntegerType(), False),
        StructField("v_pix", IntegerType(), False),
        StructField("vs_real", DoubleType(), False),
        StructField("vs_imag", DoubleType(), False),
        StructField("weight", DoubleType(), False)
    ])


def _make_visibility(n_channels=1, n_correlations=1, real=1.0, imag=0.0):
    return [[[real, imag] for _ in range(n_correlations)] for _ in range(n_channels)]


def _make_weight(n_channels=1, n_correlations=1, weight=1.0):
    return [[weight for _ in range(n_correlations)] for _ in range(n_channels)]


def _make_flag(n_channels=1, n_correlations=1, flag=0):
    return [[flag for _ in range(n_correlations)] for _ in range(n_channels)]


def _base_row(
    baseline_key="0-1", scan_number=1,
    u=0.0, v=0.0,
    time=0.0, interval=1.0, exposure=1.0, 
    real=1.0, imag=0.0, weight=1.0, flag=0,
    n_channels=1, n_correlations=1, 
):
    return {
        "subms_id": 0,
        "field_id": 0,
        "spw_id": 0,
        "polarization_id": 0,
        "n_channels": n_channels,
        "n_correlations": n_correlations,
        "antenna1": int(baseline_key.split("-")[0]),
        "antenna2": int(baseline_key.split("-")[1]),
        "baseline_key": baseline_key,
        "scan_number": scan_number,
        "time": time,
        "interval": interval,
        "exposure": exposure,
        "u": u,
        "v": v,
        "visibility": _make_visibility(n_channels, n_correlations, real, imag),
        "weight": _make_weight(n_channels, n_correlations, weight),
        "flag": _make_flag(n_channels, n_correlations, flag)
    }


@pytest.fixture()
def df_single_row(spark, visibility_schema):
    rows = [_base_row(time=0.0, u=100.0, v=50.0)]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_two_rows_same_window(spark, visibility_schema):
    rows = [
        _base_row(time=0.0, u=100.0, v=50.0),
        _base_row(time=0.1, u=100.001, v=50.001)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_two_rows_different_window(spark, visibility_schema):
    rows = [
        _base_row(time=0.0, u=0.0, v=0.0),
        _base_row(time=1.0, u=1000.0, v=1000.0)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)

@pytest.fixture()
def df_three_rows_varied(spark, visibility_schema):
    rows = [
        _base_row(time=0.0, u=0.0, v=0.0),
        _base_row(time=1.0, u=1000.0, v=0.0),
        _base_row(time=2.0, u=1000.001, v=0.0)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)

@pytest.fixture()
def df_multi_channel(spark, visibility_schema):
    rows = [
        _base_row(
            time=0.0, u=100.0, v=50.0,
            real=1.0, imag=0.5, weight=1.0,
            n_channels=4, n_correlations=1
        ),
        _base_row(
            time=0.1, u=100.001, v=50.001,
            real=2.0, imag=1.0, weight=1.0,
            n_channels=4, n_correlations=1
        )
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_multi_correlation(spark, visibility_schema):
    """DataFrame with 1 channel, 4 correlations (e.g., XX, XY, YX, YY)"""
    rows = [
        _base_row(
            time=0.0, u=100.0, v=50.0,
            real=1.0, imag=0.5, weight=1.0,
            n_channels=1, n_correlations=4
        ),
        _base_row(
            time=0.1, u=100.001, v=50.001,
            real=2.0, imag=1.0, weight=1.0,
            n_channels=1, n_correlations=4
        )
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_multi_channel_correlation(spark, visibility_schema):
    rows = [
        _base_row(
            time=0.0, u=100.0, v=50.0,
            real=1.0, imag=0.5, weight=1.5,
            n_channels=4, n_correlations=4
        ),
        _base_row(
            time=0.1, u=100.001, v=50.001,
            real=2.0, imag=1.0, weight=2.0,
            n_channels=4, n_correlations=4
        ),
        _base_row(
            time=0.2, u=100.002, v=50.002,
            real=3.0, imag=1.5, weight=2.5,
            n_channels=4, n_correlations=4
        )
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_identity_case(spark, visibility_schema):
    rows = [
        _base_row(
            time=0.0, u=100.0, v=50.0,
            real=3.5, imag=2.1, weight=1.5, exposure=1.0,
            n_channels=2, n_correlations=2
        ),
        _base_row(
            time=0.5, u=150.0, v=75.0,
            real=4.2, imag=-1.3, weight=2.0, exposure=1.5,
            n_channels=2, n_correlations=2
        ),
        _base_row(
            time=1.0, u=200.0, v=100.0,
            real=-2.8, imag=3.7, weight=1.8, exposure=1.2,
            n_channels=2, n_correlations=2
        )
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_constant_visibility(spark, visibility_schema):
    rows = [
        _base_row(
            time=0.0, u=100.0, v=50.0,
            real=3.0, imag=4.0, weight=1.0, exposure=1.0,
            n_channels=2, n_correlations=2
        ),
        _base_row(
            time=0.1, u=100.001, v=50.001,
            real=3.0, imag=4.0, weight=1.0, exposure=1.0,
            n_channels=2, n_correlations=2
        ),
        _base_row(
            time=0.2, u=100.002, v=50.002,
            real=3.0, imag=4.0, weight=1.0, exposure=1.0,
            n_channels=2, n_correlations=2
        )
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_constant_unequal_weights(spark, visibility_schema):
    rows = [
        _base_row(
            time=0.0, u=100.0, v=50.0,
            real=2.0, imag=1.0, weight=0.5, exposure=1.0
        ),
        _base_row(
            time=0.1, u=100.001, v=50.001,
            real=2.0, imag=1.0, weight=1.5, exposure=1.0
        ),
        _base_row(
            time=0.2, u=100.002, v=50.002,
            real=2.0, imag=1.0, weight=2.0, exposure=1.0
        )
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_constant_multi_channel(spark, visibility_schema):
    def _make_vis_multi():
        return [
            [[1.0, 1.0]],  # channel 0
            [[2.0, 2.0]],  # channel 1
            [[3.0, 3.0]]   # channel 2
        ]
    
    rows = [
        {
            **_base_row(time=0.0, u=100.0, v=50.0, n_channels=3),
            "visibility": _make_vis_multi()
        },
        {
            **_base_row(time=0.1, u=100.001, v=50.001, n_channels=3),
            "visibility": _make_vis_multi()
        },
        {
            **_base_row(time=0.2, u=100.002, v=50.002, n_channels=3),
            "visibility": _make_vis_multi()
        }
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_single_window_multi_rows(spark, visibility_schema):
    rows = [
        _base_row(
            time=0.0, u=100.0, v=50.0,
            real=1.0, imag=0.0, weight=1.0, exposure=1.0
        ),
        _base_row(
            time=0.01, u=100.0001, v=50.0001,
            real=1.0, imag=0.0, weight=1.0, exposure=1.0
        ),
        _base_row(
            time=0.02, u=100.0002, v=50.0002,
            real=1.0, imag=0.0, weight=1.0, exposure=1.0
        )
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_zero_weight_excluded(spark, visibility_schema):
    rows = [
        _base_row(time=0.0, u=100.0, v=50.0, real=999.0, imag=999.0, weight=0.0),
        _base_row(time=0.1, u=100.001, v=50.001, real=2.0, imag=1.0, weight=1.0),
        _base_row(time=0.2, u=100.002, v=50.002, real=4.0, imag=2.0, weight=2.0)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_all_zero_weights(spark, visibility_schema):
    rows = [
        _base_row(time=0.0, u=100.0, v=50.0, real=5.0, imag=3.0, weight=0.0),
        _base_row(time=0.1, u=100.001, v=50.001, real=10.0, imag=7.0, weight=0.0),
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_partially_flagged(spark, visibility_schema):
    rows = [
        _base_row(time=0.0, u=100.0, v=50.0, real=999.0, imag=999.0, weight=1.0, flag=1),
        _base_row(time=0.1, u=100.001, v=50.001, real=2.0, imag=1.0, weight=1.0, flag=0),
        _base_row(time=0.2, u=100.002, v=50.002, real=4.0, imag=2.0, weight=1.0, flag=0)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_all_flagged(spark, visibility_schema):
    rows = [
        _base_row(time=0.0, u=100.0, v=50.0, real=5.0, imag=3.0, weight=1.0, flag=1),
        _base_row(time=0.1, u=100.001, v=50.001, real=10.0, imag=7.0, weight=1.0, flag=1)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_mixed_flags(spark, visibility_schema):
    rows = [
        _base_row(time=0.0, u=100.0, v=50.0, real=6.0, imag=3.0, weight=2.0, flag=0),
        _base_row(time=0.1, u=100.001, v=50.0, real=999.0, imag=999.0, weight=5.0, flag=1),
        _base_row(time=0.2, u=100.002, v=50.0, real=8.0, imag=4.0, weight=3.0, flag=0)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_multi_baselines(spark, visibility_schema):
    """Multiple baselines with multiple rows each for determinism testing"""
    rows = [
        # Baseline 0-1
        _base_row(baseline_key="0-1", time=0.0, u=100.0, v=50.0, real=1.0, imag=1.0),
        _base_row(baseline_key="0-1", time=0.1, u=100.001, v=50.001, real=2.0, imag=2.0),
        # Baseline 0-2
        _base_row(baseline_key="0-2", time=0.0, u=200.0, v=100.0, real=3.0, imag=3.0),
        _base_row(baseline_key="0-2", time=0.1, u=200.001, v=100.001, real=4.0, imag=4.0),
        # Baseline 1-2
        _base_row(baseline_key="1-2", time=0.0, u=150.0, v=75.0, real=5.0, imag=5.0),
        _base_row(baseline_key="1-2", time=0.1, u=150.001, v=75.001, real=6.0, imag=6.0)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def df_unordered_rows(spark, visibility_schema):
    """Rows intentionally out of time order to test sorting determinism"""
    rows = [
        _base_row(time=0.2, u=100.002, v=50.002, real=3.0, imag=3.0, weight=1.0),
        _base_row(time=0.0, u=100.0, v=50.0, real=1.0, imag=1.0, weight=1.0),
        _base_row(time=0.1, u=100.001, v=50.001, real=2.0, imag=2.0, weight=1.0)
    ]
    return spark.createDataFrame(rows, schema=visibility_schema)


@pytest.fixture()
def bda_config():
    return {
        "decorr_factor": 0.95,
        "lambda_ref": 0.1,
        "fov": 1.0
    }


# Fixtures for simulation tests

@pytest.fixture
def mock_interferometer():
    interferometer = MagicMock()
    positions = MagicMock()
    positions.shape = (4, 3)
    interferometer.antenna_array.positions = positions
    diameters_mock = MagicMock()
    diameters_mock.compute.return_value = np.array([12.0] * 4)
    interferometer.antenna_array.diameters = diameters_mock
    return interferometer


@pytest.fixture
def base_sim_config():
    return {
        "interferometer": "OTHER",
        "freq_min": 1.0,
        "freq_max": 2.0,
        "n_chans": 4,
        "observation_time": 3600,
        "declination": "-30d",
        "integration_time": 10,
        "spectral_index": -0.7,
        "flux_density": 1.0,
    }


@pytest.fixture
def ska_sim_config():
    return {
        "interferometer": "SKA",
        "array_type": "MID",
        "assembly": "AA2",
        "freq_min": 350.0,
        "freq_max": 1050.0,
        "n_chans": 4,
        "observation_time": 3600,
        "declination": "-30d",
        "integration_time": 10,
        "spectral_index": -0.7,
        "source_path": "/fake/source.fits",
    }


# Fixture for extraction tests

NROWS = 5
NCHAN = 4
NCORR = 2

@pytest.fixture
def valid_data():
    rng = np.random.default_rng(0)
    return (
        np.arange(NROWS, dtype=np.int32),                              # 0  antenna1
        np.arange(NROWS, dtype=np.int32) + 1,                         # 1  antenna2
        np.ones(NROWS, dtype=np.int32),                                # 2  scan_number
        np.linspace(0.0, 1.0, NROWS),                                  # 3  time
        np.ones(NROWS) * 10.0,                                         # 4  exposure
        np.ones(NROWS) * 10.0,                                         # 5  interval
        rng.random((NROWS, 3)),                                        # 6  uvw
        rng.random((NROWS, NCHAN, NCORR)) + 1j * rng.random((NROWS, NCHAN, NCORR)),  # 7  vis
        rng.random((NROWS, NCHAN)),                                    # 8  weight
        np.zeros((NROWS, NCHAN), dtype=np.bool_),                      # 9  flag
        np.zeros(NROWS),                                               # 10 unused
        np.zeros(NROWS),                                               # 11 unused
    )


@pytest.fixture
def valid_metadata():
    return {
        "subms_id": 0,
        "field_id": 1,
        "spw_id": 2,
        "polarization_id": 0,
        "n_channels": NCHAN,
        "n_correlations": NCORR,
    }


@pytest.fixture
def mock_producer():
    producer = MagicMock()
    producer.send.return_value.get.return_value = None
    return producer


@pytest.fixture
def mock_dataset():
    row_chunks = (NROWS,)

    def _make_dask_array(ndim):
        arr = MagicMock()
        arr.chunks = (row_chunks,) + ((-1,),) * (ndim - 1)
        arr.shape = (NROWS,) + (4,) * (ndim - 1)
        arr.rechunk.return_value = arr
        return arr

    ds = MagicMock()
    ds.nrows = NROWS

    ds.data.data = _make_dask_array(3)
    ds.data.data.chunks = (row_chunks, (4,), (2,))
    ds.data.shape = (NROWS, NCHAN, NCORR)

    ds.antenna1.data = _make_dask_array(1)
    ds.antenna2.data = _make_dask_array(1)
    ds.scan_number.data = _make_dask_array(1)
    ds.time.data = _make_dask_array(1)
    ds.dataset.EXPOSURE.data = _make_dask_array(1)
    ds.dataset.INTERVAL.data = _make_dask_array(1)
    ds.uvw.data = _make_dask_array(2)
    ds.weight.data = _make_dask_array(2)
    ds.flag.data = _make_dask_array(2)

    return ds


@pytest.fixture
def mock_subms():
    """Mock de SubMS con los campos de metadata requeridos."""
    subms = MagicMock()
    subms.id = 0
    subms.field_id = 1
    subms.spw_id = 2
    subms.polarization_id = 0
    return subms


# Fixtures for gridding tests

IMG_SIZE = 8
PADDING_FACTOR = 1.0
PADDED_SIZE = IMG_SIZE  # IMG_SIZE * PADDING_FACTOR


@pytest.fixture
def grid_config():
    """Create a basic grid configuration"""
    return {
        "img_size": 8,
        "padding_factor": 1.0,
        "cellsize": 1e-5,
        "chan_freq": [1.4e9],
        "corrs_string": "XX,YY"
    }


@pytest.fixture
def grid_config_with_padding():
    """Create grid configuration with padding > 1"""
    return {
        "img_size": 8,
        "padding_factor": 2.0,
        "cellsize": 1e-5,
        "chan_freq": [1.4e9],
        "corrs_string": "XX,YY"
    }


@pytest.fixture
def grid_array():
    """Create a sample 8x8 complex grid for testing"""
    return np.random.random((8, 8)) + 1j * np.random.random((8, 8))


@pytest.fixture
def weight_array():
    """Create a sample 8x8 weight array for testing"""
    return np.random.random((8, 8)) + 0.1  # Add offset to avoid zeros

@pytest.fixture
def image_2d():
    """Create a sample 2D image for testing save functions"""
    return np.random.random((8, 8))


@pytest.fixture()
def df_in_range(spark, visibility_schema):
    row = _base_row(time=0.0, u=0.0, v=0.0)
    return spark.createDataFrame([row], schema=visibility_schema)


@pytest.fixture
def sample_row():
    """Create a pandas Series mimicking a row from a DataFrame"""
    return pd.Series({
        'u': 100.0,
        'v': 50.0,
        'visibility': [[[1.0, 0.5], [2.0, 1.0]]],  # 1 channel, 2 correlations
        'weight': [[1.0, 1.0]],
        'flag': [[0, 0]],
        'n_channels': 1,
        'n_correlations': 2,
        'is_hermitian': False
    })
    

@pytest.fixture
def df_gridding(spark, gridded_schema):
    """Create a sample gridded DataFrame for testing"""
    rows = [
        (0, 0, 1.0, 0.0, 1.0),
        (1, 1, 2.0, 1.0, 1.5),
        (2, 2, 3.0, 2.0, 2.0),
        (3, 3, 4.0, 3.0, 2.5)
    ]
    return spark.createDataFrame(rows, schema=gridded_schema)


@pytest.fixture
def _write_grid_config(tmp_path, grid_config):
    path = tmp_path / "grid_config.json"
    path.write_text(json.dumps({
        "img_size": grid_config["img_size"],
        "padding_factor": grid_config["padding_factor"],
        "cellsize": grid_config["cellsize"],
    }), encoding="utf-8")
    return path

# Fixture for testing grid config updates

@pytest.fixture
def tmp_antenna_config(tmp_path):
    """Create a temporary antenna configuration file"""
    config_file = tmp_path / "antenna.cfg"
    config_file.write_text("# Dummy antenna config")
    return str(config_file)


@pytest.fixture
def tmp_sim_config(tmp_path):
    """Create a temporary simulation configuration file"""
    config_file = tmp_path / "sim_config.json"
    sim_config = {
        "interferometer": "VLA",
        "freq_min": 1.0,
        "freq_max": 2.0,
        "n_chans": 4
    }
    config_file.write_text(json.dumps(sim_config))
    return str(config_file)


@pytest.fixture
def mock_dataset_producer():
    """Create a mock dataset with all required attributes"""
    dataset = Mock()
    
    # spws attributes
    dataset.spws = Mock()
    dataset.spws.lambda_ref = 0.21  # meters
    dataset.spws.dataset = [Mock()]
    
    # First SPW with CHAN_FREQ
    first_spw = dataset.spws.dataset[0]
    mock_chan_freq = Mock()
    # Return numpy array instead of list
    mock_values = Mock()
    mock_values.values = np.array([[1.4e9, 1.5e9, 1.6e9]])
    mock_chan_freq.compute.return_value = mock_values
    first_spw.CHAN_FREQ = mock_chan_freq
    
    # Other attributes
    dataset.theo_resolution = 1e-5
    dataset.polarization = Mock()
    dataset.polarization.corrs_string = "XX,YY"
    
    # ms_list for streaming
    dataset.ms_list = []
    
    return dataset


@pytest.fixture
def mock_interferometer_producer():
    """Create a mock interferometer with all required attributes"""
    interferometer = Mock()
    interferometer.antenna_array = Mock()
    
    # Mock get_array_extent to return (min, max)
    mock_extent = Mock()
    mock_extent.compute.return_value = 25.0  # max diameter
    interferometer.antenna_array.get_array_extent.return_value = (0, mock_extent)
    
    # Mock get_median_antenna_separation
    mock_separation = Mock()
    mock_separation.compute.return_value = 100.0
    interferometer.antenna_array.get_median_antenna_separation.return_value = mock_separation
    
    return interferometer


@pytest.fixture
def tmp_bda_config(tmp_path):
    """Create a temporary BDA configuration file"""
    config_file = tmp_path / "bda_config.json"
    bda_config = {
        "fov_strategy": "DERIVED",
        "threshold_strategy": "DERIVED"
    }
    config_file.write_text(json.dumps(bda_config))
    return str(config_file)


@pytest.fixture
def tmp_grid_config(tmp_path):
    """Create a temporary grid configuration file"""
    config_file = tmp_path / "grid_config.json"
    grid_config = {
        "cellsize_strategy": "DERIVED",
        "img_size": 512,
        "padding_factor": 2.0
    }
    config_file.write_text(json.dumps(grid_config))
    return str(config_file)
