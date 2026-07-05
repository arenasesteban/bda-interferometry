import json
import pytest
import numpy as np
import dask.array as da
import msgpack
import os
import sys
from pathlib import Path
import io


from types import SimpleNamespace
from unittest.mock import MagicMock
from types import SimpleNamespace
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    BinaryType,
    StringType,
    StructType,
    StructField,
    ArrayType,
)

from services.consumer_service import define_visibility_schema
from core.bda.bda_config import load_bda_config


# Fixtures test simulation-extraction

@pytest.fixture
def real_antenna_config_path():
    path = Path("conf/antenna/alma.cfg")

    if not path.exists():
        pytest.skip(f"No se encontró archivo de antenas: {path}")

    return str(path)


@pytest.fixture
def real_simulation_config():
    path = Path("conf/runtime/simulation/alma-band-01.json")

    if not path.exists():
        pytest.skip(f"No se encontró archivo de simulación: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    

# Fixture test producer-extraction

@pytest.fixture
def publication_topic():
    return "visibility-stream"


@pytest.fixture
def publication_bootstrap_servers():
    return "localhost:9092"


@pytest.fixture
def publication_run_id():
    return "integration-publication-test"


@pytest.fixture
def publication_nrows():
    return 20


@pytest.fixture
def publication_nchan():
    return 4


@pytest.fixture
def publication_ncorr():
    return 2


@pytest.fixture
def publication_visibility_dataset(publication_nrows, publication_nchan, publication_ncorr):
    rng = np.random.default_rng(42)

    antenna1 = np.array(
        [0, 0, 1, 1, 2] * 4,
        dtype=np.int32,
    )

    antenna2 = np.array(
        [1, 2, 2, 3, 3] * 4,
        dtype=np.int32,
    )

    scan_number = np.ones(publication_nrows, dtype=np.int32)
    time = np.linspace(0.0, 10.0, publication_nrows)
    exposure = np.ones(publication_nrows) * 10.0
    interval = np.ones(publication_nrows) * 10.0

    uvw = rng.random((publication_nrows, 3)).astype(np.float32)

    visibilities = (
        rng.random((publication_nrows, publication_nchan, publication_ncorr))
        + 1j * rng.random((publication_nrows, publication_nchan, publication_ncorr))
    ).astype(np.complex64)

    weights = rng.random((publication_nrows, publication_nchan)).astype(np.float32)
    flags = np.zeros((publication_nrows, publication_nchan), dtype=np.bool_)

    row_chunks = (publication_nrows,)

    dataset = SimpleNamespace()
    dataset.nrows = publication_nrows

    dataset.antenna1 = SimpleNamespace(
        data=da.from_array(antenna1, chunks=row_chunks)
    )

    dataset.antenna2 = SimpleNamespace(
        data=da.from_array(antenna2, chunks=row_chunks)
    )

    dataset.scan_number = SimpleNamespace(
        data=da.from_array(scan_number, chunks=row_chunks)
    )

    dataset.time = SimpleNamespace(
        data=da.from_array(time, chunks=row_chunks)
    )

    dataset.dataset = SimpleNamespace(
        EXPOSURE=SimpleNamespace(
            data=da.from_array(exposure, chunks=row_chunks)
        ),
        INTERVAL=SimpleNamespace(
            data=da.from_array(interval, chunks=row_chunks)
        ),
    )

    dataset.uvw = SimpleNamespace(
        data=da.from_array(uvw, chunks=(publication_nrows, 3))
    )

    dataset.data = SimpleNamespace(
        data=da.from_array(
            visibilities,
            chunks=(publication_nrows, publication_nchan, publication_ncorr),
        ),
        shape=visibilities.shape,
    )

    dataset.weight = SimpleNamespace(
        data=da.from_array(weights, chunks=(publication_nrows, publication_nchan))
    )

    dataset.flag = SimpleNamespace(
        data=da.from_array(flags, chunks=(publication_nrows, publication_nchan))
    )

    return dataset


@pytest.fixture
def publication_antennas():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [30.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )

    return SimpleNamespace(
        positions=da.from_array(positions, chunks=(4, 3))
    )


@pytest.fixture
def publication_subms(publication_visibility_dataset):
    return SimpleNamespace(
        id=0,
        field_id=1,
        spw_id=2,
        polarization_id=0,
        visibilities=publication_visibility_dataset,
    )


@pytest.fixture
def publication_dataset(publication_subms, publication_antennas):
    return SimpleNamespace(
        ms_list=[publication_subms],
        antenna=publication_antennas,
    )


@pytest.fixture
def mock_kafka_producer():
    producer = MagicMock()
    producer.send.return_value.get.return_value = None
    return producer


# Fixtures test consumer-ingest

def _find_project_root():
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "src" / "services" / "consumer_service.py").exists():
            return parent

    raise RuntimeError("No se pudo encontrar la raíz del proyecto con src/services/consumer_service.py")


@pytest.fixture(scope="session")
def spark():
    project_root = _find_project_root()
    src_path = str(project_root / "src")

    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    previous_pythonpath = os.environ.get("PYTHONPATH", "")

    pythonpath_parts = [src_path]

    if previous_pythonpath:
        pythonpath_parts.append(previous_pythonpath)

    pythonpath = os.pathsep.join(pythonpath_parts)

    os.environ["PYTHONPATH"] = pythonpath
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("BDA Interferometry Integration Tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.executorEnv.PYTHONPATH", pythonpath)
        .config("spark.driverEnv.PYTHONPATH", pythonpath)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


@pytest.fixture
def consumer_nrows():
    return 3


@pytest.fixture
def consumer_nchan():
    return 4


@pytest.fixture
def consumer_ncorr():
    return 2


@pytest.fixture
def consumer_message_metadata(consumer_nchan, consumer_ncorr):
    return {
        "schema": "visibilities_blocks",
        "message_id": "0-0",
        "subms_id": 0,
        "field_id": 1,
        "spw_id": 2,
        "polarization_id": 0,
        "n_channels": consumer_nchan,
        "n_correlations": consumer_ncorr,
    }


@pytest.fixture
def consumer_payload_npz_bytes(consumer_nrows, consumer_nchan, consumer_ncorr):
    rng = np.random.default_rng(42)

    antenna1 = np.array([0, 2, 3], dtype=np.int32)
    antenna2 = np.array([1, 0, 1], dtype=np.int32)

    arrays = {
        "antenna1": antenna1,
        "antenna2": antenna2,
        "scan_number": np.ones(consumer_nrows, dtype=np.int32),
        "time": np.linspace(0.0, 2.0, consumer_nrows),
        "exposure": np.ones(consumer_nrows) * 10.0,
        "interval": np.ones(consumer_nrows) * 10.0,
        "u": rng.random(consumer_nrows),
        "v": rng.random(consumer_nrows),
        "w": rng.random(consumer_nrows),
        "visibilities": rng.random(
            (consumer_nrows, consumer_nchan, consumer_ncorr, 2)
        ),
        "weights": rng.random(
            (consumer_nrows, consumer_nchan, consumer_ncorr)
        ),
        "flags": np.zeros(
            (consumer_nrows, consumer_nchan, consumer_ncorr),
            dtype=np.int8,
        ),
        "baseline_length": np.array([10.0, 20.0, 30.0]),
    }

    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()


@pytest.fixture
def kafka_headers(consumer_message_metadata):
    return [
        Row(
            key="schema",
            value=b"visibilities_blocks",
        ),
        Row(
            key="metadata",
            value=msgpack.packb(consumer_message_metadata, use_bin_type=True),
        ),
    ]


@pytest.fixture
def kafka_end_headers():
    metadata = {
        "schema": "visibilities_blocks",
        "total_blocks": 1,
    }

    return [
        Row(
            key="schema",
            value=b"visibilities_blocks",
        ),
        Row(
            key="metadata",
            value=msgpack.packb(metadata, use_bin_type=True),
        ),
    ]


@pytest.fixture
def kafka_like_batch_df(
    spark,
    consumer_payload_npz_bytes,
    kafka_headers,
    kafka_end_headers,
):
    schema = StructType([
        StructField("key", BinaryType(), True),
        StructField("value", BinaryType(), True),
        StructField(
            "headers",
            ArrayType(
                StructType([
                    StructField("key", StringType(), True),
                    StructField("value", BinaryType(), True),
                ])
            ),
            True,
        ),
    ])

    rows = [
        (
            b"message-0-0",
            consumer_payload_npz_bytes,
            kafka_headers,
        ),
        (
            b"__END__",
            None,
            kafka_end_headers,
        ),
    ]

    return spark.createDataFrame(rows, schema=schema)


# Fixtures test consumer-bda

# ---------------------------------------------------------------------
# Fixtures test consumer-bda
# ---------------------------------------------------------------------

@pytest.fixture
def bda_num_partitions():
    return 2


@pytest.fixture
def bda_nchan():
    return 2


@pytest.fixture
def bda_ncorr():
    return 2


@pytest.fixture
def bda_active_config_path(tmp_path):
    """
    Configuración con reducción BDA activa.

    decorr_factor < 1.0 fuerza la ruta:
    apply_bda -> process_rows -> assign_temporal_window -> average_by_window
    """
    config = {
        "fov": 1.0,
        "decorr_factor": 0.95,
        "lambda_ref": 1.0,
        "theta_max": 1e-4,
        "threshold": 1000.0,
    }

    path = tmp_path / "bda_active_config.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file)

    return str(path)


@pytest.fixture
def bda_passthrough_config_path(tmp_path):
    """
    Configuración sin reducción BDA.

    decorr_factor = 1.0 valida la rama donde apply_bda retorna
    el DataFrame original y no genera df_windowed.
    """
    config = {
        "fov": 1.0,
        "decorr_factor": 1.0,
        "lambda_ref": 1.0,
        "theta_max": 1e-4,
        "threshold": 1000.0,
    }

    path = tmp_path / "bda_passthrough_config.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file)

    return str(path)


@pytest.fixture
def integration_bda_active_config(bda_active_config_path):
    return load_bda_config(bda_active_config_path)


@pytest.fixture
def integration_bda_passthrough_config(bda_passthrough_config_path):
    return load_bda_config(bda_passthrough_config_path)


@pytest.fixture(scope="session")
def require_functional_pyarrow():
    """
    Verifica que el entorno actual pueda ejecutar pandas_udf.

    Si PyArrow está roto por incompatibilidad binaria con NumPy, el test
    de reducción activa se omite con una razón explícita. Esto evita marcar
    como fallo lógico del BDA un problema de entorno.
    """
    try:
        import numpy
        import pandas
        import pyarrow
    except Exception as exc:
        pytest.skip(
            "PyArrow/Pandas no está funcional en el entorno actual, "
            "por lo que no se puede ejecutar la ruta BDA con pandas_udf. "
            f"Detalle: {exc!r}"
        )

    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
    }


def _visibility(real_value, imag_value, nchan=2, ncorr=2):
    return [
        [
            [float(real_value), float(imag_value)]
            for _ in range(ncorr)
        ]
        for _ in range(nchan)
    ]


def _weights(value, nchan=2, ncorr=2):
    return [
        [
            float(value)
            for _ in range(ncorr)
        ]
        for _ in range(nchan)
    ]


def _flags(value=0, nchan=2, ncorr=2):
    return [
        [
            int(value)
            for _ in range(ncorr)
        ]
        for _ in range(nchan)
    ]


@pytest.fixture
def bda_scientific_rows(bda_nchan, bda_ncorr):
    rows = []

    baselines = [
        {
            "baseline_key": "0-1",
            "antenna1": 0,
            "antenna2": 1,
            "baseline_length": 10.0,
            "u0": 100.0,
            "v0": 200.0,
            "real0": 1.0,
        },
        {
            "baseline_key": "2-3",
            "antenna1": 2,
            "antenna2": 3,
            "baseline_length": 20.0,
            "u0": 300.0,
            "v0": 400.0,
            "real0": 10.0,
        },
    ]

    for baseline in baselines:
        for i in range(4):
            rows.append({
                "message_id": "0-0",
                "subms_id": 0,
                "field_id": 1,
                "spw_id": 2,
                "polarization_id": 0,

                "baseline_key": baseline["baseline_key"],
                "antenna1": baseline["antenna1"],
                "antenna2": baseline["antenna2"],
                "scan_number": 1,

                "exposure": 10.0,
                "interval": 10.0,
                "time": float(i),

                "n_channels": bda_nchan,
                "n_correlations": bda_ncorr,

                "u": baseline["u0"] + i * 0.001,
                "v": baseline["v0"] + i * 0.001,
                "w": 0.0,

                "visibility": _visibility(
                    baseline["real0"] + i,
                    0.0,
                    bda_nchan,
                    bda_ncorr,
                ),
                "weight": _weights(
                    1.0,
                    bda_nchan,
                    bda_ncorr,
                ),
                "flag": _flags(
                    0,
                    bda_nchan,
                    bda_ncorr,
                ),

                "baseline_length": baseline["baseline_length"],
            })

    return rows


@pytest.fixture
def bda_scientific_df(spark, bda_scientific_rows):
    return spark.createDataFrame(
        bda_scientific_rows,
        schema=define_visibility_schema(),
    )


# ---------------------------------------------------------------------
# Fixtures test consumer-imaging
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Fixtures test consumer-imaging
# ---------------------------------------------------------------------

@pytest.fixture
def imaging_num_partitions():
    return 2


@pytest.fixture
def imaging_run_id():
    return "integration-imaging-test"


@pytest.fixture
def imaging_grid_config():
    """
    Configuración mínima de gridding para una prueba de integración rápida.

    Se usa una grilla pequeña y coordenadas UV cercanas al centro para evitar
    que las visibilidades queden fuera del plano UV discretizado.
    """
    return {
        "img_size": 16,
        "padding_factor": 1.0,
        "cellsize": 1e-3,
        "chan_freq": [1.0e9, 1.1e9],
        "corrs_string": ["XX", "YY"],
        "weight_scheme": "NATURAL",
    }


@pytest.fixture
def imaging_nrows():
    return 4


@pytest.fixture
def imaging_nchan():
    return 2


@pytest.fixture
def imaging_ncorr():
    return 2


def _imaging_visibility(real_value, imag_value, nchan=2, ncorr=2):
    return [
        [
            [float(real_value), float(imag_value)]
            for _ in range(ncorr)
        ]
        for _ in range(nchan)
    ]


def _imaging_weights(value, nchan=2, ncorr=2):
    return [
        [
            float(value)
            for _ in range(ncorr)
        ]
        for _ in range(nchan)
    ]


def _imaging_flags(value=0, nchan=2, ncorr=2):
    return [
        [
            int(value)
            for _ in range(ncorr)
        ]
        for _ in range(nchan)
    ]


@pytest.fixture
def imaging_scientific_rows(imaging_nrows, imaging_nchan, imaging_ncorr):
    rows = []

    for i in range(imaging_nrows):
        rows.append({
            "message_id": "0-0",
            "subms_id": 0,
            "field_id": 1,
            "spw_id": 2,
            "polarization_id": 0,

            "baseline_key": f"{i}-{i + 1}",
            "antenna1": i,
            "antenna2": i + 1,
            "scan_number": 1,

            "exposure": 10.0,
            "interval": 10.0,
            "time": float(i),

            "n_channels": imaging_nchan,
            "n_correlations": imaging_ncorr,

            # Coordenadas pequeñas para caer dentro de la grilla.
            "u": float(i) * 0.1,
            "v": float(i) * 0.1,
            "w": 0.0,

            "visibility": _imaging_visibility(
                real_value=1.0 + i,
                imag_value=0.1 * i,
                nchan=imaging_nchan,
                ncorr=imaging_ncorr,
            ),
            "weight": _imaging_weights(
                value=1.0,
                nchan=imaging_nchan,
                ncorr=imaging_ncorr,
            ),
            "flag": _imaging_flags(
                value=0,
                nchan=imaging_nchan,
                ncorr=imaging_ncorr,
            ),

            "baseline_length": 10.0 + i,
        })

    return rows


@pytest.fixture
def imaging_scientific_df(spark, imaging_scientific_rows):
    return spark.createDataFrame(
        imaging_scientific_rows,
        schema=define_visibility_schema(),
    )