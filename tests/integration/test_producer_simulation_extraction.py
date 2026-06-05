import io

import dask
import numpy as np

from data.simulation import generate_dataset
from data.extraction import rechunk_dataset, build_payload


EXPECTED_PAYLOAD_KEYS = {
    "antenna1",
    "antenna2",
    "scan_number",
    "time",
    "exposure",
    "interval",
    "u",
    "v",
    "w",
    "visibilities",
    "weights",
    "flags",
    "baseline_length",
}


def test_pyralysis_generated_dataset_can_be_extracted_into_coherent_blocks(
    real_antenna_config_path,
    real_simulation_config,
):
    """
    This test validates the integration of the dataset generation and extraction processes in Pyralysis. It ensures that a dataset generated from a real antenna configuration and simulation setup can be successfully processed to extract coherent blocks of data, which are then structured into a payload format suitable for downstream analysis.
    """

    
    # Step 1: Generate the dataset using the provided configuration and antenna setup
    dataset = generate_dataset(
        real_antenna_config_path,
        real_simulation_config,
    )

    assert dataset is not None
    assert hasattr(dataset, "ms_list")
    assert len(dataset.ms_list) > 0
    assert hasattr(dataset, "antenna")

    subms = dataset.ms_list[0]

    assert subms is not None
    assert hasattr(subms, "visibilities")
    assert subms.visibilities is not None

    visibility_dataset = subms.visibilities

    assert hasattr(visibility_dataset, "nrows")
    assert visibility_dataset.nrows > 0

    # Step 2: Rechunk the dataset into coherent blocks
    arrays = dask.compute(
        *rechunk_dataset(
            visibility_dataset,
            dataset.antenna,
        )
    )

    assert len(arrays) == 11

    nrows = visibility_dataset.nrows

    (
        antenna1,
        antenna2,
        scan_number,
        time,
        exposure,
        interval,
        uvw,
        visibilities,
        weights,
        flags,
        baseline_length,
    ) = arrays

    assert antenna1.shape[0] == nrows
    assert antenna2.shape[0] == nrows
    assert scan_number.shape[0] == nrows
    assert time.shape[0] == nrows
    assert exposure.shape[0] == nrows
    assert interval.shape[0] == nrows
    assert uvw.shape[0] == nrows
    assert uvw.shape[1] == 3
    assert visibilities.shape[0] == nrows
    assert weights.shape[0] == nrows
    assert flags.shape[0] == nrows
    assert baseline_length.shape[0] == nrows
    assert np.all(baseline_length >= 0.0)

    # Step 3: Build the payload from the arrays and verify its structure
    payload = build_payload(arrays, start=0, end=min(nrows, 10_000))

    assert isinstance(payload, bytes)
    assert len(payload) > 0

    with io.BytesIO(payload) as buffer:
        block = np.load(buffer)

        assert set(block.files) == EXPECTED_PAYLOAD_KEYS
        assert block["antenna1"].shape[0] > 0
        assert block["antenna2"].shape[0] > 0
        assert block["u"].shape[0] > 0
        assert block["v"].shape[0] > 0
        assert block["w"].shape[0] > 0

        assert block["visibilities"].ndim == 4
        assert block["visibilities"].shape[-1] == 2

        assert block["weights"].shape[0] == block["visibilities"].shape[0]
        assert block["flags"].dtype == np.int8
        assert block["baseline_length"].shape[0] == block["visibilities"].shape[0]