import matplotlib

from core.imaging.gridding import apply_gridding, build_grid
from services.consumer_service import consolidate_processing


matplotlib.use("Agg")

EXPECTED_GRID_COLUMNS = {
    "u_pix",
    "v_pix",
    "vs_real",
    "vs_imag",
    "weight",
}


def test_consumer_gridding_weighting_and_dirty_image_pipeline(
    imaging_scientific_df,
    imaging_grid_config,
    imaging_num_partitions,
    imaging_run_id,
    tmp_path,
    monkeypatch,
):
    """
    Valida el tramo de integración de imagen del consumidor.

    El test parte desde un DataFrame científico válido y ejecuta:
    gridding parcial -> ponderación completa -> construcción de grilla
    -> generación de dirty image y PSF.

    No valida Kafka, BDA ni deserialización, porque esos tramos se cubren
    en otros tests de integración.
    """

    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "output" / imaging_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    df_partial_grid = apply_gridding(
        imaging_scientific_df,
        imaging_num_partitions,
        imaging_grid_config,
        strategy="PARTIAL",
    )

    assert df_partial_grid is not None
    assert set(df_partial_grid.columns) == EXPECTED_GRID_COLUMNS

    partial_count = df_partial_grid.count()

    assert partial_count > 0

    padded_size = int(
        imaging_grid_config["img_size"]
        * imaging_grid_config["padding_factor"]
    )

    pixel_bounds = df_partial_grid.select("u_pix", "v_pix").collect()

    for row in pixel_bounds:
        assert 0 <= row["u_pix"] < padded_size
        assert 0 <= row["v_pix"] < padded_size

    df_weighted = apply_gridding(
        df_partial_grid,
        imaging_num_partitions,
        imaging_grid_config,
        strategy="COMPLETE",
    )

    assert df_weighted is not None
    assert set(df_weighted.columns) == EXPECTED_GRID_COLUMNS
    assert df_weighted.count() > 0

    grid, weights = build_grid(
        df_weighted,
        imaging_grid_config,
        imaging_num_partitions,
    )

    assert grid.shape == (padded_size, padded_size)
    assert weights.shape == (padded_size, padded_size)
    assert grid.size > 0
    assert weights.size > 0
    assert weights.sum() > 0.0

    consolidate_processing(
        grid=[df_partial_grid],
        num_partitions=imaging_num_partitions,
        grid_config=imaging_grid_config,
        slurm_job_id=imaging_run_id,
    )

    dirty_image_path = output_dir / f"dirtyimage_{imaging_run_id}.png"
    psf_image_path = output_dir / f"psf_{imaging_run_id}.png"

    assert dirty_image_path.exists()
    assert psf_image_path.exists()

    assert dirty_image_path.stat().st_size > 0
    assert psf_image_path.stat().st_size > 0