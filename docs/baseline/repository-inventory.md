# Thesis Repository Inventory

## Purpose

This document describes the structure and executable components of the restored thesis implementation.

It intentionally documents the repository **as it exists at the thesis baseline**. It does not describe or anticipate the target v1.0 Ports & Adapters architecture.

Historical reference: `thesis-original`.

---

## 1. Primary pipeline

The restored thesis pipeline is composed of two executable services:

```text
Pyralysis simulation
        |
        v
Producer service
        |
        v
Dask extraction / serialization
        |
        v
Kafka
        |
        v
Spark Structured Streaming consumer
        |
        v
Baseline Dependent Averaging
        |
        +------------------+
        |                  |
        v                  v
    Evaluation          Gridding
                           |
                           v
                       Weighting
                           |
                           v
                       Dirty image
```

---

## 2. Executable entrypoints

| Path                               | Role                                                                                                                                                                                    | Status                                                               |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `src/services/producer_service.py` | Primary thesis producer CLI. Generates the simulated dataset, updates runtime-derived BDA/imaging configuration and starts Kafka publication.                                           | Active thesis pipeline entrypoint                                    |
| `src/services/consumer_service.py` | Primary thesis consumer CLI. Creates the Spark streaming application, consumes Kafka messages, applies BDA, performs gridding, creates the final image and computes evaluation metrics. | Active thesis pipeline entrypoint                                    |
| `main.py`                          | Legacy standalone dataset-generation entrypoint.                                                                                                                                        | Historical/legacy; not considered the canonical streaming entrypoint |

### Known documentation mismatch

The root `README.md` still refers to:

```text
python services/producer_service.py
python services/consumer_service.py
```

The restored repository currently stores these files under:

```text
src/services/producer_service.py
src/services/consumer_service.py
```

This discrepancy is preserved here as part of the baseline. It is not corrected as part of this inventory.

### Legacy `main.py`

`main.py` imports `data.simulation.generate_dataset`, but invokes it using an older argument interface based on individual simulation parameters.

The current implementation of `generate_dataset` expects:

```python
generate_dataset(antenna_config_path, sim_config)
```

Therefore, `main.py` must be treated as a legacy entrypoint until its historical role is clarified. CP4 does not modify it.

---

## 3. Producer components

| Component                  | Path                               | Responsibility                                                                                                                            | Direct dependencies                    |
| -------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Producer service           | `src/services/producer_service.py` | Producer CLI and orchestration. Loads simulation configuration, generates data, updates derived configuration and starts Kafka streaming. | simulation, extraction, Astropy, JSON  |
| Simulation                 | `src/data/simulation.py`           | Builds the Pyralysis interferometer and observation, creates sky sources and runs the simulator.                                          | Pyralysis, Dask, NumPy, Astropy        |
| Extraction / serialization | `src/data/extraction.py`           | Converts Pyralysis/Dask arrays into streamed blocks, computes baseline lengths and serializes Kafka payloads.                             | Dask, NumPy, MessagePack, kafka-python |

### Serialized visibility block

The producer transmits the following numerical data:

```text
antenna1
antenna2
scan_number
time
exposure
interval
u
v
w
visibilities
weights
flags
baseline_length
```

The metadata header contains:

```text
schema
message_id
subms_id
field_id
spw_id
polarization_id
n_channels
n_correlations
```

The numerical payload is serialized as compressed NumPy `.npz`.

Metadata is encoded with MessagePack.

The end-of-stream control record uses the Kafka key:

```text
__END__
```

---

## 4. Consumer components

| Component              | Path                                    | Responsibility                                                                                                                   | Direct dependencies                                   |
| ---------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Consumer service       | `src/services/consumer_service.py`      | Spark Structured Streaming orchestration, Kafka deserialization, BDA invocation, partial/final gridding, imaging and evaluation. | PySpark, NumPy, MessagePack, BDA, imaging, evaluation |
| BDA integration        | `src/core/bda/bda_integration.py`       | Repartitions data by baseline, orders records and invokes BDA processing.                                                        | PySpark, BDA processor                                |
| BDA processor          | `src/core/bda/bda_processor.py`         | Assigns temporal windows and performs weighted visibility averaging.                                                             | PySpark, Pandas, NumPy, BDA core                      |
| BDA mathematics        | `src/core/bda/bda_core.py`              | UV-distance, phase-difference, sinc and baseline-dependent window-size calculations.                                             | NumPy, Python math                                    |
| BDA configuration      | `src/core/bda/bda_config.py`            | Loads and validates BDA JSON configuration.                                                                                      | JSON                                                  |
| Gridding               | `src/core/imaging/gridding.py`          | Hermitian duplication, UV pixel assignment, partial accumulation and final distributed grid construction.                        | PySpark, Pandas, NumPy, Astropy                       |
| Weighting              | `src/core/imaging/weighting_schemes.py` | Natural and uniform visibility weighting.                                                                                        | PySpark                                               |
| Dirty image            | `src/core/imaging/dirty_image.py`       | FFT/PSF generation and PNG output.                                                                                               | NumPy, Matplotlib, cmcrameri                          |
| Evaluation coordinator | `src/core/evaluation/metrics.py`        | Coordinates scientific and compression metrics.                                                                                  | amplitude, RMS, baseline, UV coverage                 |
| Amplitude evaluation   | `src/core/evaluation/amplitude.py`      | Measures amplitude error introduced by averaging.                                                                                | PySpark, Pandas, NumPy                                |
| RMS evaluation         | `src/core/evaluation/rms.py`            | Measures complex visibility RMS error.                                                                                           | PySpark, Pandas, NumPy                                |
| Baseline evaluation    | `src/core/evaluation/baseline.py`       | Computes compression per baseline and baseline-dependent summaries.                                                              | PySpark                                               |
| UV coverage            | `src/core/evaluation/coverage.py`       | Exports UV coordinates and comparison plots.                                                                                     | PySpark, NumPy, Matplotlib                            |

---

## 5. Major dependency flow

```text
src/services/producer_service.py
    |
    +--> src/data/simulation.py
    |       |
    |       +--> Pyralysis
    |       +--> Astropy
    |       +--> Dask / NumPy
    |
    +--> src/data/extraction.py
            |
            +--> Dask
            +--> kafka-python
            +--> MessagePack
            +--> NumPy


Kafka
  |
  v

src/services/consumer_service.py
    |
    +--> src/core/bda/bda_config.py
    +--> src/core/bda/bda_integration.py
    |       |
    |       +--> src/core/bda/bda_processor.py
    |               |
    |               +--> src/core/bda/bda_core.py
    |
    +--> src/core/imaging/gridding.py
    |       |
    |       +--> src/core/imaging/weighting_schemes.py
    |
    +--> src/core/imaging/dirty_image.py
    |
    +--> src/core/evaluation/metrics.py
            |
            +--> amplitude.py
            +--> rms.py
            +--> baseline.py
            +--> coverage.py
```

---

## 6. Spark data flow

The consumer performs the following processing sequence:

```text
Kafka DataFrame
    |
    v
control-message filtering
    |
    v
MessagePack metadata decoding
    |
    v
NPZ payload deserialization
    |
    v
Spark visibility DataFrame
    |
    v
repartition(baseline_key)
    |
    v
sortWithinPartitions(
    baseline_key,
    scan_number,
    time
)
    |
    v
BDA temporal-window assignment
    |
    v
weighted averaging
    |
    v
partial gridding per micro-batch
    |
    v
final grid consolidation
    |
    v
weighting
    |
    v
dirty image / PSF
    |
    v
evaluation metrics
```

---

## 7. Repository-level infrastructure

| Path              | Responsibility                                                        |
| ----------------- | --------------------------------------------------------------------- |
| `environment.yml` | Micromamba/Conda environment specification                            |
| `compose.yml`     | Local Kafka broker definition                                         |
| `pytest.ini`      | pytest and coverage configuration                                     |
| `conf/`           | Thesis runtime/scientific configuration files                         |
| `scripts/`        | Execution/HPC helper scripts                                          |
| `notebooks/`      | Historical exploratory notebooks; not the primary execution mechanism |
| `output/`         | Runtime result location; ignored by Git                               |

---

## 8. Thesis execution scripts

| Path | Purpose | Thesis use | Runtime |
| --- | --- | --- | --- |
| `node_run.sh` | Single-node execution script. Starts Kafka and ZooKeeper with Singularity, launches the Spark consumer in local mode, launches the producer and waits for both services to finish. | Uncertain; the script content supports thesis execution, but no execution record is available in the repository. | SLURM, Singularity, local Spark, Kafka and ZooKeeper |
| `run_bash.sh` | Multi-node execution script. Allocates three SLURM nodes, starts a Spark master and workers, starts Kafka and ZooKeeper, launches the consumer through `spark-submit`, then launches the producer. | Uncertain; the script content supports thesis execution, but no execution record is available in the repository. | SLURM, Singularity, distributed Spark, Kafka and ZooKeeper |

---

## 9. Known baseline inconsistencies

### README entrypoint paths

The README refers to a top-level `services/` directory, while the recovered implementation stores the producer and consumer under `src/services/`.

### Legacy `main.py`

The standalone `main.py` invokes an earlier simulation API and is not aligned with the current `generate_dataset` signature.

### pytest coverage configuration

`pytest.ini` contains:

```text
--cov=src --cov=services
```

while the service implementations are located under `src/services/`.

The same configuration explicitly omits an evaluation path from coverage.

These observations are documented only. They are not corrected during CP4.

### Script paths versus recovered repository paths

Both tracked execution scripts refer to legacy paths such as:

```text
services/
antenna_configs/
configs/simulation/
configs/bda_config.json
configs/grid_config.json
```

The recovered repository currently stores the corresponding components and
configuration files under:

```text
src/services/
conf/antenna/
conf/runtime/simulation/
bda_config.json
grid_config.json
```

These path differences are documented as baseline inconsistencies. The scripts are classified according to their contents and declared execution topology; they are not assumed to be directly runnable against the recovered checkout.


---

## 10. Scope boundary

This inventory intentionally does not:

```text
move source files
rename modules
introduce VisibilityBatch
introduce Ports & Adapters
extract Spark-independent BDA
externalize hardcoded configuration
change serialization
change scientific algorithms
change runtime behavior
```

Those changes belong to later phases.

---

## Acceptance status

The inventory is complete when:

```text
all thesis pipeline stages map to existing repository components
producer and consumer entrypoints are unambiguous
all tracked thesis execution scripts are classified
the document describes the restored implementation rather than v1.0 target architecture
all uncertain historical roles are explicitly marked
```
