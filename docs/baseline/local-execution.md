# Thesis Local Execution

## Purpose

This document describes the local execution model recoverable from the restored thesis implementation.

It documents the producer, consumer, Kafka dependency, configuration files, startup sequence and known limitations of the thesis baseline.

It does not introduce a new CLI, demo workflow or simplified execution path.

Historical reference: `thesis-original`.

---

## 1. Execution model

The thesis pipeline consists of two independent Python services connected through Kafka:

```text
Pyralysis simulation
        |
        v
Producer service
        |
        v
Kafka
        |
        v
Spark Structured Streaming consumer
        |
        v
BDA
        |
        v
Gridding / Weighting
        |
        v
Dirty image + Metrics
```

The primary executable entrypoints in the restored repository are:

```text
src/services/producer_service.py
src/services/consumer_service.py
```

---

## 2. Historical README procedure

The recovered root README describes the local workflow as:

```text
1. Create and activate the Micromamba environment.
2. Install Pyralysis.
3. Start Docker Compose.
4. Start the producer.
5. Start the consumer.
```

The README uses the commands:

```bash
python services/producer_service.py
python services/consumer_service.py
```

These commands no longer correspond exactly to the restored checkout.

The actual source files are under:

```text
src/services/
```

and both services currently expose command-line arguments that are not included in the historical README.

Therefore the README is preserved as historical execution evidence, but it is not a complete executable procedure for the restored checkout.

---

## 3. Prerequisites

The repository declares or requires the following local runtime components:

```text
Python 3.11
Micromamba
OpenJDK 11
PySpark 3.5.0
Docker / Docker Compose
Kafka
Pyralysis
Dask
NumPy
Pandas
Astropy
MessagePack
kafka-python
Matplotlib
```

The environment is defined by:

```text
environment.yml
```

and is named:

```text
bda-env
```

Create and activate it with:

```bash
micromamba env create -f environment.yml
micromamba activate bda-env
```

If the environment already exists:

```bash
micromamba activate bda-env
```

---

## 4. Pyralysis dependency

Pyralysis is installed separately from the environment specification.

The historical README records:

```bash
git clone https://gitlab.com/clirai/pyralysis.git
cd pyralysis
pip install \
  --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple \
  -e .
```

No exact Pyralysis revision is pinned by the recovered repository.

Consequently, this installation procedure describes the dependency relationship but does not guarantee bit-for-bit reconstruction of the historical thesis environment.

---

## 5. Known environment limitation

`src/core/imaging/dirty_image.py` imports:

```python
cmcrameri
```

but `cmcrameri` is not declared in the recovered `environment.yml`.

If the package is absent, the consumer will fail while importing the imaging module.

This dependency mismatch is part of the recovered baseline and is not corrected during Phase 0.

---

## 6. Python module path

The source tree uses imports such as:

```python
from data.simulation import generate_dataset
from core.bda.bda_config import load_bda_config
```

when the corresponding modules are located below `src/`.

For direct execution from the repository root, expose `src/` on the Python module path:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

This requirement reflects the current source layout. No packaging layer is introduced by this baseline documentation.

---

## 7. Local Kafka service

The repository contains:

```text
compose.yml
```

with a single Kafka broker based on:

```text
confluentinc/cp-kafka:7.4.0
```

The broker is exposed locally on:

```text
localhost:9092
```

Start it from the repository root:

```bash
docker compose -f compose.yml up -d
```

Check its state with:

```bash
docker compose -f compose.yml ps
```

The container includes a health check against:

```text
localhost:9092
```

The thesis topic used by both producer and consumer is:

```text
visibility-stream
```

No definitive historical local partition count is recoverable from the README or local Compose configuration.

The HPC scripts use 16 Kafka partitions, but that value must not be silently assumed to have been the local historical configuration.

---

## 8. Thesis configuration files

The restored producer/consumer workflow uses:

```text
conf/antenna/skamid.cfg
conf/runtime/simulation/ska-mid-band-02.json
conf/runtime/bda_config.json
conf/runtime/grid_config.json
```

An ALMA antenna/simulation configuration is also tracked, but the SKA-MID files correspond to the primary thesis pipeline configuration recovered from the repository.

The producer modifies the BDA and imaging configuration files before streaming.

Specifically, it writes runtime-derived values such as:

```text
lambda_ref
fov
theta_max
threshold
cellsize
corrs_string
chan_freq
```

Therefore these configuration files are not purely immutable inputs during execution.

---

## 9. Output directory

The consumer writes generated artifacts below:

```text
./output/<run-id>/
```

Before a local execution, create a run identifier and its output directory:

```bash
RUN_ID="local-baseline"
mkdir -p "output/$RUN_ID"
```

The identifier is passed to the consumer through:

```text
--run-id
```

and is subsequently used when generating output filenames.

---

## 10. Recommended startup order for the restored entrypoints

For the restored code, the operational order is:

```text
1. Activate environment.
2. Expose src/ through PYTHONPATH.
3. Start Kafka.
4. Prepare the output directory.
5. Start the Spark consumer.
6. Start the producer.
7. Producer sends visibility blocks.
8. Producer sends the __END__ control record.
9. Consumer completes final gridding, imaging and metrics.
```

Starting the consumer before the producer matches the orchestration visible in the recovered HPC scripts and ensures the streaming query is already waiting when data begins to arrive.

This ordering should not be interpreted as proof that every historical local thesis run was launched in exactly this sequence. The root README lists producer before consumer, and no complete local execution log is preserved in the repository.

---

## 11. Consumer execution

Run the consumer in a first terminal from the repository root.

```bash
micromamba activate bda-env

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="local-baseline"

mkdir -p "output/$RUN_ID"

spark-submit \
  --master "local[*]" \
  --packages "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0" \
  src/services/consumer_service.py \
  --topic "visibility-stream" \
  --bootstrap-server "localhost:9092" \
  --run-id "$RUN_ID" \
  --bda-config "conf/runtime/bda_config.json" \
  --grid-config "conf/runtime/grid_config.json" \
  --decorr-factor 0.95
```

### Why `spark-submit`

The consumer creates a Spark Structured Streaming source using:

```text
format("kafka")
```

and therefore requires the Spark Kafka integration package.

The recovered HPC scripts explicitly provide:

```text
org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0
```

through `spark-submit`.

The historical README command using plain `python` does not record how this Kafka connector dependency was supplied.

---

## 12. Producer execution

After the consumer has started and reports that it is waiting for data, run the producer from a second terminal:

```bash
micromamba activate bda-env

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="local-baseline"

python src/services/producer_service.py \
  --topic "visibility-stream" \
  --bootstrap-servers "localhost:9092" \
  --run-id "$RUN_ID" \
  --antenna-config "conf/antenna/skamid.cfg" \
  --simulation-config "conf/runtime/simulation/ska-mid-band-02.json" \
  --bda-config "conf/runtime/bda_config.json" \
  --grid-config "conf/runtime/grid_config.json" \
  --offset 0.01
```

The producer:

```text
loads simulation configuration
creates the Pyralysis dataset
derives BDA configuration values
derives imaging configuration values
extracts visibility blocks
serializes metadata with MessagePack
serializes numerical payloads with compressed NPZ
publishes them to Kafka
sends an __END__ control record
```

---

## 13. Kafka endpoint limitation

`producer_service.py` accepts:

```text
--bootstrap-servers
```

and reports that value in its logs.

However, `src/data/extraction.py` constructs its `KafkaProducer` with:

```text
localhost:9092
```

hardcoded.

Therefore the producer CLI parameter does not currently control the actual Kafka client endpoint used by the extraction layer.

For local execution this happens to match the Compose broker.

This behavior is documented as part of the thesis baseline and is not corrected during Phase 0.

---

## 14. Consumer processing behavior

The consumer:

```text
creates a Spark session
subscribes to Kafka
reads from earliest available offsets
deserializes MessagePack metadata
deserializes NPZ payloads
reconstructs visibility rows
partitions by baseline
sorts by baseline, scan and time
applies BDA when decorrelation factor < 1
performs partial gridding for each micro-batch
waits for the producer control record
consolidates all partial grids
applies final weighting
builds the UV grid
generates dirty image and PSF
calculates scientific/compression metrics
stops Spark
```

The Structured Streaming trigger interval is:

```text
60 seconds
```

and Kafka input is limited to:

```text
300 offsets per trigger
```

in the recovered implementation.

---

## 15. Expected outputs

Depending on the configured processing path, the run may create:

```text
output/<run-id>/
├── dirtyimage_<run-id>.png
├── psf_<run-id>.png
├── metrics_<run-id>.txt
├── coverage_uv_<run-id>.png
├── coverage_uv_<run-id>_overlay.png
├── coverage_uv_<run-id>_zoom.png
├── coverage_uv_<run-id>_coordinates.csv
├── baseline_dependency/
└── baseline_quartiles/
```

Metrics are produced only when BDA is active:

```text
decorr_factor < 1.0
```

---

## 16. Execution termination

The producer sends a Kafka control record using the key:

```text
__END__
```

The consumer inspects incoming micro-batches for that record and uses it to terminate the streaming query before performing final image generation and evaluation.

The current implementation has an important control-flow limitation:

if a micro-batch contains the `__END__` record but contains no scientific records after control-message filtering, the batch returns before updating the shared end-of-stream state.

Therefore termination behavior depends on how the final control message is grouped into Spark micro-batches.

This behavior is preserved and documented rather than modified in Phase 0.

---

## 17. Historical reproducibility limitations

The recovered local procedure is not fully reproducible from the repository alone.

Known limitations include:

```text
Pyralysis revision is not pinned.

Several Python dependencies are unpinned.

cmcrameri is imported but absent from environment.yml.

The README contains obsolete service paths.

The README does not provide the currently required CLI arguments.

The README does not document the Spark Kafka package.

The producer Kafka endpoint is hardcoded in extraction.py.

The SKA simulation derives its reference date from the execution date.

BDA and grid JSON files are mutated during producer startup.

No complete historical local execution log is preserved.

The historical local Kafka partition count is unknown.
```

These limitations are part of the baseline.

---

## 18. Scope boundary

This document does not:

```text
create a demo command
introduce a new CLI
change source paths
package the Python project
fix dependency declarations
change Kafka configuration
externalize hardcoded values
change simulation parameters
change BDA behavior
change imaging behavior
```

Those concerns belong to later roadmap phases.

---

## Acceptance status

Issue #6 is satisfied when:

```text
the prerequisite environment and external services are documented
the real restored producer and consumer entrypoints are referenced
the current required CLI parameters are represented
the startup relationship between Kafka, consumer and producer is clear
inputs and generated outputs are identified
historical versus currently recoverable behavior is distinguished
known execution limitations are explicit
no new demo or CLI workflow has been introduced
```
