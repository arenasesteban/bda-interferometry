# Thesis Environment and Configuration

## Purpose

This document records the runtime dependencies, configuration locations and hardcoded values visible in the restored thesis implementation.

Values that are not recoverable from the repository are explicitly marked as unknown rather than inferred.

---

## 1. Declared software environment

Source: `environment.yml`.

| Dependency     | Declared version | Recovery status              |
| -------------- | ---------------: | ---------------------------- |
| Python         |             3.11 | Exact repository declaration |
| NumPy          |         Unpinned | Version unknown              |
| Pandas         |         Unpinned | Version unknown              |
| PyArrow        |           14.0.2 | Exact repository declaration |
| Matplotlib     |         Unpinned | Version unknown              |
| Astropy        |         Unpinned | Version unknown              |
| Dask           |         Unpinned | Version unknown              |
| IPython kernel |         Unpinned | Version unknown              |
| PySpark        |            3.5.0 | Exact repository declaration |
| findspark      |         Unpinned | Version unknown              |
| OpenJDK        |               11 | Exact repository declaration |
| pytest         |         Unpinned | Version unknown              |
| pytest-cov     |         Unpinned | Version unknown              |
| pytest-mock    |         Unpinned | Version unknown              |
| kafka-python   |         Unpinned | Version unknown              |
| msgpack        |         Unpinned | Version unknown              |

Environment name:

```text
bda-env
```

Channels:

```text
conda-forge
defaults
```

---

## 2. Pyralysis

Pyralysis is an external dependency and is not pinned in `environment.yml`.

The repository README documents installation from:

```text
https://gitlab.com/clirai/pyralysis.git
```

using an editable installation.

No Pyralysis tag, release number or Git commit is pinned by the restored repository.

Historical thesis version:

```text
UNKNOWN
```

This value must remain unknown unless an original environment export, Pyralysis checkout or thesis execution record can establish the exact revision.

A currently installed Pyralysis version must not automatically be reported as the historical thesis version.

---

## 3. Undeclared imported dependency

`src/core/imaging/dirty_image.py` imports:

```python
import cmcrameri
```

`cmcrameri` is not declared in the recovered `environment.yml`.

Historical version:

```text
UNKNOWN
```

CP4 records this dependency mismatch without modifying the environment.

---

## 4. Runtime tools

| Runtime        | Recovered information                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| Micromamba     | Required by README; historical version unknown                                                        |
| Docker Engine  | Required by README; historical version unknown                                                        |
| Docker Compose | Used for the local Kafka service; historical version unknown                                          |
| Kafka broker   | `confluentinc/cp-kafka:7.4.0`                                                                         |
| Spark runtime  | PySpark 3.5.0 declared; external submission/runtime parameters may additionally be defined by scripts |
| SLURM          | Used for HPC thesis execution; exact cluster/runtime version unknown from the core repository         |
| SingularityCE  | Used for HPC execution; exact historical version unknown from the core repository                     |

---

## 5. Kafka broker configuration

Source: `compose.yml`.

```text
Image:
  confluentinc/cp-kafka:7.4.0

Host port:
  9092

Mode:
  KRaft broker + controller

Message maximum:
  10,485,760 bytes

Replica fetch maximum:
  10,485,760 bytes

Log retention:
  300,000 ms

Log segment:
  1,073,741,824 bytes

Heap:
  -Xms256m
  -Xmx512m

Container memory limit:
  1 GiB

Container memory reservation:
  512 MiB
```

The setup is a single-broker local environment with replication factor 1.

---

## 6. Producer runtime configuration

Primary source:

```text
src/services/producer_service.py
```

### CLI parameters

```text
--topic
--bootstrap-servers
--run-id
--antenna-config
--simulation-config
--bda-config
--grid-config
--offset
```

Producer default topic:

```text
visibility-stream
```

Producer default FOV offset:

```text
0.01
```

### Important baseline behavior

Although `producer_service.py` accepts a `--bootstrap-servers` argument, Kafka producer construction currently occurs in:

```text
src/data/extraction.py
```

and uses:

```text
localhost:9092
```

directly.

Therefore the Kafka endpoint is partially hardcoded in the restored thesis implementation.

This is documented but not corrected in CP4.

---

## 7. Dask producer configuration

Source:

```text
src/data/extraction.py
```

Recovered values:

```text
workers:             1
threads per worker:  4
memory limit:        350 GB
processes:           true

memory target:       0.60
memory spill:        0.80
memory pause:        0.95
memory terminate:    0.98

split large chunks:  true
```

Temporary Dask directory:

```text
environment variable: DASK_DIR
default: tmp/dask
```

The historical value of `DASK_DIR`, when explicitly configured during thesis executions, is not recoverable from the repository.

---

## 8. Producer Kafka client configuration

Source:

```text
src/data/extraction.py
```

```text
bootstrap servers:       localhost:9092
acks:                    all
retries:                 10
linger:                  50 ms
batch size:              1,048,576 bytes
max request size:        10,485,760 bytes
request timeout:         120,000 ms
delivery timeout:        180,000 ms
compression:             lz4
max block:               120,000 ms
API detection timeout:   30,000 ms
```

Visibility rows per emitted block:

```text
10,000
```

---

## 9. Simulation configuration

Primary source:

```text
src/data/simulation.py
```

The simulation JSON consumed by the producer can define or supply values including:

```text
interferometer
array_type
assembly
freq_min
freq_max
n_chans
observation_time
declination
integration_time
source_path
spectral_index
flux_density
```

The tracked simulation configuration files are:

```text
alma-band-01.json
ska-mid-band-02.json
```

The antenna configuration files used together with these simulation
configurations are:

```text
alma.cfg
skamid.cfg
```

The simulation configuration files are loaded by `src/services/producer_service.py` and passed to `src/data/simulation.py`. The antenna files are supplied through the producer --antenna-config argument and loaded by the dataset-generation path.

---

## 10. Simulation hardcodes

The restored simulator also contains values that are not supplied through external configuration.

Random seeds:

```text
NumPy: 42
Dask:  42
```

For SKA simulation:

```text
reference observation date:
  current system date at execution time

additional point-source count:
  randomly selected from 8 through 14

additional point-source reference intensity:
  0.15 Jy
```

Point-source pixel positions are selected randomly from the configured FITS image extent.

Because the SKA reference date is generated from the execution date, simulation configuration alone does not fully identify an historical run.

CP4 records this behavior without changing it.

---

## 11. Derived BDA configuration

The producer mutates the BDA configuration before streaming.

Derived values include:

```text
lambda_ref
fov
theta_max
threshold
```

The current calculation is:

```text
fov       = 1.02 * lambda_ref / min_diameter
theta_fov = fov / 2
theta_max = theta_fov * offset
threshold = lambda_ref / (theta_fov * offset)
```

The consumer additionally sets:

```text
decorr_factor
```

from its CLI argument.

Consumer default:

```text
decorr_factor = 0.95
```

The BDA processor consumes:

```text
decorr_factor
lambda_ref
theta_max
threshold
```

---

## 12. BDA hardcoded behavior

The baseline-dependent maximum number of temporal samples uses:

```text
samples_ref = 8
min_samples = 1
max_samples = 16
```

These defaults are defined in `src/core/bda/bda_core.py`.

They are not externalized in the restored implementation.

---

## 13. Imaging configuration

The grid configuration is supplied through the consumer CLI and modified by the producer.

Required/consumed fields include:

```text
img_size
padding_factor
cellsize
cellsize_strategy
cellsize_flag
corrs_string
chan_freq
weight_scheme
```

When:

```text
cellsize_strategy = DERIVED
```

the producer computes:

```text
cellsize = theoretical_resolution / 7
```

When fixed cell size is used, the producer can convert the supplied arcsecond value to radians.

Supported weighting schemes are:

```text
NATURAL
UNIFORM
```

---

## 14. Consumer Spark configuration

Source:

```text
src/services/consumer_service.py
```

Spark application name:

```text
BDA-Interferometry-Consumer
```

The application does not define its Spark master directly. Master/executor configuration is therefore expected to come from the external execution context.

Recovered processing parameters:

```text
Kafka max offsets per trigger:
  300

Structured Streaming trigger:
  60 seconds

Base processing partitions:
  defaultParallelism * 4

BDA repartition:
  num_partitions * 2 by baseline_key

BDA output coalesce:
  num_partitions

Final grid repartition:
  num_partitions * 2 by u_pix, v_pix

Grid materialization repartition:
  num_partitions * 3 by v_pix
```

Streaming checkpoints are generated under:

```text
/tmp/spark-bda-<random-id>-<timestamp>
```

---

## 15. Configuration location inventory

The authoritative tracked configuration files under `conf/` are:

| Path                                           | Category                           | Main parameters                                                                                                                            | Consumed by                                                                                       |
| ---------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `conf/antenna/alma.cfg`                        | Antenna                            | Observatory, coordinate system, antenna positions, diameters and identifiers                                                               | `src/data/simulation.py`, through the producer `--antenna-config` argument                        |
| `conf/antenna/skamid.cfg`                      | Antenna                            | Observatory, coordinate system, antenna positions, diameters and identifiers                                                               | `src/data/simulation.py`, through the producer `--antenna-config` argument                        |
| `conf/runtime/bda_config.json`                 | BDA and evaluation                 | `lambda_ref`, `fov`, `threshold`, `amplitude_tolerance`, `rms_tolerance`                                                                   | `src/services/producer_service.py`, `src/services/consumer_service.py` and BDA/evaluation modules |
| `conf/runtime/grid_config.json`                | Imaging                            | `weight_scheme`, `img_size`, `padding_factor`, `cellsize`, `cellsize_strategy`, `cellsize_flag`, `corrs_string`, `chan_freq`               | `src/services/producer_service.py`, `src/services/consumer_service.py` and imaging modules        |
| `conf/runtime/simulation/alma-band-01.json`    | Simulation                         | Interferometer, frequency range, number of channels, observation time, declination, integration time and source parameters                 | `src/services/producer_service.py` and `src/data/simulation.py`                                   |
| `conf/runtime/simulation/ska-mid-band-02.json` | Simulation                         | Interferometer, array type, assembly, frequency range, number of channels, observation time, declination, integration time and source path | `src/services/producer_service.py` and `src/data/simulation.py`                                   |
| `conf/runtime/spark.json`                      | Spark, Kafka and streaming runtime | Spark application settings, master, partition settings, Kafka servers, topic, consumer group, trigger interval and checkpoint location     | Tracked runtime configuration; no direct reference was found in the current Python code           |

The tracked configuration inventory was verified with:

```bash
git ls-files conf | sort
```

The inventory contains seven tracked files.

`conf/runtime/spark.json` contains runtime settings, but the current Python implementation does not load this file directly. Its values must therefore be treated as recorded configuration rather than confirmed active runtime settings.

---

## 16. Legacy standalone configuration

`main.py` contains an older hardcoded simulation setup, including an ALMA antenna configuration path and direct values for frequencies, observation duration, declination, integration time and source parameters.

Its invocation does not match the current `generate_dataset` signature.

Those values therefore represent legacy repository state and must not be silently interpreted as the canonical configuration of the final thesis streaming pipeline.

---

## 17. Known unknowns

The recovered repository does not establish with certainty:

```text
exact historical Pyralysis revision
exact NumPy version
exact Pandas version
exact Dask version
exact Astropy version
exact Matplotlib version
exact kafka-python version
exact msgpack version
exact cmcrameri version
exact Micromamba version
exact Docker Engine / Compose version
exact SLURM version
exact SingularityCE version
historical DASK_DIR value
runtime Spark master/executor settings unless recoverable from thesis scripts
```

These values must remain marked as unknown unless historical evidence is recovered.

---

## Scope boundary

CP4 documents the current placement and origin of configuration.

It does not:

```text
externalize hardcoded parameters
normalize configuration files
introduce a new configuration schema
pin previously unpinned dependencies
change runtime values
change scientific parameters
```

Those are later-phase tasks.
