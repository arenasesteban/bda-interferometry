# Thesis Datasets and Historical Results

## Purpose

This document provides traceability for the datasets, experiment inputs and historical outputs associated with the thesis implementation.

The purpose is preservation and traceability.

No experiment is rerun solely to complete this inventory, and no historical scientific result is recomputed or reinterpreted.

---

## 1. Dataset origin

The primary thesis datasets are synthetic radio-interferometric visibility datasets generated with Pyralysis.

Current data flow:

```text
antenna configuration
        +
simulation configuration
        +
astronomical source model / FITS input
        |
        v
Pyralysis Simulator
        |
        v
Pyralysis Dataset
        |
        v
Sub-MS visibility datasets
        |
        v
Dask extraction
        |
        v
Kafka visibility blocks
```

The restored thesis pipeline therefore does not require the generated visibility dataset to be persisted as an intermediate file before streaming.

---

## 2. Known simulation inputs

The simulator receives:

```text
antenna configuration path
interferometer
array type
assembly stage
minimum frequency
maximum frequency
number of channels
observation time
declination
integration time
astronomical source path
spectral index
flux density where applicable
```

For SKA simulations, the source model additionally includes a FITS-backed non-parametric source and generated point sources.

The exact input files used by each historical experiment must be associated with the experiment whenever they are recoverable.

---

## 3. Thesis SKA-MID experiment families

The thesis work used SKA-MID assembly stages including:

| Stage | Antennas |
| ----- | -------: |
| AA0.5 |        4 |
| AA1   |        8 |
| AA2   |       64 |
| AA*   |      144 |
| AA4   |      197 |

BDA decorrelation-factor experiments included:

```text
0.99
0.97
0.95
```

Horizontal-scaling experiments were designed around:

```text
1 node
2 nodes
4 nodes
8 nodes
```

The large-scale horizontal-scaling workload was associated with the AA4 stage.

These experiment families identify the historical study space; they do not imply that every combination still has preserved output artifacts.

---

## 4. Dataset inventory schema

Each recoverable historical dataset or generated experiment input must be recorded using:

| Field                    | Description                                 |
| ------------------------ | ------------------------------------------- |
| ID                       | Stable descriptive identifier               |
| Origin                   | Pyralysis simulation or other source        |
| Interferometer           | Telescope/array family                      |
| Assembly                 | Array assembly stage                        |
| Antennas                 | Number of antennas when known               |
| Duration                 | Observation duration                        |
| Integration              | Integration time                            |
| Channels                 | Number of frequency channels                |
| Correlations             | Number of correlations                      |
| Source input             | FITS/source configuration                   |
| Antenna configuration    | Configuration file used                     |
| Simulation configuration | Configuration file used                     |
| Row count                | Visibility row count when known             |
| Historical location      | Where the dataset/result originally existed |
| Current availability     | Git / local-only / thesis-only / missing    |
| Reproducible             | Yes / partial / no / unknown                |
| Notes                    | Provenance limitations                      |

---

## 5. Availability vocabulary

Use the following status values consistently:

| Status        | Meaning                                                                   |
| ------------- | ------------------------------------------------------------------------- |
| `tracked`     | Artifact is currently versioned in Git                                    |
| `local-only`  | Artifact exists on a recovered local machine/storage but is not versioned |
| `thesis-only` | Result is preserved only in thesis text, tables or figures                |
| `external`    | Artifact exists in external/HPC storage                                   |
| `missing`     | Artifact is referenced historically but no copy has been recovered        |
| `unknown`     | Current availability has not been established                             |

---

## 6. Generated result structure

The current consumer uses:

```text
./output/<run_id>/
```

as the base result location.

The implementation is expected to generate the following artifacts under this
directory. These are implementation-level output paths, not recovered
historical artifacts.

```text
metrics_<run_id>.txt

dirtyimage_<run_id>.png
psf_<run_id>.png

coverage_uv_<run_id>.png
coverage_uv_<run_id>_overlay.png
coverage_uv_<run_id>_zoom.png
coverage_uv_<run_id>_coordinates.csv

baseline_dependency/
baseline_quartiles/
```

`baseline_dependency/` and `baseline_quartiles/` are Spark CSV output directories.

No corresponding output artifacts were found in the current local checkout.

---

## 7. Scientific result categories

### Amplitude error

The evaluation pipeline compares individual scientific visibilities with the BDA visibility associated with their temporal window.

The results are appended to:

```text
metrics_<run_id>.txt
```

and include absolute and relative amplitude error together with the configured tolerance.

### RMS error

The same metrics file records absolute and relative complex visibility RMS error.

### Baseline dependency

The pipeline records:

```text
rows before BDA
rows after BDA
compression ratio
reduction fraction
reduction percentage
```

per baseline.

It also produces quartile summaries and short/long baseline statistics.

### UV coverage

The evaluation produces:

```text
original UV coordinates
BDA UV coordinates
comparison image
overlay image
zoomed comparison
coordinate CSV
```

### Imaging

The pipeline generates:

```text
dirty image
PSF image
```

from the consolidated weighted UV grid.

### Performance

The consumer prints and records during execution:

```text
per-micro-batch processing duration
processed row count
total processing time
final image-generation time
Spark application information
```

Whether these logs were preserved for each historical experiment must be determined from the recovered thesis artifacts.

---

## 8. Git preservation status

The root `.gitignore` contains:

```text
output/
```

Therefore runtime outputs under `output/` are not expected to be preserved by Git unless they were force-added before the ignore rule applied.

The authoritative check is:

```bash
git ls-files output
```

If this command produces no paths, the restored repository does not contain versioned thesis output artifacts.

For the recovered checkout, `git ls-files output` returns no paths. No
versioned output artifacts were recovered from Git.

This does not mean that the historical results did not exist. They may still be present:

```text
on the local development machine
on NLHPC storage
inside thesis figures/tables
inside notebooks
inside archived experiment directories
```

---

## 9. Local historical-output inventory

Run:

```bash
find output -type f 2>/dev/null | sort
git ls-files notebooks | sort
git ls-files docs | sort
```

The local checkout was inspected for generated outputs and tracked output
artifacts. The `output/` directory is absent, and no output paths are tracked
by Git. Therefore, no local or Git-tracked historical run output was recovered
from the current checkout.

The documented output types have the following local status:

| Output type | Expected location or pattern | Local status |
| ----------- | ---------------------------- | ------------ |
| Metrics | `output/<run_id>/metrics_<run_id>.txt` | Not recovered |
| Dirty image | `output/<run_id>/dirtyimage_<run_id>.png` | Not recovered |
| PSF image | `output/<run_id>/psf_<run_id>.png` | Not recovered |
| UV coverage image | `output/<run_id>/coverage_uv_<run_id>.png` | Not recovered |
| UV overlay | `output/<run_id>/coverage_uv_<run_id>_overlay.png` | Not recovered |
| UV zoom | `output/<run_id>/coverage_uv_<run_id>_zoom.png` | Not recovered |
| UV coordinates | `output/<run_id>/coverage_uv_<run_id>_coordinates.csv` | Not recovered |
| Baseline dependency | `output/<run_id>/baseline_dependency/` | Not recovered |
| Baseline quartiles | `output/<run_id>/baseline_quartiles/` | Not recovered |

No historical run identifier can be established from the recovered local
checkout.

No HPC storage, archived experiment directory or thesis source package was
available in the recovered repository for direct inspection. Consequently,
HPC and thesis availability remain `unknown` until those external sources are
inspected.

| Storage location | Recovered artifacts | Status |
| ---------------- | ------------------- | ------ |
| Local checkout | None | `missing` for locally expected outputs |
| Git-tracked repository | None | `missing` |
| HPC / NLHPC storage | Not inspected from this checkout | `unknown` |
| Thesis tables and figures | Not available in this checkout | `unknown` |
| Historical experiment directories | Not available in this checkout | `unknown` |

No recovered run is available to populate an experiment-level inventory
entry. If a historical run is recovered later, record it using the following
fields: run or experiment identifier, input/configuration context, result
types, historical location, current availability and reproducibility status.

A run must not be classified as reproducible unless its required input and configuration can be recovered.

---

## 10. Historical numerical results

Numerical values reported by the thesis must be copied verbatim from preserved thesis tables, result files or experiment records.

Do not recreate missing values from memory, rerun a large experiment solely for this inventory or derive a value from a different configuration.

For every preserved numerical result, record the result, verbatim value,
dataset or stage, configuration context and evidence source.

If the associated configuration cannot be identified:

```text
Configuration context: UNKNOWN
```

No historical numerical result files, thesis tables or thesis figures were
available in the recovered checkout for inventory. No numerical result is
classified as recovered from local, HPC or thesis sources.

---

## 11. Reproducibility classification

A historical experiment can be marked `yes` only when its necessary inputs, scientific configuration and runtime procedure are recoverable.

Use:

```text
yes
    Required inputs and relevant configuration are available.

partial
    Main configuration is known but one or more historical dependencies,
    external inputs or runtime details are missing.

no
    Required input data or configuration has been lost.

unknown
    The required provenance has not yet been checked.
```

Large AA4 experiments do not need to be rerun in order to classify their provenance.

---

## 12. Important reproducibility limitation

For SKA simulation, the restored implementation derives the observation reference date from the system date at runtime.

Consequently, a simulation configuration file does not by itself identify the complete historical observation state.

When the original execution date is unavailable, this limitation must be recorded rather than silently reconstructed.

---

## 13. Scope boundary

This inventory does not:

```text
introduce a new experiment-result format
move historical output files
change evaluation metrics
recalculate historical metrics
rerun large thesis experiments
reinterpret thesis conclusions
introduce new datasets
```

Its only purpose is to establish provenance and availability before future refactoring.
