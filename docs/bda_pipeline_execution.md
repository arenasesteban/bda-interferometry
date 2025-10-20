# BDA Interferometry Pipeline Execution

This document provides an overview of the execution components of the BDA Interferometry Pipeline, focusing on the roles of producers and consumers within the system.

## Producer - producer_service.py

Producer is the component responsible for creating and sending messages to a Kafka topic. Generate simulated interferometric dataset, format it according to a defined schema, and deliver them to the appropriate topic.

### Key Responsibilities of Producer

- **Data Generation**: Producer generates simulated interferometric dataset using Pyralysis library.
- **Extraction**: Producer extracts visibilities information from the generated dataset to create meaningful messages.
- **Message Formatting**: Before transmitting, the producer formats the messages with the required schema.

### Execution Flow

Producer generates a dataset with Pyralysis library, extracts visibility information, formats the data into messages, and sends these messages to a designated Kafka topic.

The overall execution flow can be summarized as follows:

```txt
Generated Dataset -> Extract Visibilities -> Format Messages -> Send to Topic
```

#### Visibility data structure

Datasets are generated in chunks, with each chunk containing a fixed number of rows. The producer iterates through the dataset, extracting visibility information for each chunk and packing it into a message. Each message corresponds to one data chunk and includes both metadata and scientific data arrays.

Example of the visibility data structure sent by the producer:

```python
{
    # Metadata
    'subms_id',
    'chunk_id',
    'field_id',
    'spw_id',
    'polarization_id',
    
    # Chunk information
    'row_start': start_row,
    'row_end': end_row,
    'nrows': chunk_size,
    
    # Dataset dimensions
    'n_channels',
    'n_correlations',

    # Baseline information
    'antenna1',
    'antenna2',
    'scan_number',

    # Frequency information
    'longitude',
    'lambda_ref',
    'ra',
    'dec',

    # Timing information
    'exposure',
    'interval',
    'integration_time_s',

    # UVW coordinates and time
    'time',
    'u',
    'v',
    'w',

    # Essential scientific data arrays
    'visibilities',
    'weight',
    'flag',
}
```

## Consumer - consumer_service.py

A consumer subscribes to a topic and processes the messages it receives. The consumer is responsible for handling incoming data, performing distributed computations, and preparing output for further analysis.

### Key Responsibilities of Consumer

- **Message Subscription**: The consumer subscribes to specific topic to receive messages produced by the producer.
- **Micro-batching**: Messages are processed in micro-batches of intervals 10 seconds.
- **Deserialization**: Messages are deserialized to reconstruct the original scientific structures for processing.
- **Distributed Processing**: The consumer leverages Apache Spark to process large volumes of data in parallel across multiple nodes.
- **BDA Processing (Interferometry)**: Applies baseline-dependent averaging (BDA) techniques to reduce visibility data volume, based on distance between antenna pairs and their decorrelation properties.

### Execution Flow

The execution basic flow for a consumer can be summarized as follows:

```txt
Subscribe to Topic -> Receive Messages -> Deserialize Messages -> Distributed Processing -> BDA Processing -> Output Results
```

### Micro-batch Processing Diagram

The following diagram illustrates the micro-batch processing flow within the consumer service:

```
MICROBATCH 01
[DESERIALIZE]
+----------------------------+
|                   CHUNK 01 | # Partition 0
+----------------------------+
|                   CHUNK 02 | # Partition 1
+----------------------------+
|                   CHUNK 03 | # Partition 2
+----------------------------+
              |
[DATAFRAME]   v
+----------------------------+
|                     ROW 01 |
|                     ROW 02 |
|                     ROW 03 |
|                        ... |
+----------------------------+
              |
[BDA MODULE]  v
+----------------------------+
|       apply_bda(dataframe) |
+----------------------------+
```

### Spark DataFrame Schema

The consumer processes the incoming messages and constructs a Spark DataFrame with the following schema:

```python
return StructType([
    # Metadata
    StructField("subms_id", IntegerType(), True),
    StructField("chunk_id", IntegerType(), True),
    StructField("field_id", IntegerType(), True),
    StructField("spw_id", IntegerType(), True),
    StructField("polarization_id", IntegerType(), True),

    # Chunk information
    StructField("row_start", IntegerType(), True),
    StructField("row_end", IntegerType(), True),
    StructField("nrows", IntegerType(), True),

    # Dataset dimensions
    StructField("n_channels", IntegerType(), True),
    StructField("n_correlations", IntegerType(), True),

    # Baseline information
    StructField("antenna1", IntegerType(), True),
    StructField("antenna2", IntegerType(), True),
    StructField("scan_number", IntegerType(), True),
    StructField("baseline_key", StringType(), True),

    # Frequency information
    StructField("longitude", DoubleType(), True),
    StructField("lambda_ref", DoubleType(), True),
    StructField("ra", DoubleType(), True),
    StructField("dec", DoubleType(), True),

    # Timing information
    StructField("exposure", DoubleType(), True),
    StructField("interval", DoubleType(), True),
    StructField("integration_time_s", DoubleType(), True),

    # UVW coordinates and time
    StructField("time", DoubleType(), True),
    StructField("u", DoubleType(), True),
    StructField("v", DoubleType(), True),
    StructField("w", DoubleType(), True),

    # Essential scientific data arrays
    StructField("visibilities", ArrayType(ArrayType(ArrayType(DoubleType()))), True),
    StructField("weight", ArrayType(DoubleType()), True),
    StructField("flag", ArrayType(ArrayType(BooleanType())), True)
])
```

## BDA Module - /src/bda

The BDA module contains the core logic for performing baseline-dependent averaging on visibility data. It follows the principles described by Wijnholds (2018) in "Baseline dependent averaging in radio interferometry".

### File Structure

```
/src/bda
├── __init__.py
├── bda_config.py
├── bda_core.py
├── bda_processor.py
└── bda_integration.py
```
- `bda_config.py`: Configuration parameters for BDA processing.
- `bda_core.py`: Core algorithms and mathematical functions for BDA computation.
- `bda_processor.py`: Windowing and temporal processing logic.
- `bda_integration.py`: Integration utilities for connecting BDA with the consumer service.

### Execution Flow

1. **Data Input**: Receives a Spark DataFrame containing visibility data.
2. **Grouping**: Groups data by `baseline_key` and `scan_number` to prepare for averaging.
3. **Partitioning**: Split grouped data into partitions for parallel processing.
4. **Windowing**: Divides data into temporal windows based on decorrelation time.
   - Calculates decorrelation time for each row.
   - Assigns each row to a window according to its time.
   - Closes the window when the next observation time exceeds the initial decorrelation period.
5. **Averaging**: Computes the average for each temporal window.
6. **Output**: Returns a new Spark DataFrame with reduced (averaged) visibility data.

### BDA Processing Diagram

```
[DATAFRAME]
+----------------------------+
|                     ROW 01 |
|                     ROW 02 |
|                     ROW 03 |
|                        ... |
+----------------------------+
              |
[GROUP BY]    v
+----------------------------+
| BASELINE_KEY_01     ROW 01 | # Partition 0
|                     ROW 02 |
|                        ... |
+----------------------------+
| BASELINE_KEY_02     ROW 01 | # Partition 1
|                     ROW 02 |
|                        ... |
+----------------------------+
| BASELINE_KEY_03     ROW 01 | # Partition 2
|                     ROW 02 |
|                        ... |
+----------------------------+
|                        ... |
+----------------------------+
              |
[WINDOWING]   v # Partition 01
+----------------------------+
| BASELINE_KEY_01     ROW 01 |
| WINDOW 01           ROW 02 |
|                        ... |
+----------------------------+
| BASELINE_KEY_01     ROW 01 |
| WINDOW 02           ROW 02 |
|                        ... |
+----------------------------+
|                        ... |
+----------------------------+
              |
[AVERAGING]   v # Partition 01
+----------------------------+
| BASELINE_KEY_01    AVG ROW |
| WINDOW 01                  |
+----------------------------+
| BASELINE_KEY_01    AVG ROW |
| WINDOW 02                  |
+----------------------------+
|                        ... |
+----------------------------+
              |
[RESULT]      v 
+----------------------------+
| BASELINE_KEY_01 AVG ROW 01 |
|                 AVG ROW 02 |
|                        ... |
+----------------------------+
| BASELINE_KEY_01 AVG ROW 01 |
|                 AVG ROW 02 |
|                        ... |
+----------------------------+
| BASELINE_KEY_03 AVG ROW 01 |
|                 AVG ROW 02 |
|                        ... |
+----------------------------+
|                        ... |
+----------------------------+
```

### Keys Concepts

- **Decorrelation Time**: The time interval over which the visibility data remain correlated:

```python
decorrelation_time = np.sqrt(6 * (1.0 - decorr_factor)) / abs(np.pi * (u_dot * l + v_dot * m))
```

- **Windowing**: Groups data points into time windows based on decorrelation times to enable averaging.

A window is closed when the time difference from its start exceeds the decorrelation time:

```python
if time_since_start >= window['decorr_time']:
    return True
```

### Averaging Visibilities

Each visibility within a window is averaged to reduce data volume, producing a single averaged visibility per window.

- **Averaging Fields (u, v, w, time)**.

```python
def average_fields(u, v, w, time, flags):
    us = np.array(u, dtype=np.float64)
    vs = np.array(v, dtype=np.float64)
    ws = np.array(w, dtype=np.float64)
    times = np.array(time, dtype=np.float64)

    fs = np.array(flags)
    masked_fs = ~fs

    rows = (masked_fs).any(axis=(1, 2))

    if np.any(rows):
        u_avg = np.mean(us[rows])
        v_avg = np.mean(vs[rows])
        w_avg = np.mean(ws[rows])
        time_avg = np.mean(times[rows])

    return u_avg, v_avg, w_avg, time_avg
```

- **Averaging Visibilities**: .

```python
def average_visibilities(visibilities, flags):
    vs = np.stack(visibilities, axis=0)  # Shape: (N, C, P, 2)
    fs = np.stack(flags, axis=0)         # Shape: (N, C, P)
    
    masked_fs = ~fs
    count = masked_fs.sum(axis=0)

    vs_real = (vs[..., 0] * masked_fs).sum(axis=0)
    vs_imag = (vs[..., 1] * masked_fs).sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        real_avg = np.where(count > 0, vs_real / count, 0)
        imag_avg = np.where(count > 0, vs_imag / count, 0)

    vs_avg = np.stack([real_avg, imag_avg], axis=-1)

    return vs_avg
```

- **Averaging Weights**: .

```python
def average_weights(weights, flags):
    ws = np.stack(weights, axis=0)       # Shape: (N, P)
    fs = np.stack(flags, axis=0)         # Shape: (N, C, P)
    
    masked_fs = ~fs
    count = masked_fs.sum(axis=0)

    N, C, P = fs.shape
    ws_bc = np.broadcast_to(ws[:, None, :], (N, C, P))
    ws_valid = np.where(masked_fs, ws_bc, 0.0)
    
    ws_avg = ws_valid.sum(axis=0)

    return ws_avg
```

- **Averaging Flags**: .

```python
def average_flags(flags):
    fs = np.stack(flags, axis=0)         # Shape: (N, C, P)
    
    masked_fs = ~fs
    count = masked_fs.sum(axis=0)

    fs_avg = (count == 0).astype(np.int8)

    return fs_avg
```
