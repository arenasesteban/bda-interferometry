# Thesis HPC Execution

## Purpose

This document records the HPC execution model recoverable from the thesis repository.

It describes the tracked SLURM scripts, resource requests, Spark topology, Kafka/Singularity usage, Micromamba environment activation, Dask configuration and known historical inconsistencies.

The scripts are documented in place.

They are not reorganized, modernized or corrected during Phase 0.

---

## 1. Tracked HPC scripts

Two execution scripts are preserved:

```text
scripts/node_run.sh
scripts/run_bash.sh
```

Their intended roles are recoverable from their contents:

| Script                | Intended topology                                                                                       | Historical execution evidence                                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/node_run.sh` | Single SLURM node, local Spark execution, Kafka and ZooKeeper on the same node                          | Script is compatible with thesis-era infrastructure concepts, but no execution record in the repository proves which specific thesis experiments used it |
| `scripts/run_bash.sh` | Multi-node SLURM allocation with standalone Spark master/workers and Kafka/ZooKeeper on the master node | Script clearly represents distributed scaling infrastructure, but exact historical runs are not recoverable from the repository alone                    |

The script contents are therefore treated as historical runtime evidence, not as proof that every declared configuration was used in a published experiment.

---

## 2. Single-node script

File:

```text
scripts/node_run.sh
```

### SLURM allocation

```text
partition:       largemem
tasks:           1
cpus per task:   20
memory:          100 GB
walltime:        03:00:00
```

The job name is:

```text
radio-astronomy-pipeline
```

SLURM stdout/stderr are written below:

```text
logs/
```

using the job identifier.

### Environment modules

The script loads:

```text
intel/2022.00
singularityCE
```

after:

```bash
ml purge
```

### Micromamba

The script activates:

```text
bda-env
```

using:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate bda-env
```

---

## 3. Single-node Spark topology

The script configures:

```text
Spark master:          local[16]
driver memory:         16g
executor memory:       80g
shuffle partitions:    32
```

Spark Kafka integration:

```text
org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0
```

Additional Spark options include:

```text
adaptive query execution enabled
partition coalescing enabled
skew join handling enabled
Kryo serializer
Python worker reuse
Arrow disabled
driver maxResultSize = 2g
Spark memory fraction = 0.6
Spark storage fraction = 0.3
event logging enabled
```

The consumer therefore executes in local Spark mode inside the allocated SLURM node rather than using a multi-node Spark cluster.

---

## 4. Single-node Kafka topology

The script expects:

```text
Kafka topic:       visibility-stream
Kafka bootstrap:   localhost:9092
Kafka partitions:  16
```

Kafka and ZooKeeper are launched through a Singularity image:

```text
$HOME/radio-astronomy-pipeline/kafka/cp-kafka_7.4.0.sif
```

Kafka data is stored below:

```text
$HOME/kafka-hpc
```

The script clears existing Kafka and ZooKeeper data before startup.

Kafka topic configuration includes:

```text
max.message.bytes = 104857600
segment.bytes     = 1073741824
retention.bytes   = 10737418240
replication       = 1
```

---

## 5. Single-node Dask relationship

The producer implementation creates a local Dask distributed client with:

```text
workers:             1
threads per worker:  4
memory limit:        350 GB
processes:           true
```

The SLURM script, however, requests:

```text
100 GB
```

for the node.

Therefore the Dask worker memory limit configured in the current source exceeds the physical memory requested by `node_run.sh`.

This is recorded as a baseline inconsistency.

No attempt is made during Phase 0 to determine whether an earlier version of the extraction code used a different Dask limit.

---

## 6. Single-node directories

The script creates:

```text
logs/<SLURM_JOB_ID>/
output/<SLURM_JOB_ID>/
tmp/<SLURM_JOB_ID>/
```

and exports:

```text
DASK_DIR=tmp/<SLURM_JOB_ID>/dask
```

Spark event logs are placed below the job log directory.

---

## 7. Multi-node script

File:

```text
scripts/run_bash.sh
```

### Default SLURM allocation

```text
partition:        largemem
nodes:            3
tasks per node:   1
cpus per task:    20
memory per node:  350 GB
walltime:         06:00:00
```

The topology encoded by the script is:

```text
Node 1
├── SLURM job shell
├── Spark master
├── Spark driver
├── Kafka
├── ZooKeeper
└── Producer / consumer launch coordination

Node 2
└── Spark worker

Node 3
└── Spark worker
```

The script explicitly follows a rule of:

```text
one Spark executor per worker node
```

---

## 8. Spark installation

The multi-node script expects a standalone Spark installation at:

```text
$HOME/spark-3.5.0
```

and exports:

```text
SPARK_HOME
PATH
PYSPARK_PYTHON
PYSPARK_DRIVER_PYTHON
PYTHONPATH
```

The exact historical installation procedure for this Spark directory is not recorded in the repository.

---

## 9. Multi-node resource derivation

For each remote worker, the script derives:

```text
worker cores =
    SLURM_CPUS_PER_TASK - 2
    when more than two CPUs are available

worker memory =
    85% of SLURM memory per node

executor cores =
    worker cores

executor memory =
    90% of the worker's 85% memory allocation
```

For the default request of:

```text
20 CPUs per node
```

each remote Spark worker therefore advertises:

```text
18 cores
```

With three total nodes there are two worker nodes, giving:

```text
expected executors:   2
maximum Spark cores:  36
shuffle partitions:   72
```

The driver memory is calculated as:

```text
12% of SLURM memory per node
```

with a minimum of:

```text
4096 MB
```

---

## 10. Multi-node Spark startup

The first allocated node becomes the Spark master.

The script resolves its IPv4 address and creates:

```text
spark://<MASTER_IP>:7077
```

The Spark master web UI uses:

```text
8080
```

Each remaining node starts one Spark worker through `srun`.

Worker web UI:

```text
8081
```

The script verifies that all expected workers are registered before continuing.

---

## 11. Multi-node Spark submission

The consumer is submitted in:

```text
client
```

deploy mode.

Relevant options include:

```text
executor instances = number of remote worker nodes
cores max          = sum of remote executor cores
adaptive execution = enabled
Kryo serializer
driver maxResultSize = 4g
network timeout      = 600s
RPC ask timeout       = 600s
Spark event logging   = enabled
```

The Spark Kafka connector is:

```text
org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0
```

---

## 12. Multi-node Kafka topology

Kafka and ZooKeeper run on the master node using:

```text
cp-kafka_7.4.0.sif
```

The Kafka listener is dynamically configured with the master node IPv4 address.

Bootstrap address:

```text
<MASTER_IP>:9092
```

Topic:

```text
visibility-stream
```

Default partitions:

```text
16
```

The script validates Kafka availability before starting the consumer.

---

## 13. Kafka architecture difference between local and HPC

The root `compose.yml` currently uses:

```text
Kafka KRaft mode
```

without ZooKeeper.

The HPC scripts use:

```text
Kafka + ZooKeeper
```

through the Singularity image.

Therefore local and HPC execution do not use identical Kafka deployment topologies.

This difference is historical/runtime infrastructure coupling and is not normalized during Phase 0.

---

## 14. Producer-side Dask in multi-node execution

Dask is not deployed as a multi-node cluster by `run_bash.sh`.

The producer creates its own local Dask client internally.

Therefore the distributed topology is asymmetric:

```text
Pyralysis / extraction:
    local Dask client on the producer/master node

BDA / imaging:
    Spark executors on remote worker nodes
```

The current producer Dask client uses:

```text
1 worker
4 threads
350 GB configured memory limit
```

This means Dask is used for producer-side materialization/extraction, while Spark provides the distributed processing layer after Kafka.

---

## 15. Service startup sequence

The multi-node script explicitly executes:

```text
1. Activate software environment.
2. Determine allocated nodes.
3. Start Spark master.
4. Start Spark workers.
5. Verify Spark workers.
6. Start ZooKeeper.
7. Start Kafka.
8. Verify Kafka.
9. Create Kafka topic.
10. Start Spark consumer.
11. Start producer.
12. Wait for producer.
13. Wait for consumer.
14. Report logs/output.
15. Clean up background services.
```

The single-node script follows the same broad dependency order:

```text
ZooKeeper
Kafka
Micromamba environment
Spark consumer
producer
```

---

## 16. Runtime outputs

Both scripts associate execution state with:

```text
SLURM_JOB_ID
```

and create job-specific directories for:

```text
logs
output
Spark event logs
temporary files
```

This establishes SLURM job identifiers as the historical execution/run identifier in the HPC workflow.

---

## 17. Historical script/API drift

The tracked HPC scripts no longer correspond exactly to the recovered Python CLI.

### Service paths

The scripts refer to:

```text
$PROJECT_ROOT/services/
```

while the restored source files are located at:

```text
src/services/
```

### Configuration paths

The scripts refer to paths such as:

```text
antenna_configs/skamid.cfg
configs/simulation/ska-mid-band-02.json
configs/bda_config.json
configs/grid_config.json
```

while the restored repository uses:

```text
conf/antenna/skamid.cfg
conf/runtime/simulation/ska-mid-band-02.json
conf/runtime/bda_config.json
conf/runtime/grid_config.json
```

### Consumer identifier argument

Both HPC scripts pass:

```text
--slurm-job-id
```

The restored consumer currently requires:

```text
--run-id
```

### Producer arguments

The restored producer requires:

```text
--bda-config
--grid-config
```

but the historical scripts do not supply those arguments.

### Runtime variables

The scripts define values such as:

```text
DECORR_FACTOR
FOV
```

but those variables are not consistently forwarded through the currently recovered service CLI.

Therefore:

```text
scripts/node_run.sh
scripts/run_bash.sh
```

must not be described as directly runnable against the restored checkout.

They are historical runtime artifacts.

---

## 18. Producer Kafka incompatibility with distributed script

The multi-node script advertises Kafka to the application as:

```text
<MASTER_IP>:9092
```

However, the restored producer eventually creates its Kafka client in `src/data/extraction.py` using:

```text
localhost:9092
```

directly.

This means the recovered producer implementation is coupled to a local broker endpoint even though the multi-node script models Kafka through the master node network address.

Whether the historical thesis execution used an earlier compatible extraction implementation cannot be established from the current repository.

This inconsistency is documented rather than corrected.

---

## 19. Recoverable versus unknown HPC information

### Recoverable from repository

```text
partition names used by scripts
requested nodes
requested CPUs
requested memory
requested walltime
Spark topology
Spark package version
Kafka image path/name
Kafka topic
Kafka partition defaults in scripts
Micromamba environment name
Singularity usage
Dask configuration in source
directory layout
startup ordering
```

### Not established by repository evidence

```text
exact SLURM software version
exact SingularityCE version
exact Micromamba version
exact NLHPC node hardware used for every run
which thesis experiment used each script revision
whether sbatch parameters were overridden externally
exact Pyralysis revision
exact environment package builds
complete historical execution logs
whether every scaling configuration completed successfully
```

Unknown settings must remain explicitly unknown.

---

## 20. Scope boundary

Phase 0 does not:

```text
move scripts to deploy/hpc/
repair script paths
update CLI arguments
replace ZooKeeper with KRaft
change SLURM resources
change Spark topology
change Dask memory
externalize runtime variables
modernize Singularity usage
rerun HPC experiments
```

Those changes belong to later roadmap phases.

---

## Acceptance status

Issue #8 is satisfied when:

```text
both tracked HPC scripts are identifiable
their intended topology is documented
node/CPU/memory/walltime/partition settings are recorded
Spark and Dask roles are distinguished
Micromamba and Singularity usage are documented
known script/API drift is explicit
unknown historical values remain marked unknown
the scripts remain unchanged and in their existing location
```
