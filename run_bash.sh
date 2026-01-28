#!/bin/bash
#SBATCH -J bda-interferometry
#SBATCH -p debug
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:29:00
#SBATCH -o logs/bda-interferometry-%j.out
#SBATCH -e logs/bda-interferometry-%j.err
#SBATCH --mail-user=esteban.arenas.a@usach.cl
#SBATCH --mail-type=ALL

# ============================================================================
# CONFIGURATION
# ============================================================================

set -e # Exit on any error

# Modules
ml purge
ml intel/2022.00
ml singularityCE

# Variables
KAFKA_DIR=$HOME/kafka-hpc
KAFKA_IMAGE=$HOME/bda-interferometry/kafka/cp-kafka_7.4.0.sif
SERVICE_DIR="$HOME/bda-interferometry/services"
LOG_DIR="$HOME/bda-interferometry/logs"
OUTPUT_DIR="$HOME/bda-interferometry/output"

# Spark Configuration
SPARK_MASTER="local[6]"              # Use 6 cores (leave 2 for Kafka/ZooKeeper)
SPARK_DRIVER_MEMORY="8g"             # Driver memory
SPARK_EXECUTOR_MEMORY="4g"           # Executor memory
SPARK_SHUFFLE_PARTITIONS="8"         # Shuffle partitions
SPARK_KAFKA_PACKAGE="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

# Kafka Configuration
KAFKA_TOPIC="visibility-stream"
KAFKA_BOOTSTRAP="localhost:9092"
KAFKA_PARTITIONS="1"

# Paths
DIRTYIMAGE_OUTPUT_DIR="./output/dirtyimage_$SLURM_JOB_ID.png"
PSF_OUTPUT_DIR="./output/psf_$SLURM_JOB_ID.png"
ANTENNA_CONFIG="./antenna_configs/skamid.cfg"
SIMULATION_CONFIG="./configs/simulation/ska-mid-band-02.json"
BDA_CONFIG="./configs/bda_config.json"
GRID_CONFIG="./configs/grid_config.json"

# ============================================================================
# CLEANUP FUNCTION
# ============================================================================
cleanup() {
    kill $CONSUMER_PID $KAFKA_PID $ZOOKEEPER_PID 2>/dev/null || true
}
trap cleanup EXIT ERR INT TERM

rm -rf ${KAFKA_DIR}/kafka-data/*
rm -rf ${KAFKA_DIR}/zookeeper-data/*

# ============================================================================
# JOB INFORMATION
# ============================================================================
echo "=== Interferometry Producer ==="
echo "Node: $SLURM_NODELIST"
echo "Job: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Date: $(date)"
echo "=============================="

# Create necessary directories
mkdir -p ${KAFKA_DIR}/{kafka-data,logs,zookeeper-data}
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}

# ============================================================================
# START SERVICES
# ============================================================================
# Start Zookeeper
singularity exec \
    --bind ${KAFKA_DIR}/zookeeper-data:/var/lib/zookeeper \
    --bind ${KAFKA_DIR}/logs:/var/log/kafka \
    ${KAFKA_IMAGE} \
    zookeeper-server-start /etc/kafka/zookeeper.properties \
    > ${LOG_DIR}/zookeeper-${SLURM_JOB_ID}.log 2>&1 &

ZOOKEEPER_PID=$! # Get Zookeeper PID
sleep 15

# Start Kafka
singularity exec \
    --bind ${KAFKA_DIR}/kafka-data:/var/lib/kafka \
    --bind ${KAFKA_DIR}/logs:/var/log/kafka \
    ${KAFKA_IMAGE} \
    kafka-server-start /etc/kafka/server.properties \
    > ${LOG_DIR}/kafka-${SLURM_JOB_ID}.log 2>&1 &


KAFKA_PID=$! # Get Kafka PID
sleep 30

# Topic creation
singularity exec ${KAFKA_IMAGE} \
    kafka-topics --create \
    --topic ${KAFKA_TOPIC} \
    --bootstrap-server ${KAFKA_BOOTSTRAP} \
    --partitions ${KAFKA_PARTITIONS} \
    --replication-factor 1 \
    --if-not-exists

# ============================================================================
# ACTIVATE ENVIRONMENT
# ============================================================================
eval "$(micromamba shell hook --shell bash)"
micromamba activate bda-env

# ============================================================================
# START CONSUMER
# ============================================================================
spark-submit \
    --master ${SPARK_MASTER} \
    --driver-memory ${SPARK_DRIVER_MEMORY} \
    --executor-memory ${SPARK_EXECUTOR_MEMORY} \
    --packages ${SPARK_KAFKA_PACKAGE} \
    --conf spark.sql.shuffle.partitions=${SPARK_SHUFFLE_PARTITIONS} \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.sql.adaptive.coalescePartitions.enabled=true \
    --conf spark.sql.adaptive.skewJoin.enabled=true \
    --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
    --conf spark.python.worker.reuse=true \
    --conf spark.sql.execution.arrow.pyspark.enabled=false \
    --conf spark.streaming.kafka.maxRatePerPartition=1000 \
    $SERVICE_DIR/consumer_service.py \
    --topic ${KAFKA_TOPIC} \
    --bootstrap-server ${KAFKA_BOOTSTRAP} \
    --bda-config ${BDA_CONFIG} \
    --grid-config ${GRID_CONFIG} \
    --dirty-image-output ${DIRTYIMAGE_OUTPUT_DIR} \
    --psf-output ${PSF_OUTPUT_DIR} \
    > ${LOG_DIR}/consumer-${SLURM_JOB_ID}.log 2>&1 &

CONSUMER_PID=$! # Get Consumer PID
sleep 10

# ============================================================================
# START PRODUCER
# ============================================================================
python $SERVICE_DIR/producer_service.py \
    --topic ${KAFKA_TOPIC} \
    --antenna-config ${ANTENNA_CONFIG} \
    --simulation-config ${SIMULATION_CONFIG} \
    > ${LOG_DIR}/producer-${SLURM_JOB_ID}.log 2>&1

PRODUCER_EXIT_CODE=$? # Capture exit code

wait $CONSUMER_PID
CONSUMER_EXIT_CODE=$?

# ============================================================================
# LOGS
# ============================================================================
echo "=== Job Completed ==="
echo "End time: $(date)"
echo "====================="

if [ $CONSUMER_EXIT_CODE -ne 0 ]; then
    exit $CONSUMER_EXIT_CODE
else
    exit $PRODUCER_EXIT_CODE
fi