#!/bin/bash
#SBATCH -J bda-interferometry
#SBATCH -p debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -o logs/bda-interferometry-%j.out
#SBATCH -e logs/bda-interferometry-%j.err
#SBATCH --mail-user=esteban.arenas.a@usach.cl
#SBATCH --mail-type=ALL

set -euo pipefail

# ==============================================================================
# ENTORNO
# ==============================================================================
ml purge
ml singularityCE

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate bda-env

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT="$HOME/bda-interferometry"
SERVICE_DIR="$PROJECT_ROOT/services"
KAFKA_DIR="$HOME/kafka-hpc"
KAFKA_IMAGE="$PROJECT_ROOT/kafka/cp-kafka_7.4.0.sif"

SPARK_HOME="$HOME/spark-3.5.0"
export SPARK_HOME
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
export PYSPARK_PYTHON="$(which python)"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
export PYTHONPATH="$PROJECT_ROOT:$SERVICE_DIR:${PYTHONPATH:-}"

LOG_DIR="$PROJECT_ROOT/logs/$SLURM_JOB_ID"
OUTPUT_DIR="$PROJECT_ROOT/output/$SLURM_JOB_ID"
SPARK_EVENTS_DIR="$LOG_DIR/spark-events"
SPARK_LOCAL_BASE="/tmp/$USER/bda-interferometry/$SLURM_JOB_ID"

ANTENNA_CONFIG="$PROJECT_ROOT/antenna_configs/skamid.cfg"
SIMULATION_CONFIG="$PROJECT_ROOT/configs/simulation/ska-mid-band-02.json"
BDA_CONFIG="$PROJECT_ROOT/configs/bda_config.json"
GRID_CONFIG="$PROJECT_ROOT/configs/grid_config.json"

# ==============================================================================
# PARÁMETROS CONFIGURABLES
# ==============================================================================
DECORR_FACTOR="${DECORR_FACTOR:-0.95}"
FOV="${FOV:-0.0003}"
KAFKA_TOPIC="visibility-stream"
KAFKA_PARTITIONS="${KAFKA_PARTITIONS:-16}"
SPARK_KAFKA_PACKAGE="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"
SPARK_MASTER_PORT=7077
SPARK_MASTER_WEBUI_PORT=8080

# ==============================================================================
# VALIDACIONES
# ==============================================================================
[ -x "$SPARK_HOME/bin/spark-submit" ] || { echo "[ERROR] spark-submit no encontrado"; exit 1; }
[ -x "$SPARK_HOME/bin/spark-class"  ] || { echo "[ERROR] spark-class no encontrado"; exit 1; }
[ -f "$KAFKA_IMAGE" ]                 || { echo "[ERROR] Imagen Kafka no encontrada: $KAFKA_IMAGE"; exit 1; }
command -v java >/dev/null 2>&1       || { echo "[ERROR] Java no disponible"; exit 1; }
[ -f "$ANTENNA_CONFIG" ]              || { echo "[ERROR] Falta: $ANTENNA_CONFIG"; exit 1; }
[ -f "$SIMULATION_CONFIG" ]           || { echo "[ERROR] Falta: $SIMULATION_CONFIG"; exit 1; }
[ -f "$BDA_CONFIG" ]                  || { echo "[ERROR] Falta: $BDA_CONFIG"; exit 1; }
[ -f "$GRID_CONFIG" ]                 || { echo "[ERROR] Falta: $GRID_CONFIG"; exit 1; }

# ==============================================================================
# NODOS
# ==============================================================================
mapfile -t NODE_LIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_NODE="${NODE_LIST[0]}"
WORKER_NODES=("${NODE_LIST[@]:1}")
N_WORKERS="${#WORKER_NODES[@]}"

[ "$N_WORKERS" -ge 1 ] || { echo "[ERROR] Se requiere al menos 1 worker (mínimo 2 nodos)"; exit 1; }

MASTER_IP=$(getent ahostsv4 "$MASTER_NODE" | awk 'NR==1{print $1}')
[ -n "$MASTER_IP" ] || { echo "[ERROR] No se pudo resolver IP de $MASTER_NODE"; exit 1; }

SPARK_MASTER_URL="spark://${MASTER_IP}:${SPARK_MASTER_PORT}"
KAFKA_BOOTSTRAP="${MASTER_IP}:9092"

echo "==================================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Master      : $MASTER_NODE ($MASTER_IP)"
echo "Workers     : ${WORKER_NODES[*]}"
echo "Nodos total : $((N_WORKERS + 1))"
echo "CPUs/nodo   : $SLURM_CPUS_PER_TASK"
echo "Mem/nodo    : ${SLURM_MEM_PER_NODE:-?} MB"
echo "==================================================="

# ==============================================================================
# RECURSOS SPARK
# ==============================================================================
TOTAL_MEM_MB="${SLURM_MEM_PER_NODE:-102400}"
SPARK_WORKER_CORES=$(( SLURM_CPUS_PER_TASK > 2 ? SLURM_CPUS_PER_TASK - 2 : 1 ))
SPARK_WORKER_MEMORY="$(( TOTAL_MEM_MB * 85 / 100 ))m"
SPARK_EXECUTOR_MEMORY="$(( TOTAL_MEM_MB * 85 * 90 / 10000 ))m"
SPARK_DRIVER_MEMORY_MB=$(( TOTAL_MEM_MB * 12 / 100 ))
[ "$SPARK_DRIVER_MEMORY_MB" -ge 4096 ] || SPARK_DRIVER_MEMORY_MB=4096
SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY_MB}m"
TOTAL_CORES=$(( N_WORKERS * SPARK_WORKER_CORES ))
SHUFFLE_PARTITIONS=$(( TOTAL_CORES * 2 > 8 ? TOTAL_CORES * 2 : 8 ))

# ==============================================================================
# DIRECTORIOS
# ==============================================================================
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$SPARK_EVENTS_DIR"
mkdir -p "$KAFKA_DIR/kafka-data" "$KAFKA_DIR/zookeeper-data" "$KAFKA_DIR/logs"
rm -rf "$KAFKA_DIR/kafka-data/"* "$KAFKA_DIR/zookeeper-data/"*

for node in "${NODE_LIST[@]}"; do
    srun --overlap --exact -N1 -n1 -c1 -w "$node" \
        bash --noprofile --norc -c "mkdir -p '$SPARK_LOCAL_BASE/worker' '$SPARK_LOCAL_BASE/blockmgr'"
done

# ==============================================================================
# CLEANUP
# ==============================================================================
MASTER_PID=""
WORKER_STEP_PIDS=()
CONSUMER_PID=""
ZOOKEEPER_PID=""
KAFKA_PID=""

cleanup() {
    set +e
    echo "[cleanup] Deteniendo servicios..."

    for pid in "$CONSUMER_PID" "$KAFKA_PID" "$ZOOKEEPER_PID" "$MASTER_PID"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done

    for pid in "${WORKER_STEP_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done

    pkill -f 'org.apache.spark.deploy.master.Master' 2>/dev/null || true
    pkill -f 'kafka.Kafka'    2>/dev/null || true
    pkill -f 'QuorumPeerMain' 2>/dev/null || true

    for worker in "${WORKER_NODES[@]}"; do
        srun --overlap --exact -N1 -n1 -c1 -w "$worker" \
            pkill -f 'org.apache.spark.deploy.worker.Worker' 2>/dev/null || true &
    done

    wait
    echo "[cleanup] Listo"
}
trap cleanup EXIT INT TERM

# ==============================================================================
# 1) SPARK MASTER
# ==============================================================================
echo "[1/6] Iniciando Spark master..."

unset LD_LIBRARY_PATH
export SPARK_LOCAL_DIRS="$SPARK_LOCAL_BASE"
export SPARK_LOG_DIR="$LOG_DIR"
export SPARK_MASTER_HOST="$MASTER_IP"

pkill -f 'org.apache.spark.deploy.master.Master' 2>/dev/null || true

"$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.master.Master \
    --host "$MASTER_IP" \
    --port "$SPARK_MASTER_PORT" \
    --webui-port "$SPARK_MASTER_WEBUI_PORT" \
    > "$LOG_DIR/spark-master.log" 2>&1 &
MASTER_PID=$!

sleep 10
for _ in {1..20}; do
    curl -s "http://${MASTER_IP}:${SPARK_MASTER_WEBUI_PORT}/json/" >/dev/null 2>&1 && break
    sleep 3
done
curl -s "http://${MASTER_IP}:${SPARK_MASTER_WEBUI_PORT}/json/" >/dev/null 2>&1 \
    || { echo "[ERROR] Spark master no respondió"; tail -n 40 "$LOG_DIR/spark-master.log"; exit 1; }

echo "  ✓ Spark master OK → $SPARK_MASTER_URL"

# ==============================================================================
# 2) SPARK WORKERS
# ==============================================================================
echo "[2/6] Iniciando $N_WORKERS worker(s)..."

WORKER_IPS=()
for worker in "${WORKER_NODES[@]}"; do
    worker_ip=$(getent ahostsv4 "$worker" | awk 'NR==1{print $1}')
    [ -n "$worker_ip" ] || { echo "[ERROR] No se pudo resolver IP de $worker"; exit 1; }
    WORKER_IPS+=("$worker_ip")
done

for i in "${!WORKER_NODES[@]}"; do
    worker="${WORKER_NODES[$i]}"
    worker_ip="${WORKER_IPS[$i]}"

    WORKER_SCRIPT="$LOG_DIR/worker-launch-${worker}.sh"
    cat > "$WORKER_SCRIPT" << EOF
#!/bin/bash
unset LD_LIBRARY_PATH
export SPARK_HOME="$SPARK_HOME"
export SPARK_LOCAL_DIRS="$SPARK_LOCAL_BASE"
export SPARK_LOG_DIR="$LOG_DIR"
export SPARK_WORKER_DIR="$SPARK_LOCAL_BASE/worker"
export SPARK_LOCAL_IP="$worker_ip"
export SPARK_PUBLIC_DNS="$worker_ip"
pkill -f 'org.apache.spark.deploy.worker.Worker' 2>/dev/null || true
exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker \
    --host "$worker_ip" \
    --webui-port 8081 \
    --cores "$SPARK_WORKER_CORES" \
    --memory "$SPARK_WORKER_MEMORY" \
    --work-dir "$SPARK_LOCAL_BASE/worker" \
    "$SPARK_MASTER_URL"
EOF
    chmod +x "$WORKER_SCRIPT"

    srun --overlap --exact -N1 -n1 -c"$SLURM_CPUS_PER_TASK" -w "$worker" \
        bash --noprofile --norc "$WORKER_SCRIPT" \
        > "$LOG_DIR/spark-worker-${worker}.log" 2>&1 &
    WORKER_STEP_PIDS+=($!)
done

sleep 20
REGISTERED=0
for _ in {1..12}; do
    REGISTERED=$(curl -s "http://${MASTER_IP}:${SPARK_MASTER_WEBUI_PORT}/json/" \
        | "$PYSPARK_PYTHON" -c "import sys,json; print(len(json.load(sys.stdin).get('workers',[])))" \
        2>/dev/null || echo 0)
    [ "$REGISTERED" -ge "$N_WORKERS" ] && break
    sleep 5
done

[ "$REGISTERED" -ge "$N_WORKERS" ] \
    || { echo "[ERROR] $REGISTERED/$N_WORKERS workers registrados"; tail -n 40 "$LOG_DIR/spark-worker-${WORKER_NODES[0]}.log"; exit 1; }

echo "  ✓ Spark workers OK ($REGISTERED/$N_WORKERS)"

# ==============================================================================
# 3) ZOOKEEPER
# ==============================================================================
echo "[3/6] Iniciando Zookeeper..."

singularity exec \
    --bind "$KAFKA_DIR/zookeeper-data:/var/lib/zookeeper" \
    --bind "$KAFKA_DIR/logs:/var/log/kafka" \
    "$KAFKA_IMAGE" \
    zookeeper-server-start /etc/kafka/zookeeper.properties \
    > "$LOG_DIR/zookeeper.log" 2>&1 &
ZOOKEEPER_PID=$!

sleep 12
kill -0 "$ZOOKEEPER_PID" 2>/dev/null \
    || { echo "[ERROR] Zookeeper no arrancó"; tail -n 40 "$LOG_DIR/zookeeper.log"; exit 1; }

echo "  ✓ Zookeeper OK"

# ==============================================================================
# 4) KAFKA
# ==============================================================================
echo "[4/6] Iniciando Kafka..."

# Se genera un server.properties con los listeners correctos para este job,
# derivado del server.properties base de la imagen.
KAFKA_PROPS="$LOG_DIR/kafka-server.properties"
singularity exec "$KAFKA_IMAGE" cat /etc/kafka/server.properties > "$KAFKA_PROPS"
printf '\nlisteners=PLAINTEXT://0.0.0.0:9092\nadvertised.listeners=PLAINTEXT://%s:9092\n' "$MASTER_IP" >> "$KAFKA_PROPS"

singularity exec \
    --bind "$KAFKA_DIR/kafka-data:/var/lib/kafka" \
    --bind "$KAFKA_DIR/logs:/var/log/kafka" \
    --bind "$KAFKA_PROPS:/etc/kafka/server.properties" \
    "$KAFKA_IMAGE" \
    kafka-server-start /etc/kafka/server.properties \
    > "$LOG_DIR/kafka.log" 2>&1 &
KAFKA_PID=$!

sleep 20
KAFKA_OK=0
for _ in {1..12}; do
    singularity exec "$KAFKA_IMAGE" \
        kafka-topics --bootstrap-server "$KAFKA_BOOTSTRAP" --list >/dev/null 2>&1 \
        && KAFKA_OK=1 && break
    sleep 5
done
[ "$KAFKA_OK" -eq 1 ] \
    || { echo "[ERROR] Kafka no respondió"; tail -n 40 "$LOG_DIR/kafka.log"; exit 1; }

singularity exec "$KAFKA_IMAGE" kafka-topics \
    --create --if-not-exists \
    --topic "$KAFKA_TOPIC" \
    --bootstrap-server "$KAFKA_BOOTSTRAP" \
    --partitions "$KAFKA_PARTITIONS" \
    --replication-factor 1 \
    --config max.message.bytes=104857600 \
    --config segment.bytes=1073741824 \
    --config retention.bytes=10737418240 \
    > "$LOG_DIR/kafka-topic.log" 2>&1

echo "  ✓ Kafka OK → $KAFKA_BOOTSTRAP | topic: $KAFKA_TOPIC ($KAFKA_PARTITIONS particiones)"

# ==============================================================================
# 5) CONSUMER
# ==============================================================================
echo "[5/6] Lanzando consumer..."

"$SPARK_HOME/bin/spark-submit" \
    --master "$SPARK_MASTER_URL" \
    --deploy-mode client \
    --driver-memory "$SPARK_DRIVER_MEMORY" \
    --executor-cores "$SPARK_WORKER_CORES" \
    --executor-memory "$SPARK_EXECUTOR_MEMORY" \
    --packages "$SPARK_KAFKA_PACKAGE" \
    --conf "spark.executor.instances=$N_WORKERS" \
    --conf "spark.cores.max=$TOTAL_CORES" \
    --conf "spark.sql.shuffle.partitions=$SHUFFLE_PARTITIONS" \
    --conf "spark.sql.adaptive.enabled=true" \
    --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.driver.maxResultSize=4g" \
    --conf "spark.local.dir=$SPARK_LOCAL_BASE" \
    --conf "spark.eventLog.enabled=true" \
    --conf "spark.eventLog.dir=$SPARK_EVENTS_DIR" \
    --conf "spark.scheduler.minRegisteredResourcesRatio=1.0" \
    --conf "spark.scheduler.maxRegisteredResourcesWaitingTime=120s" \
    --conf "spark.driver.host=$MASTER_IP" \
    --conf "spark.driver.bindAddress=0.0.0.0" \
    --conf "spark.network.timeout=600s" \
    --conf "spark.rpc.askTimeout=600s" \
    --conf "spark.sql.streaming.forceDeleteTempCheckpointLocation=true" \
    --conf "spark.streaming.stopGracefullyOnShutdown=true" \
    --conf "spark.pyspark.python=$PYSPARK_PYTHON" \
    --conf "spark.executorEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
    --conf "spark.executorEnv.PYTHONPATH=$PYTHONPATH" \
    "$SERVICE_DIR/consumer_service.py" \
        --topic            "$KAFKA_TOPIC" \
        --bootstrap-server "$KAFKA_BOOTSTRAP" \
        --bda-config       "$BDA_CONFIG" \
        --grid-config      "$GRID_CONFIG" \
        --slurm-job-id     "$SLURM_JOB_ID" \
        --decorr-factor    "$DECORR_FACTOR" \
        --fov              "$FOV" \
    > "$LOG_DIR/consumer.log" 2>&1 &
CONSUMER_PID=$!

sleep 12
kill -0 "$CONSUMER_PID" 2>/dev/null \
    || { echo "[ERROR] Consumer terminó prematuramente"; tail -n 40 "$LOG_DIR/consumer.log"; exit 1; }

echo "  ✓ Consumer OK (PID $CONSUMER_PID)"

# ==============================================================================
# 6) PRODUCER
# ==============================================================================
echo "[6/6] Lanzando producer..."

python "$SERVICE_DIR/producer_service.py" \
    --topic             "$KAFKA_TOPIC" \
    --antenna-config    "$ANTENNA_CONFIG" \
    --simulation-config "$SIMULATION_CONFIG" \
    > "$LOG_DIR/producer.log" 2>&1

PRODUCER_EXIT=$?
echo "  ✓ Producer finalizado (exit $PRODUCER_EXIT)"

# ==============================================================================
# ESPERAR CONSUMER Y REPORTAR
# ==============================================================================
wait "$CONSUMER_PID"
CONSUMER_EXIT=$?

echo ""
echo "================ JOB COMPLETADO ================"
echo "Producer : exit $PRODUCER_EXIT"
echo "Consumer : exit $CONSUMER_EXIT"
echo "Logs     : $LOG_DIR"
echo "Output   : $OUTPUT_DIR"
echo "================================================="

[ "$PRODUCER_EXIT" -eq 0 ] && [ "$CONSUMER_EXIT" -eq 0 ] || exit 1
exit 0