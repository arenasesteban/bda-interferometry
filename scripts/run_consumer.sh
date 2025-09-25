# Simple Spark Consumer Launcher for BDA Interferometry
# Launches Spark consumer to receive visibility data from Kafka

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default parameters
KAFKA_SERVERS="localhost:9092"
TOPIC="visibility-stream"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --kafka-servers)
            KAFKA_SERVERS="$2"
            shift 2
            ;;
        --topic)
            TOPIC="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--kafka-servers SERVERS] [--topic TOPIC]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set environment
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Run the consumer
echo "Starting Spark Consumer..."
echo "Kafka: $KAFKA_SERVERS | Topic: $TOPIC"
echo "Press Ctrl+C to stop"

cd "$PROJECT_ROOT"
python services/consumer_service.py \
    --kafka-servers "$KAFKA_SERVERS" \
    --topic "$TOPIC"
