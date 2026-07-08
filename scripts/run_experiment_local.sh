#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

echo "Starting consumer..."
bash scripts/run_consumer_local.sh &

sleep 40

echo "Starting producer..."
bash scripts/run_producer_local.sh