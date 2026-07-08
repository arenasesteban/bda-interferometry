#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

python -m radio_pipeline producer