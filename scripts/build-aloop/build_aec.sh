#!/bin/bash
# Cross-compile aec_daemon for aarch64
# Uses arm64 Docker container for native compilation (works on Apple Silicon)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "$OUTPUT_DIR"

echo "[1/2] Compiling aec_daemon in arm64 container..."
docker run --rm --platform linux/arm64 \
  -v "${SCRIPT_DIR}:/src" \
  -v "${OUTPUT_DIR}:/output" \
  ubuntu:22.04 bash -c '
    apt-get update && apt-get install -y gcc libasound2-dev libspeexdsp-dev
    gcc /src/aec_daemon.c \
      -lasound -lspeexdsp -lpthread \
      -O2 -Wall \
      -o /output/aec_daemon
    echo "Build successful"
  '

echo "[2/2] Done."
ls -la "$OUTPUT_DIR/aec_daemon"
file "$OUTPUT_DIR/aec_daemon"
