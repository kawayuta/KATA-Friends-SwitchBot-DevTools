#!/bin/bash
# Cross-compile libsv_lang.so for aarch64
# LD_PRELOAD library to override SenseVoice ASR language setting
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "$OUTPUT_DIR"

echo "[1/2] Compiling libsv_lang.so in arm64 container..."
docker run --rm --platform linux/arm64 \
  -v "${SCRIPT_DIR}:/src" \
  -v "${OUTPUT_DIR}:/output" \
  ubuntu:22.04 bash -c '
    apt-get update && apt-get install -y gcc
    gcc -shared -fPIC -O2 -Wall \
      /src/sv_lang_override.c \
      -ldl \
      -o /output/libsv_lang.so
    echo "Build successful"
  '

echo "[2/2] Done."
ls -la "$OUTPUT_DIR/libsv_lang.so"
file "$OUTPUT_DIR/libsv_lang.so"
