#!/bin/bash
# Setup TTS/ASR (Qwen3-TTS + Whisper) on Kata Friends device via ADB
# Usage: bash devtools/setup_tts.sh [KATA_IP]
#
# This script automates:
#   1. Disk space check (need ~2 GB)
#   2. TTS RKNN model download from HuggingFace
#   3. Whisper RKNN model download (rknn_model_zoo)
#   4. Deploy flask_server_tts.py
#   5. Create systemd service + start

set -euo pipefail

KATA_IP="${1:-${KATA_IP:-192.168.11.17}}"
ADB_TARGET="${KATA_IP}:5555"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."

# Paths on device
TTS_MODEL_DIR="/data/ai_brain/tts"
ASR_MODEL_DIR="/data/ai_brain/asr"
DEVTOOLS_DIR="/data/devtools"
BIN_DIR="/opt/wlab/sweepbot/bin"
OVERLAY_UPPER="/data/overlay_upper"

# Model download URLs (update after converting and uploading to HuggingFace)
HF_TTS_BASE="https://huggingface.co/JiahaoLi/Qwen3-TTS-0.6B-RKNN-RK3576/resolve/main"
HF_ASR_BASE="https://huggingface.co/JiahaoLi/whisper-rknn-rk3576/resolve/main"

# TTS RKNN model files (~1.6GB total)
TTS_MODELS=(
    "talker_prefill_q.rknn"
    "talker_decode_q.rknn"
    "text_project_q.rknn"
    "tokenizer12hz_decode_q.rknn"
    "code_predictor_q.rknn"
    "tokenizer12hz_encode_q.rknn"
    "code_predictor_embed_q.rknn"
    "speaker_encoder_q.rknn"
    "codec_embed_q.rknn"
)

# Whisper RKNN model files
ASR_MODELS=(
    "whisper_encoder.rknn"
    "whisper_decoder.rknn"
)

echo "============================================"
echo "  Kata Friends TTS/ASR Setup"
echo "============================================"
echo "Target: ${ADB_TARGET}"
echo ""

# --- Helper ---
adb_sh() {
    adb -s "${ADB_TARGET}" shell "$@"
}

# --- ADB Connect ---
echo "[0/5] ADB connect..."
adb connect "${ADB_TARGET}" 2>/dev/null || true
adb -s "${ADB_TARGET}" wait-for-device
echo "  Connected."
echo ""

# ============================================================
# [1/5] Disk space check
# ============================================================
echo "[1/5] Checking disk space on /data..."
AVAIL_KB=$(adb_sh "df /data | tail -1 | awk '{print \$4}'" | tr -d '\r\n')
AVAIL_MB=$((AVAIL_KB / 1024))
echo "  Available: ${AVAIL_MB} MB"
if [ "${AVAIL_MB}" -lt 2048 ]; then
    echo "  ERROR: Need at least 2 GB free on /data (have ${AVAIL_MB} MB)"
    exit 1
fi
echo "  OK (>= 2 GB)"
echo ""

# ============================================================
# [2/5] TTS RKNN model download
# ============================================================
echo "[2/5] Downloading TTS RKNN models..."
adb_sh "mkdir -p ${TTS_MODEL_DIR}"

download_model() {
    local filename="$1"
    local base_url="$2"
    local dest_dir="$3"
    local url="${base_url}/${filename}"
    local dest="${dest_dir}/${filename}"

    # Check if file already exists with reasonable size (> 1MB)
    EXISTING_SIZE=$(adb_sh "[ -f ${dest} ] && stat -c%s ${dest} 2>/dev/null || echo 0" | tr -d '\r\n')

    if [ "${EXISTING_SIZE}" -gt 1000000 ]; then
        echo "  ${filename}: already exists ($(( EXISTING_SIZE / 1024 / 1024 )) MB), skipping"
        return 0
    fi

    echo "  ${filename}: downloading..."
    adb_sh "wget -c -q --show-progress -O ${dest} '${url}'" || {
        echo "  WARNING: Failed to download ${filename}"
        echo "  URL: ${url}"
        echo "  (Model may not be uploaded yet. Upload after running convert_tts_rknn.py)"
        return 1
    }

    FINAL_SIZE=$(adb_sh "stat -c%s ${dest}" | tr -d '\r\n')
    echo "  ${filename}: done ($(( FINAL_SIZE / 1024 / 1024 )) MB)"
}

TTS_FAIL=0
for model in "${TTS_MODELS[@]}"; do
    download_model "${model}" "${HF_TTS_BASE}" "${TTS_MODEL_DIR}" || TTS_FAIL=$((TTS_FAIL + 1))
done

if [ "${TTS_FAIL}" -gt 0 ]; then
    echo "  WARNING: ${TTS_FAIL} TTS model(s) failed to download."
    echo "  Make sure models are uploaded to HuggingFace first."
fi
echo ""

# ============================================================
# [3/5] Whisper RKNN model download
# ============================================================
echo "[3/5] Downloading Whisper RKNN models..."
adb_sh "mkdir -p ${ASR_MODEL_DIR}"

ASR_FAIL=0
for model in "${ASR_MODELS[@]}"; do
    download_model "${model}" "${HF_ASR_BASE}" "${ASR_MODEL_DIR}" || ASR_FAIL=$((ASR_FAIL + 1))
done

# Also download vocab.json for Whisper
download_model "vocab.json" "${HF_ASR_BASE}" "${ASR_MODEL_DIR}" || true

if [ "${ASR_FAIL}" -gt 0 ]; then
    echo "  WARNING: ${ASR_FAIL} ASR model(s) failed to download."
fi
echo ""

# ============================================================
# [4/5] Deploy flask_server_tts.py
# ============================================================
echo "[4/5] Deploying flask_server_tts.py..."
LOCAL_TTS="${REPO_DIR}/devtools/ondevice/flask_server_tts.py"

if [ ! -f "${LOCAL_TTS}" ]; then
    echo "  ERROR: ${LOCAL_TTS} not found"
    exit 1
fi

# Create devtools directory on device
adb_sh "mkdir -p ${DEVTOOLS_DIR}/tts"

# Push to devtools dir
adb -s "${ADB_TARGET}" push "${LOCAL_TTS}" "${DEVTOOLS_DIR}/flask_server_tts.py"
echo "  Pushed to ${DEVTOOLS_DIR}/flask_server_tts.py"

# Also push to overlay upper for persistence
adb_sh "mkdir -p ${OVERLAY_UPPER}${DEVTOOLS_DIR}"
adb_sh "cp ${DEVTOOLS_DIR}/flask_server_tts.py ${OVERLAY_UPPER}${DEVTOOLS_DIR}/flask_server_tts.py"
echo "  Copied to overlay upper for persistence"

# Also deploy updated app_flask.py and index.html
LOCAL_APP="${REPO_DIR}/devtools/ondevice/app_flask.py"
LOCAL_HTML="${REPO_DIR}/devtools/ondevice/static/index.html"

if [ -f "${LOCAL_APP}" ]; then
    adb -s "${ADB_TARGET}" push "${LOCAL_APP}" "${DEVTOOLS_DIR}/app_flask.py"
    adb_sh "mkdir -p ${OVERLAY_UPPER}${DEVTOOLS_DIR}"
    adb_sh "cp ${DEVTOOLS_DIR}/app_flask.py ${OVERLAY_UPPER}${DEVTOOLS_DIR}/app_flask.py"
    echo "  Updated app_flask.py"
fi

if [ -f "${LOCAL_HTML}" ]; then
    adb_sh "mkdir -p ${DEVTOOLS_DIR}/static"
    adb -s "${ADB_TARGET}" push "${LOCAL_HTML}" "${DEVTOOLS_DIR}/static/index.html"
    adb_sh "mkdir -p ${OVERLAY_UPPER}${DEVTOOLS_DIR}/static"
    adb_sh "cp ${DEVTOOLS_DIR}/static/index.html ${OVERLAY_UPPER}${DEVTOOLS_DIR}/static/index.html"
    echo "  Updated static/index.html"
fi

echo "  OK"
echo ""

# ============================================================
# [5/5] Systemd service + start
# ============================================================
echo "[5/5] Creating systemd service for TTS/ASR..."

SERVICE_FILE="/etc/systemd/system/tts_server.service"
OVERLAY_SERVICE="${OVERLAY_UPPER}${SERVICE_FILE}"

# Create service file on device
adb_sh "mkdir -p ${OVERLAY_UPPER}/etc/systemd/system"
adb_sh "cat > ${OVERLAY_SERVICE}" << 'SERVICEEOF'
[Unit]
Description=Kata Friends TTS/ASR Server (Qwen3-TTS + Whisper)
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /data/devtools/flask_server_tts.py --port 8084
WorkingDirectory=/data/devtools
Restart=on-failure
RestartSec=10
Environment=TZ=Asia/Shanghai

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Copy to merged path
adb_sh "cp ${OVERLAY_SERVICE} ${SERVICE_FILE}"

# Reload and enable
adb_sh "systemctl daemon-reload"
adb_sh "systemctl enable tts_server"
adb_sh "systemctl restart tts_server"

echo "  Service created and started."

# Wait for service to become active
echo "  Waiting for tts_server service..."
for i in $(seq 1 30); do
    STATUS=$(adb_sh "systemctl is-active tts_server 2>/dev/null" 2>/dev/null | tr -d '\r' || echo "unknown")
    if [ "${STATUS}" = "active" ]; then
        echo "  tts_server service is active (${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  WARNING: tts_server did not become active within 30s"
        echo "  Status: ${STATUS}"
        adb_sh "journalctl -u tts_server --no-pager -n 20" 2>/dev/null || true
    fi
    sleep 1
done

# Quick health check
sleep 2
echo ""
echo "  Running health check..."
TEST_RESULT=$(curl -s --max-time 5 "http://${KATA_IP}:8084/tts/status" 2>/dev/null || echo "CURL_FAILED")

if [ "${TEST_RESULT}" = "CURL_FAILED" ]; then
    echo "  WARNING: TTS server not responding yet."
    echo "  Service may still be starting. Check with:"
    echo "  curl http://${KATA_IP}:8084/tts/status"
else
    echo "  TTS server status: ${TEST_RESULT}"
fi

# Also restart devtools to pick up app_flask.py changes
echo ""
echo "  Restarting DevTools (app_flask.py)..."
adb_sh "systemctl restart devtools 2>/dev/null" || adb_sh "pkill -f app_flask.py; cd ${DEVTOOLS_DIR} && python3 app_flask.py &" 2>/dev/null || true

echo ""
echo "============================================"
echo "  TTS/ASR Setup Complete!"
echo "============================================"
echo ""
echo "Endpoints:"
echo "  TTS synthesize: http://${KATA_IP}:8084/tts/synthesize"
echo "  TTS status:     http://${KATA_IP}:8084/tts/status"
echo "  ASR transcribe: http://${KATA_IP}:8084/asr/transcribe"
echo "  DevTools UI:    http://${KATA_IP}:9001"
echo ""
echo "Next steps:"
echo "  1. Upload reference audio via DevTools UI (Custom LLM tab)"
echo "  2. Enable TTS checkbox and run Custom LLM"
echo ""
