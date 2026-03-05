#!/bin/bash
# Deploy Kata Friends DevTools to device via ADB
# Usage: bash scripts/deploy_devtools.sh [KATA_IP]
#
# Device filesystem notes:
#   - rootfs is overlayfs (ro by default)
#   - /data/ is writable ext4
#   - /overlay/overlay_upper flag file must exist for overlay rw + persistence
#   - /data/overlay_upper/ is the overlay upper layer
#   - Without the flag, init clears overlay_upper/* on every boot

set -euo pipefail

KATA_IP="${1:-${KATA_IP:-192.168.11.17}}"
ADB_TARGET="${KATA_IP}:5555"
DEVICE_DIR="/data/devtools"
OVERLAY_UPPER="/data/overlay_upper"
SYSTEMD_DIR="${OVERLAY_UPPER}/etc/systemd/system"
SERVICE_NAME="kata-devtools"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/../devtools/ondevice"

echo "=== Kata Friends DevTools Deploy ==="
echo "Target: ${ADB_TARGET}"
echo "Source: ${SOURCE_DIR}"
echo ""

# Connect ADB
echo "[1/6] ADB connect..."
adb connect "${ADB_TARGET}" 2>/dev/null || true
adb -s "${ADB_TARGET}" wait-for-device
echo "  OK"

# Create directories on device
echo "[2/6] Creating directories..."
adb -s "${ADB_TARGET}" shell "mkdir -p ${DEVICE_DIR}/static"
echo "  OK"

# Push application files
echo "[3/6] Pushing files..."
adb -s "${ADB_TARGET}" push "${SOURCE_DIR}/app_flask.py" "${DEVICE_DIR}/app_flask.py"
adb -s "${ADB_TARGET}" push "${SOURCE_DIR}/zmq_publish.py" "${DEVICE_DIR}/zmq_publish.py"
adb -s "${ADB_TARGET}" push "${SOURCE_DIR}/static/index.html" "${DEVICE_DIR}/static/index.html"

# Create start.sh wrapper
adb -s "${ADB_TARGET}" shell "cat > ${DEVICE_DIR}/start.sh << 'STARTEOF'
#!/bin/bash
cd /data/devtools
exec python3 app_flask.py
STARTEOF
chmod +x ${DEVICE_DIR}/start.sh"
echo "  OK"

# Enable overlay persistence flag
echo "[4/6] Enabling overlay persistence..."
adb -s "${ADB_TARGET}" shell "touch /overlay/overlay_upper"
echo "  OK"

# Set up systemd service in overlay upper
echo "[5/6] Setting up systemd service..."
adb -s "${ADB_TARGET}" shell "
mkdir -p ${SYSTEMD_DIR}/multi-user.target.wants

cat > ${SYSTEMD_DIR}/${SERVICE_NAME}.service << 'UNIT'
[Unit]
Description=Kata Friends DevTools
After=network.target master.service

[Service]
Type=simple
User=wlab
WorkingDirectory=/data/devtools
ExecStart=/data/devtools/start.sh
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT

ln -sf /etc/systemd/system/${SERVICE_NAME}.service \
  ${SYSTEMD_DIR}/multi-user.target.wants/${SERVICE_NAME}.service
"
echo "  OK"

# Try to start immediately, or advise reboot
echo "[6/6] Starting service..."
if adb -s "${ADB_TARGET}" shell "systemctl daemon-reload && systemctl enable --now ${SERVICE_NAME}" 2>/dev/null; then
  sleep 2
  adb -s "${ADB_TARGET}" shell "systemctl status ${SERVICE_NAME} --no-pager -l" || true
else
  echo "  Service file not yet visible (overlayfs needs reboot)."
  echo "  Run: adb -s ${ADB_TARGET} reboot"
  echo "  After reboot, service will start automatically."
fi

echo ""
echo "=== Deploy complete ==="
echo "Access: http://${KATA_IP}:9001"
