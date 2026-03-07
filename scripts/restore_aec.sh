#!/bin/bash
# Restore device to pre-AEC state
# NOTE: Uses cp on device (adb push to overlayfs merged paths creates 0-byte files)
set -e
KATA_IP="${1:-192.168.11.17}"
ADB="adb -s ${KATA_IP}:5555"

MERGED_ASOUND="/etc/asound.conf"
UPPER_ASOUND="/data/overlay_upper/etc/asound.conf"
KWS_MERGED="/opt/wlab/sweepbot/share/ai_brain/model/voice/kws/config.json"
KWS_UPPER="/data/overlay_upper/opt/wlab/sweepbot/share/ai_brain/model/voice/kws/config.json"

echo "[1/4] Stopping aec_daemon..."
$ADB shell "start-stop-daemon -K -x /data/devtools/aec_daemon 2>/dev/null; true"
$ADB shell "rm -f /tmp/aec_daemon.pid 2>/dev/null; true"

echo "[2/4] Restoring asound.conf..."
$ADB shell "cp /data/devtools/backup_asound.conf $MERGED_ASOUND"
$ADB shell "cp /data/devtools/backup_asound.conf $UPPER_ASOUND 2>/dev/null; true"

echo "[3/4] Restoring KWS config..."
$ADB shell "cp /data/devtools/backup_kws_config.json $KWS_MERGED"
$ADB shell "cp /data/devtools/backup_kws_config.json $KWS_UPPER 2>/dev/null; true"

echo "[4/4] Restarting master service..."
$ADB shell "systemctl restart master"

echo "Restored to pre-AEC state."
