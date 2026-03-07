#!/bin/bash
# 元に戻す
KATA_IP="${1:-192.168.11.17}"
ADB="adb -s ${KATA_IP}:5555"

echo "Unloading snd-aloop..."
$ADB shell "rmmod snd_aloop 2>/dev/null; true"

echo "Restoring KWS config..."
$ADB shell "cp /data/devtools/backup_kws_config.json \
  /opt/wlab/sweepbot/share/ai_brain/model/voice/kws/config.json 2>/dev/null; true"

echo "Restarting pet_voice..."
$ADB shell "systemctl restart master 2>/dev/null; true"

echo "Restored to original state."
