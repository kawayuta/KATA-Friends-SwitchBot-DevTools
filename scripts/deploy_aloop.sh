#!/bin/bash
# 復元可能なデプロイ
KATA_IP="${1:-192.168.11.17}"
ADB="adb -s ${KATA_IP}:5555"

# バックアップ
echo "[1/4] Backing up current config..."
$ADB shell "cp /opt/wlab/sweepbot/share/ai_brain/model/voice/kws/config.json \
  /data/devtools/backup_kws_config.json"

# モジュール転送
echo "[2/4] Pushing snd-aloop.ko..."
$ADB push scripts/build-aloop/output/snd-aloop.ko /data/devtools/snd-aloop.ko

# モジュールロード
echo "[3/4] Loading module..."
$ADB shell "insmod /data/devtools/snd-aloop.ko"
$ADB shell "lsmod | grep aloop"
$ADB shell "aplay -l"

echo "[4/4] Done. Module loaded."
echo "To unload: adb shell rmmod snd_aloop"
