#!/bin/bash
# Deploy AEC pipeline to device
# NOTE: On overlayfs, adb push to merged paths creates 0-byte files.
# Always push to /data/devtools/ staging area then cp on device.
set -e
KATA_IP="${1:-192.168.11.17}"
ADB="adb -s ${KATA_IP}:5555"

BINARY="scripts/build-aloop/output/aec_daemon"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run scripts/build-aloop/build_aec.sh first."
    exit 1
fi

MERGED_ASOUND="/etc/asound.conf"
UPPER_ASOUND="/data/overlay_upper/etc/asound.conf"
KWS_MERGED="/opt/wlab/sweepbot/share/ai_brain/model/voice/kws/config.json"
KWS_UPPER="/data/overlay_upper/opt/wlab/sweepbot/share/ai_brain/model/voice/kws/config.json"
LOWER_ASOUND="/app/etc/asound.conf"
LOWER_KWS="/app/opt/wlab/sweepbot/share/ai_brain/model/voice/kws/config.json"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "[1/6] Backing up original configs (from lower layer)..."
$ADB shell "cat $LOWER_ASOUND" > "$TMPDIR/backup_asound.conf"
$ADB shell "cat $LOWER_KWS" > "$TMPDIR/backup_kws_config.json"
$ADB push "$TMPDIR/backup_asound.conf" /data/devtools/backup_asound.conf
$ADB push "$TMPDIR/backup_kws_config.json" /data/devtools/backup_kws_config.json

echo "[2/6] Pushing aec_daemon binary..."
$ADB push "$BINARY" /data/devtools/aec_daemon
$ADB shell "chmod +x /data/devtools/aec_daemon"

echo "[3/6] Updating asound.conf (adding tts_out and kws_in)..."
# Pull original, append AEC config, push back
$ADB shell "cat $LOWER_ASOUND" > "$TMPDIR/asound.conf"
if grep -q 'pcm.tts_out' "$TMPDIR/asound.conf"; then
    echo "  tts_out already present, skipping."
else
    cat >> "$TMPDIR/asound.conf" <<'ASOUND_EOF'

# --- AEC pipeline devices ---
# TTS output -> loopback (for AEC reference)
pcm.tts_out {
    type plug
    slave {
        pcm "hw:1,0,0"
        rate 16000
        format S16_LE
        channels 1
    }
}

# AEC cleaned output for pet_voice
pcm.kws_in {
    type plug
    slave {
        pcm "hw:1,1,1"
        rate 16000
        format S16_LE
        channels 1
    }
}
ASOUND_EOF
fi
# Push to staging area, then cp to merged + upper (overlayfs-safe)
$ADB push "$TMPDIR/asound.conf" /data/devtools/new_asound.conf

echo "[4/6] Applying asound.conf to merged + overlayfs upper..."
$ADB shell "cp /data/devtools/new_asound.conf $MERGED_ASOUND"
$ADB shell "mkdir -p /data/overlay_upper/etc"
$ADB shell "cp /data/devtools/new_asound.conf $UPPER_ASOUND"

echo "[5/6] Updating KWS config (plughw:0,0 -> plughw:1,1,1)..."
# Modify locally, push to staging, then cp
sed 's|plughw:0,0|plughw:1,1,1|g' "$TMPDIR/backup_kws_config.json" > "$TMPDIR/kws_config.json"
$ADB push "$TMPDIR/kws_config.json" /data/devtools/new_kws_config.json
$ADB shell "cp /data/devtools/new_kws_config.json $KWS_MERGED"
$ADB shell "mkdir -p $(dirname $KWS_UPPER)"
$ADB shell "cp /data/devtools/new_kws_config.json $KWS_UPPER"

echo "[6/6] Starting aec_daemon and restarting pet_voice..."
# Kill existing aec_daemon if any
$ADB shell "start-stop-daemon -K -x /data/devtools/aec_daemon 2>/dev/null; true"
# Start aec_daemon as daemon
$ADB shell "start-stop-daemon -S -b -m -p /tmp/aec_daemon.pid -x /data/devtools/aec_daemon"
sleep 1
$ADB shell "ps aux | grep aec_daemon | grep -v grep"

# Restart master service (which includes pet_voice)
$ADB shell "systemctl restart master"

echo ""
echo "Deploy complete. AEC pipeline is active."
echo "  TTS output  → tts_out (loopback 1,0,0)"
echo "  AEC daemon  → mic + ref → cleaned → loopback 1,0,1"
echo "  pet_voice   → plughw:1,1,1 (cleaned audio)"
echo ""
echo "To restore: ./scripts/restore_aec.sh"
