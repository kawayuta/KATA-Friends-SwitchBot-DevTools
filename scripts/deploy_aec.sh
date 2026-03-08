#!/bin/bash
# Deploy AEC pipeline (6ch) to device
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
LOWER_ASOUND="/app/etc/asound.conf"
SERVICE_DIR="/data/overlay_upper/etc/systemd/system"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "[1/5] Backing up original asound.conf (from lower layer)..."
$ADB shell "cat $LOWER_ASOUND" > "$TMPDIR/backup_asound.conf"
$ADB push "$TMPDIR/backup_asound.conf" /data/devtools/backup_asound.conf

echo "[2/5] Pushing aec_daemon binary..."
$ADB push "$BINARY" /data/devtools/aec_daemon
$ADB shell "chmod +x /data/devtools/aec_daemon"

echo "[3/5] Updating asound.conf (adding tts_out, redirecting record to AEC output)..."
$ADB shell "cat $LOWER_ASOUND" > "$TMPDIR/asound.conf"

# Remove existing AEC config if present (from previous deploy)
sed -i.bak '/^# --- AEC pipeline/,/^$/d' "$TMPDIR/asound.conf"

# Append AEC pipeline devices
cat >> "$TMPDIR/asound.conf" <<'ASOUND_EOF'

# --- AEC pipeline devices ---
# TTS output -> loopback (for AEC reference capture)
pcm.tts_out {
    type plug
    slave {
        pcm "hw:1,0,0"
        rate 16000
        format S16_LE
        channels 1
    }
}
ASOUND_EOF

# Override 'record' to point to AEC output loopback (6ch)
# media reads via default capture -> record -> now gets AEC'd audio
if grep -q '^pcm.record' "$TMPDIR/asound.conf"; then
    # Replace existing record definition
    # Use python for reliable multi-line replacement
    python3 -c "
import re, sys
with open('$TMPDIR/asound.conf', 'r') as f:
    content = f.read()
# Match pcm.record { ... } block (handles nested braces)
pattern = r'pcm\.record\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}'
replacement = '''pcm.record {
    type plug
    slave {
        pcm \"hw:1,1,1\"
        rate 16000
        format S16_LE
        channels 6
    }
}'''
content = re.sub(pattern, replacement, content)
with open('$TMPDIR/asound.conf', 'w') as f:
    f.write(content)
"
else
    cat >> "$TMPDIR/asound.conf" <<'RECORD_EOF'

# Override record to read AEC output (6ch loopback)
pcm.record {
    type plug
    slave {
        pcm "hw:1,1,1"
        rate 16000
        format S16_LE
        channels 6
    }
}
RECORD_EOF
fi

# Push to staging area, then cp to merged + upper (overlayfs-safe)
$ADB push "$TMPDIR/asound.conf" /data/devtools/new_asound.conf

echo "[4/5] Applying asound.conf to merged + overlayfs upper..."
$ADB shell "cp /data/devtools/new_asound.conf $MERGED_ASOUND"
$ADB shell "mkdir -p /data/overlay_upper/etc"
$ADB shell "cp /data/devtools/new_asound.conf $UPPER_ASOUND"

echo "[5/5] Updating aec-pipeline.service and restarting..."
# Update service with ExecStartPost sleep for media timing
$ADB shell "cat $SERVICE_DIR/aec-pipeline.service" > "$TMPDIR/aec-pipeline.service" 2>/dev/null || true
if [ -s "$TMPDIR/aec-pipeline.service" ]; then
    if ! grep -q 'ExecStartPost' "$TMPDIR/aec-pipeline.service"; then
        python3 -c "
import re
with open('$TMPDIR/aec-pipeline.service', 'r') as f:
    content = f.read()
content = re.sub(r'(ExecStart=.*)', r'\1\nExecStartPost=/bin/sleep 1', content, count=1)
with open('$TMPDIR/aec-pipeline.service', 'w') as f:
    f.write(content)
"
        $ADB push "$TMPDIR/aec-pipeline.service" /data/devtools/aec-pipeline.service
        $ADB shell "cp /data/devtools/aec-pipeline.service $SERVICE_DIR/aec-pipeline.service"
        $ADB shell "systemctl daemon-reload"
    fi
fi

# Restart aec-pipeline service (or start manually)
if $ADB shell "systemctl is-enabled aec-pipeline" 2>/dev/null | grep -q enabled; then
    $ADB shell "systemctl restart aec-pipeline"
else
    # Kill existing aec_daemon if any
    $ADB shell "start-stop-daemon -K -x /data/devtools/aec_daemon 2>/dev/null; true"
    # Start aec_daemon as daemon
    $ADB shell "start-stop-daemon -S -b -m -p /tmp/aec_daemon.pid -x /data/devtools/aec_daemon"
fi
sleep 1
$ADB shell "ps aux | grep aec_daemon | grep -v grep"

# --- Deploy libsv_lang.so (SenseVoice ASR language override) ---
SV_LANG_LIB="scripts/build-aloop/output/libsv_lang.so"
if [ -f "$SV_LANG_LIB" ]; then
    echo "[6/7] Deploying libsv_lang.so (ASR language override)..."
    $ADB push "$SV_LANG_LIB" /data/devtools/libsv_lang.so

    # Write default language config (ja) if not already present
    $ADB shell "[ -f /data/devtools/asr_language.conf ] || echo -n ja > /data/devtools/asr_language.conf"

    # Create pet_voice.service override for LD_PRELOAD
    $ADB shell "mkdir -p /data/overlay_upper/etc/systemd/system/pet_voice.service.d/"
    cat > "$TMPDIR/sv_lang_override.conf" <<'OVERRIDE_EOF'
[Service]
Environment=LD_PRELOAD=/data/devtools/libsv_lang.so
OVERRIDE_EOF
    $ADB push "$TMPDIR/sv_lang_override.conf" /data/devtools/sv_lang_override.conf
    $ADB shell "cp /data/devtools/sv_lang_override.conf /data/overlay_upper/etc/systemd/system/pet_voice.service.d/override.conf"
    $ADB shell "systemctl daemon-reload"
else
    echo "[6/7] libsv_lang.so not found, skipping ASR language override."
    echo "       Run scripts/build-aloop/build_sv_lang.sh to build it."
fi

# Restart media + pet_voice (separate services) to pick up new record PCM
echo "[7/7] Restarting media + pet_voice..."
$ADB shell "systemctl restart media"
$ADB shell "systemctl restart pet_voice"

echo ""
echo "Deploy complete. AEC pipeline (6ch) is active."
echo "  TTS output    → tts_out (loopback 1,0,0)"
echo "  AEC daemon    → cap_dsnoop (6ch) + ref → AEC ×4 → loopback 1,0,1 (6ch)"
echo "  record (media)→ hw:1,1,1 (6ch AEC output)"
if [ -f "$SV_LANG_LIB" ]; then
echo "  ASR language  → LD_PRELOAD override (libsv_lang.so)"
fi
echo ""
echo "To restore: ./scripts/restore_aec.sh"
