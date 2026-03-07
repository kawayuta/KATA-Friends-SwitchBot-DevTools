English | **[日本語](aec_pipeline_ja.md)**

# AEC Pipeline (Echo Cancellation + Barge-in)

Solves the problem of speaker echo leaking into the microphone during TTS playback, causing false/missed wake word detections (KWS).
Uses speexdsp AEC to cancel the echo on all 4 mic channels, **enabling wake word detection (barge-in) even during TTS playback.**

## Prerequisites

- `snd-aloop` kernel module (`snd-aloop.ko`) already built
- `libspeexdsp.so.1` installed on device
- Docker (for arm64 container builds; tested on Apple Silicon Mac)

## Architecture

```
TTS (mpg123 -a tts_out) → plug → hw:1,0,0 (loopback write)
                                      │
                                      ↓
                            hw:1,1,0 (loopback capture = TTS mono reference)
                                      │
                            ┌─────────┤ aec_daemon (6ch, speexdsp ×4)
                            │         │
                            ↓         ↓
                    softvol_ply     reference buf (ring buffer, mono)
                    → speaker         │
                                      │
    Real mic (cap_dsnoop hw:0,0) ──→ AEC process (4 mic channels)
    (6ch: mic0,ref,mic1,mic2,mic3,?)     │
                                    cleaned 6ch audio
                                      │
                                      ↓
                            hw:1,0,1 (loopback write, 6ch)
                                      │
                                      ↓
                            hw:1,1,1 (loopback capture, 6ch)
                                      │
                              record (plug) ← default capture
                                      │
                                      ↓
                              media (RK VQE + KWS) → pet_voice → ZMQ
```

### Signal Flow

1. **TTS audio**: mpg123 outputs to `tts_out` (ALSA plug → `hw:1,0,0`)
2. **Reference capture**: aec_daemon's reference thread captures TTS audio (mono) from `hw:1,1,0` (loopback), stores it in a ring buffer, and simultaneously plays it through `softvol_ply` (speaker)
3. **AEC processing**: Main thread reads 6ch from `cap_dsnoop`, applies `speex_echo_cancellation()` to each of the 4 mic channels (ch0, ch2, ch3, ch4) individually using the mono TTS reference. ch1 (hw reference) and ch5 are passed through unchanged
4. **Clean audio output**: AEC-processed 6ch audio is written to `hw:1,0,1` (loopback)
5. **media input**: `record` PCM (default capture) is redirected to `hw:1,1,1`, so media reads AEC'd 6ch audio. Media's built-in RK VQE provides additional beamforming and residual echo suppression
6. **KWS**: media → pet_voice → ZMQ as before

### Channel Mapping

| Channel | Content | AEC |
|---------|---------|-----|
| 0 | mic0 | Applied |
| 1 | speaker reference | Passthrough |
| 2 | mic1 | Applied |
| 3 | mic2 | Applied |
| 4 | mic3 | Applied |
| 5 | unknown | Passthrough |

## Quick Start

```bash
# 1. Build AEC daemon
./scripts/build-aloop/build_aec.sh

# 2. Deploy to device
./scripts/deploy_aec.sh

# 3. Restore to original state
./scripts/restore_aec.sh
```

## Build

```bash
./scripts/build-aloop/build_aec.sh
```

Compiles natively inside an arm64 Docker container (`ubuntu:22.04`).
Output: `scripts/build-aloop/output/aec_daemon` (ELF 64-bit aarch64)

### Build Files

| File | Description |
|------|-------------|
| `scripts/build-aloop/aec_daemon.c` | AEC daemon source (C, speexdsp, 6ch) |
| `scripts/build-aloop/build_aec.sh` | Build script |
| `scripts/build-aloop/Dockerfile` | Build environment (arm64 multiarch) |

### aec_daemon Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16kHz |
| Format | S16_LE |
| Mic input | 6ch (cap_dsnoop) |
| Output | 6ch (loopback hw:1,0,1) |
| Reference | mono (loopback hw:1,1,0) |
| Frame size | 160 samples (10ms) |
| Filter length | 2048 (128ms tail) |
| AEC instances | 4 (one per mic channel) |

## Deploy

```bash
./scripts/deploy_aec.sh [KATA_IP]
```

The deploy script:

1. Backs up original `asound.conf` (`/data/devtools/backup_asound.conf`)
2. Pushes `aec_daemon` binary to `/data/devtools/`
3. Adds `tts_out` device definition to `asound.conf`
4. Redirects `record` PCM to `hw:1,1,1` (6ch AEC output loopback)
5. Updates `aec-pipeline.service` with `ExecStartPost` sleep
6. Starts `aec_daemon` and restarts `media` + `pet_voice` services

### Changes on Device

**asound.conf** (`/etc/asound.conf`):
```
pcm.tts_out {
    type plug
    slave { pcm "hw:1,0,0"; rate 16000; format S16_LE; channels 1 }
}

pcm.record {
    type plug
    slave { pcm "hw:1,1,1"; rate 16000; format S16_LE; channels 6 }
}
```

**app_flask.py** (mpg123 output device, 3 locations):
```python
["mpg123", "-q", "-a", "tts_out", mp3_path]
```

## Restore

```bash
./scripts/restore_aec.sh [KATA_IP]
```

Restores `asound.conf` from backup, stops `aec_daemon`, and restarts `master`.

## systemd Service (Auto-start)

When `aec-pipeline.service` is enabled, after device reboot it automatically:
1. Loads `snd-aloop` kernel module
2. Starts `aec_daemon`
3. Waits 1 second (`ExecStartPost`) for ALSA devices to be ready
4. Completes before `master.service` (media + pet_voice)

```bash
# Check service status
adb shell "systemctl status aec-pipeline"

# Stop manually (also unloads module)
adb shell "systemctl stop aec-pipeline"

# Start manually
adb shell "systemctl start aec-pipeline"

# Disable auto-start
adb shell "systemctl disable aec-pipeline"

# Re-enable auto-start
adb shell "systemctl enable aec-pipeline"
```

Service file: `/data/overlay_upper/etc/systemd/system/aec-pipeline.service`

## Barge-in (TTS Interruption in Conversation Mode)

With the AEC pipeline active, in conversation mode (`/api/conversation/enable`), detecting a wake word during TTS playback will:

1. Immediately kill TTS playback (`mpg123` kill)
2. If a question was spoken with the wake word, immediately process it through LLM → new TTS response
3. If only wake word, return to listening state
4. Chained barge-ins are supported (can barge-in during the new TTS response)

### Related Functions (app_flask.py)

| Function | Description |
|----------|-------------|
| `_tts_play_with_bargein()` | Non-blocking TTS + ZMQ monitoring. Kills TTS on wake word |
| `_conv_respond_with_bargein()` | LLM → TTS → barge-in loop handler |

## Troubleshooting

```bash
# Check if aec_daemon is running
adb shell "ps aux | grep aec_daemon | grep -v grep"

# Check Loopback device
adb shell "aplay -l | grep Loopback"

# Check if snd-aloop module is loaded
adb shell "lsmod | grep snd_aloop"

# Verify 6ch output to loopback
adb shell "cat /proc/asound/card1/pcm0p/sub1/hw_params"

# Verify media reading from loopback
adb shell "cat /proc/asound/card1/pcm1c/sub1/hw_params"

# systemd service logs
adb shell "journalctl -u aec-pipeline --no-pager -n 30"

# Run aec_daemon manually for debugging
adb shell "/data/devtools/aec_daemon"
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| media can't start | `hw:1,1,1` not available (snd-aloop not loaded or aec_daemon not running) | `systemctl start aec-pipeline` |
| No TTS audio from speaker | aec_daemon is not running | `systemctl start aec-pipeline` |
| Wake word not detected | AEC daemon crashed | Check `journalctl -u aec-pipeline`, restart service |
| Build error (glibc mismatch) | Using x86 cross-compiler | `build_aec.sh` uses arm64 Docker container for native compilation |
| High latency | Loopback buffer too large | Adjust `FRAME_SAMPLES * 8` buffer size in aec_daemon |
