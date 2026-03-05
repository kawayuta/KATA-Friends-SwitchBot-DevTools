English | **[日本語](README_ja.md)**

# Kata Friends Device Internals

Documentation of the device's internal structure discovered via ADB.

## How to Connect

```bash
# Install adb (first time only)
brew install android-platform-tools

# Connect (no auth required, root access)
adb connect <KATA_IP>:5555

# Open shell
adb shell
```

Available anytime the device is on Wi-Fi. No SwitchBot app required.

## Hardware

| Item | Value |
|---|---|
| CPU | ARM Cortex-A53 x4 (ARMv8-A) |
| Chip | Rockchip RK3576 |
| NPU | RKNN (Rockchip Neural Network) |
| RAM | 7.7GB |
| Storage | 28GB (/data) + SD card slot (/media/mmcblk1p1) |
| OS | Linux 6.1.99 aarch64 (Debian-based) |
| Hostname | WlabRobot |
| Python | 3.12.3 |

## Filesystem Overview

```
/
├── app/          196MB  Application (tmpfs overlay)
├── data/         8.5GB  User data, cache, AI models
├── rom/          1.5GB  Read-only filesystem
├── usr/          1.3GB  System binaries
├── media/        517MB  SD card
├── opt/          195MB  Additional packages
└── overlay/      229MB  Overlay FS
```

## Application Structure

### Main App: `/app/opt/wlab/sweepbot/`

```
sweepbot/
├── bin/              # Executables (69 files)
│   ├── master        # Main process (395KB)
│   ├── media         # Media handling (1.2MB)
│   ├── pet_voice     # Voice processing (985KB)
│   ├── recorder      # Recording service (591KB)
│   ├── rknn_server   # Neural network inference (455KB)
│   ├── uart_ota      # OTA updates
│   │
│   │   # Python/Flask servers
│   ├── flask_server_action.py  # LLM action server (port 8080)
│   ├── flask_server_diary.py   # LLM diary server (port 8082)
│   ├── route.py                # Unified router (port 8083)
│   │
│   │   # Shell scripts (35 files)
│   ├── rknn_server.sh
│   ├── llm_action_server.sh
│   └── ...
│
├── config/           # Per-model device configs
│   ├── K20/ K20Pro/ S1/ S1+/ S10/ S20/ S20mini/ A01/
│   └── *.lua         # SLAM configs
│
├── lib/              # Shared libraries
│   ├── libonnxruntime.so   # ML inference (13MB)
│   ├── libmosquitto.so     # MQTT client
│   ├── librkllmrt.so       # RKLLM inference runtime
│   └── ai_brain/ bt_bridge/ control_center/ lds_slam/
│
└── share/            # Resources & model configs
    └── llm_server/res/
        ├── action_system_prompt.txt
        ├── system_prompt_diary.txt
        └── system_prompt_diary_translation.txt
```

## AI Models

### LLM (Large Language Models)

Stored in `/data/ai_brain/`.

| Model | Path | Size | Purpose |
|---|---|---|---|
| Qwen3-1.7B | `Qwen3-1.7B_w8a8_RK3576_v3.rkllm` | 2.2GB | Diary generation |
| Action Model (Qwen3 LoRA SFT) | `qwen3_v7.0.2_lora_sft_nothink_*.rkllm` | 900MB | Action classification |
| Action Model v1.1 | `actionmodel_w8a8_RK3576_v1.1.rkllm` | 900MB | Legacy action model |

Symlinks:
- `actionmodel.rkllm` → latest action model
- `diarymodel.rkllm` → Qwen3-1.7B

### Voice Recognition Models

Stored in `/data/ai_brain/voice/`.

| Model | File | Purpose |
|---|---|---|
| VAD | `vad/silero_vad.onnx` | Voice Activity Detection |
| KWS | `kws/encoder.onnx`, `decoder.onnx`, `joiner.onnx` | Keyword Spotting (wake words) |
| SenseVoice | `sensevoice/model.rknn` | Speech Recognition (ASR) |

Wake words defined in `kws/keywords.txt`.

### Face Recognition

Binary feature vectors stored in `/data/ai_brain_data/face_metadata/`.

## Data Storage

### `/data/` Directory (8.5GB)

```
data/
├── ai_brain/              # AI models (5GB+)
│   ├── *.rkllm            # LLM models
│   ├── voice/             # Voice models (VAD, KWS, SenseVoice)
│   └── model_version.json # Model version management
│
├── ai_brain_data/         # AI runtime data (19MB)
│   └── face_metadata/
│       ├── known/         # Registered faces (ID_*/)
│       │   └── ID_xxx/
│       │       ├── enrolled_faces/   # Registration photos (.jpg)
│       │       ├── features/         # Face feature vectors (.bin, 2KB each)
│       │       └── recognized_faces/ # Recognized photos (.jpg)
│       └── unknown/       # Unregistered faces
│
├── control_center/        # Main control data
│   ├── db/sqlite.db       # SQLite database
│   ├── maps/              # Navigation maps
│   ├── ai_images/         # AI-generated images
│   └── task/              # Task management
│
├── cache/                 # Cache (835MB)
│   ├── log/               # Log files (40+)
│   ├── image_recorder_archive/  # Captured photos
│   ├── video_recorder_archive/  # Recorded videos
│   └── vad/               # VAD cache
│
├── common/                # Shared resources (2.1GB)
│   └── resource/
│       ├── pink/          # Default theme
│       │   ├── actions/   # Action files (169, .act)
│       │   └── eyes/      # Eye animations (PNG, L/R)
│       ├── blue/          # Blue theme
│       ├── black/         # Black theme
│       ├── limbs/         # Limb data
│       ├── wheels/        # Wheel data
│       └── sounds/        # Sound effects
│
├── map_server/            # SLAM navigation
└── slam/                  # SLAM debug data
```

## Internal Services

### systemd Services (28)

| Service | Function |
|---|---|
| `master.service` | Main process controller |
| `ai_brain.service` | AI brain (recognition/decision) |
| `rknn_server.service` | Neural network inference |
| `llm_action.service` | LLM action classification (port 8080) |
| `llm_diary.service` | LLM diary generation (port 8082) |
| `llm_route.service` | LLM router (port 8083) |
| `media.service` | Media processing |
| `pet_voice.service` | Voice processing |
| `petbot_eye.service` | Eye animations |
| `recorder.service` | Recording |
| `slam.service` | SLAM (mapping/localization) |
| `bt_bridge.service` | Bluetooth |
| `network_monitor.service` | Network monitoring |
| `upload_image/video/audio.service` | Cloud upload |
| `update-robotic.service` | OTA updates |

### Internal HTTP Servers

| Port | Service | Description |
|---|---|---|
| 5555 | adbd | ADB daemon |
| 5558 | master (ZMQ XPUB) | ZMQ subscriber port (internal IPC) |
| 5559 | master (ZMQ XSUB) | ZMQ publisher port (internal IPC, sensor data flows here) |
| 8080 | flask_server_action.py | LLM action: takes voice text, returns `mood/instruction` |
| 8082 | flask_server_diary.py | LLM diary: generates diary from event list |
| 8083 | route.py | Unified router: auto-detects and routes to 8080/8082 |
| 27999 | control_center_runner (C++) | Local API: photos, faces, storage (auth required) |
| 50001 | control_center_runner (C++) | Unknown (externally accessible) |

### Internal IPC (ZeroMQ)

The device uses ZeroMQ (ZMQ) for inter-process communication with a classic XPUB/XSUB proxy pattern. The `master` process acts as the proxy.

#### Sockets

| Socket | Address | Role |
|---|---|---|
| XPUB | `tcp://127.0.0.1:5558` | Subscribers connect here |
| XSUB | `tcp://127.0.0.1:5559` | Publishers connect here |
| XPUB (IPC) | `ipc:///dev/shm/ipc.xpub` | IPC subscriber socket |
| XSUB (IPC) | `ipc:///dev/shm/ipc.xsub` | IPC publisher socket |

Note: IPC socket files may not always exist. TCP ports are more reliable.

#### Message Format

ZMQ multipart messages with 2 frames:

1. **Frame 1 (Topic)**: `#` prefix + topic name (e.g., `#/imu`, `#/agent/start_cc_task`)
2. **Frame 2 (Payload)**: MessagePack-encoded data (msgpack str8 `\xd9` + length + JSON payload)

#### Observed Topics (port 5559)

| Topic | Content | Data |
|---|---|---|
| `/imu` | IMU sensor data | Accelerometer, gyroscope (imu_link frame) |
| `/tf` | Transform tree | odom → base_footprint transforms |
| `/odom` | Odometry | Position, velocity, covariance matrix |
| `/curr_limb_pose` | Limb positions | Current servo/limb state |

#### Control Topics (from binary analysis)

| Topic | Purpose |
|---|---|
| `/ai/do_action` | Trigger an action (dance, photo, etc.) |
| `/ai/mood` | Set mood/emotion |
| `/ai/sound` | Play sound |
| `/ai/show_eyes` | Change eye animation |
| `/agent/start_cc_task` | Start a control center task |
| `/agent/stop_cc_task` | Stop a control center task |

#### Publishing Messages

```bash
# On-device via ADB (pyzmq not installed, use ctypes or push a script)
adb shell

# Monitor topics (connect to XPUB port 5558 as SUB)
# Publish commands (connect to XSUB port 5559 as PUB)
```

Example message for `/agent/start_cc_task`:
```
Frame 1: b'#/agent/start_cc_task'
Frame 2: msgpack(json_payload_string)
```

## How to Trigger Actions

### Flow: Voice Command → Action Execution

```mermaid
flowchart LR
    A["🎤 Voice Input"] --> B[pet_voice\nSenseVoice ASR]
    B --> C[Text]
    C --> D["LLM Action Server\n:8080 Qwen3+LoRA"]
    D --> E["mood/instruction\ne.g. happy/dance"]
    E --> F[ZMQ Publish\n/ai/do_action]
    F --> G[master process\nXPUB/XSUB proxy]
    G --> H[control_center\naction executor]
    H --> I["🤖 Motor/Eyes/Sound"]
```

### Flow: External Control (from Mac)

```mermaid
flowchart TB
    subgraph Mac
        M1["curl / script"]
        M2[ble_watcher.py]
        M3[kata_local_api.py]
    end

    subgraph Device["Kata Friends (192.168.x.x)"]
        subgraph LLM["LLM Servers (no auth)"]
            L1[":8080 Action"]
            L2[":8082 Diary"]
            L3[":8083 Router"]
        end
        subgraph ZMQ["ZMQ IPC (localhost only)"]
            Z1[":5558 XPUB"]
            Z2[":5559 XSUB"]
        end
        subgraph API["Local API (auth required)"]
            A1[":27999 thing_model"]
        end
        ADB[":5555 ADB\nroot shell"]
    end

    M1 -->|"POST /rkllm_action\n(no auth)"| L1
    M1 -->|"POST /rkllm_diary\n(no auth)"| L2
    M3 -->|"POST + MD5 auth"| A1
    M2 -.->|"BLE passive scan"| Device
    M1 -->|"adb shell"| ADB
    ADB -->|"ZMQ client"| Z2

    L3 --> L1
    L3 --> L2
```

### Flow: Diary Generation

```mermaid
flowchart LR
    A[Daily events log] --> B["POST :8082/rkllm_diary\ntask: diary"]
    B --> C["Qwen3-1.7B\non NPU"]
    C --> D["Title / Body / Emotion\n(Pixar-style narrative)"]
    D --> E{Language?}
    E -->|Chinese| F[Done]
    E -->|Other| G[Translation\nprompt]
    G --> F
```

There are 3 ways to make Kata Friends perform actions:

### Method 1: LLM Action Server (Easiest)

Send natural language text and the on-device LLM decides the action. **No auth required.**

```bash
# Basic example
curl -X POST http://<KATA_IP>:8080/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "dance please"}'
# Response: happy/dance

# Via the unified router (same result)
curl -X POST http://<KATA_IP>:8083/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "take a photo"}'
# Response: happy/take_photo

# Japanese works too
curl -X POST http://<KATA_IP>:8080/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "踊って"}'
```

Response format: `mood/instruction` (e.g., `happy/dance`, `neutral/no_action`)

**Important**: This only returns the *decision* — it does NOT execute the action. The actual execution is handled by other internal processes that subscribe to ZMQ topics.

### Method 2: ZMQ IPC (Direct Control, WIP)

Publish directly to internal ZMQ topics from within the device via ADB. This bypasses the LLM entirely.

```bash
# Connect to device
adb shell

# Publish to /ai/do_action topic (requires ZMQ client)
# Message format: multipart [#/ai/do_action, msgpack(payload)]
```

Status: Message format partially decoded. Payload structure for control topics still under investigation.

### Method 3: Local API (Auth Required)

For data retrieval (photos, faces, storage). Not for triggering actions directly.

```bash
python3 scripts/kata_local_api.py photos    # Photo list
python3 scripts/kata_local_api.py faces     # Face recognition data
python3 scripts/kata_local_api.py storage   # Storage info
```

## LLM Action Server

### Overview

Receives voice recognition text and returns the AI pet's reaction (emotion + action). Runs on the Rockchip NPU using RKLLM (quantized Qwen3 with LoRA SFT).

### Endpoint

```
POST http://<KATA_IP>:8080/rkllm_action
Content-Type: application/json

{"voiceText": "dance please"}
```

Response: `happy/dance`

### Available Actions (41 total)

| Category | Actions |
|---|---|
| Movement | `move_forward`, `move_back`, `move_left`, `move_right`, `spin`, `turn_left`, `turn_right`, `come_over`, `go_away`, `follow_me`, `stop` |
| Navigation | `go_to_kitchen`, `go_to_bedroom`, `go_to_balcony` |
| Expression | `dance`, `sing`, `nod`, `shake_head`, `wave_hand` |
| Looking | `look_left`, `look_right`, `look_up`, `look_down` |
| Greeting | `good_morning`, `bye`, `good_night`, `say_hello`, `welcome` |
| Emotion | `show_love`, `get_praise` |
| Function | `take_photo`, `go_power`, `go_play`, `go_sleep`, `wake_up` |
| Audio | `volume_up`, `volume_down`, `be_silent`, `speak` |
| Other | `user_leave`, `no_action` |

### Available Emotions (7 total)

`happy`, `angry`, `sad`, `scared`, `disgusted`, `surprised`, `neutral`

### Decision Rules

| Input | Result |
|---|---|
| Wake word only (hello, niko, noa, kata) | `neutral/no_action` |
| Background noise / filler words | `neutral/no_action` |
| Wake word + clear command | Ignore wake word, execute command |
| Compliment (appearance) | `happy/show_love` |
| Compliment (behavior) | `happy/get_praise` |
| Scolding | `angry/no_action` or `sad/stop` |

## LLM Diary Server

### Overview

Generates a first-person diary entry from the day's interaction events, written in Pixar-style warm narrative.

### Endpoint

```
POST http://<KATA_IP>:8082/rkllm_diary
Content-Type: application/json

{
  "task": "diary",
  "prompt": "language:Chinese\nlocal_date:2026-03-05\nevents:\n08:00 - Woke up\n19:15 - Got ear scratches"
}
```

Response: `Title/Diary content/Emotion`

### Diary Emotions

`Happy`, `Excited`, `Relaxed`, `Curious`, `Loved`, `Sleepy`, `Sad`, `Scared`, `Angry`, `Lonely`

## Face Recognition Data

### Directory Structure

```
/data/ai_brain_data/face_metadata/
├── known/                     # Registered
│   └── ID_<timestamp>/
│       ├── enrolled_faces/   # Registration photos (.jpg)
│       ├── features/         # Face feature vectors (.bin, 2056B each)
│       └── recognized_faces/ # Recognition history (.jpg)
└── unknown/                   # Unregistered
    └── <timestamp>/
        ├── enrolled_faces/
        └── features/
```

### Access Methods

```bash
# List registered faces
adb shell "ls /data/ai_brain_data/face_metadata/known/"

# Download face photos to Mac
adb pull /data/ai_brain_data/face_metadata/known/ID_xxx/enrolled_faces/

# Also available via local API
python3 scripts/kata_local_api.py faces
```

## Photos & Videos

```bash
# Photo list via local API
python3 scripts/kata_local_api.py photos

# Download photos via ADB
adb pull /data/cache/image_recorder_archive/ ./kata_photos/

# Download videos via ADB
adb pull /data/cache/video_recorder_archive/ ./kata_videos/
```

## Logs

### Log Files

40+ log files in `/data/cache/log/`.

| Log | Content |
|---|---|
| `cc_main.*.log` | Main process (auth, events) |
| `cc_mqtt.*.log` | MQTT (tokens, properties) |
| `cc_bt.*.log` | Bluetooth (BLE advertisements) |
| `rkllm_action_server.log` | LLM action inference |
| `rkllm_server.log` | LLM router |
| `wpa_supplicant.log` | WiFi connection |

### Real-time Log Monitoring

```bash
adb shell "tail -f /data/cache/log/cc_main.*.log"      # Main process
adb shell "tail -f /data/cache/log/cc_mqtt.*.log"       # MQTT
adb shell "tail -f /data/cache/log/rkllm_action_server.log"  # LLM
```

## Resource Files

### Themes

3 color themes in `/data/common/resource/`: pink (default), blue, black.

Each contains:
- `actions/` — 169 action files (`.act` format)
- `eyes/` — Eye animation frames (PNG, separate L/R)

### Actions (169 files)

| Prefix | Meaning | Example |
|---|---|---|
| `RDANCE` | Dance | `RDANCE008.act` |
| `RKATA` | Kata-specific | `RKATA1.act` ~ `RKATA6.act` |
| `RSING` | Sing | `RSING001.act` |
| `RSLEEP` | Sleep | `RSLEEP000.act` |
| `RMAP` | Map navigation | `RMAPGO.act` |
| `RPIC` | Take photo | `RPIC001.act` |
| `RW*` | Walking | `RWF001.act`, `RWL.act`, `RWR.act` |

## SQLite Database

`/data/control_center/db/sqlite.db`

No sqlite3 on device — download to Mac to inspect:

```bash
adb pull /data/control_center/db/sqlite.db ./
sqlite3 sqlite.db ".tables"
sqlite3 sqlite.db ".schema"
```

## Quick Reference: How to Access Data

| Data | Method |
|---|---|
| Photo list | `python3 scripts/kata_local_api.py photos` |
| Face recognition | `python3 scripts/kata_local_api.py faces` |
| Storage info | `python3 scripts/kata_local_api.py storage` |
| Photo files | `adb pull /data/cache/image_recorder_archive/` |
| Face photos | `adb pull /data/ai_brain_data/face_metadata/` |
| Video files | `adb pull /data/cache/video_recorder_archive/` |
| Logs | `adb shell "cat /data/cache/log/cc_main.*.log"` |
| LLM models | `adb pull /data/ai_brain/actionmodel.rkllm` |
| Action files | `adb pull /data/common/resource/pink/actions/` |
| Eye animations | `adb pull /data/common/resource/pink/eyes/` |
| SQLite DB | `adb pull /data/control_center/db/sqlite.db` |
| System prompts | `adb shell "cat /app/opt/wlab/sweepbot/share/llm_server/res/*.txt"` |
