English | **[日本語](api_integration_ja.md)**

# Kata Friends Home API Integration

A system that detects SwitchBot Kata Friends events via BLE and forwards them to a home API server.
Local API authentication has been reverse-engineered, enabling access to photos and face recognition data.

## Device Info

| Item | Value |
|---|---|
| Device Name | KATA Friends |
| BLE Name | WoAIPE (WonderLabs AI Pet) |
| Manufacturer | Woan Technology (Shenzhen) |
| BLE Manufacturer ID | 2409 (SwitchBot/WonderLabs) |
| SwitchBot API deviceId | See .env |
| Wi-Fi IP | See .env |
| Wi-Fi | Connected to 2.4GHz band |
| OS | Linux 6.1.99 aarch64 (Android-based, hostname: WlabRobot) |
| QR Code (back panel) | Present (manufacturing serial number) |

## Findings

### 1. Wi-Fi Traffic

Almost no internet communication. Voice recognition and camera results are not sent to the cloud.

### 2. SwitchBot Official API (v1.1)

| Endpoint | Result |
|---|---|
| `GET /v1.1/devices` | Listed as "KATA Friends", but `deviceType` field is missing (present for other devices) |
| `GET /v1.1/devices/{id}/status` | `statusCode: 100` (success) but body is empty `{}` |
| Webhook | Registration succeeds but no events are sent from Kata Friends |

Auth: HMAC-SHA256 signature (implemented in `setup_webhook.py`)

### 3. Local API (Auth Solved)

Kata Friends runs an HTTP server on LAN. The port is dynamically assigned via MQTT.

| Item | Value |
|---|---|
| Endpoint | `POST /thing_model/func_request` |
| Health check | `POST /heartbeat` |
| Port | 27999 (dynamically distributed via MQTT; was previously 22090) |
| Auth | `auth: MD5(body + token)` |
| Token | Distributed via MQTT (stored in .env) |
| Status | **Working** |

#### Authentication Method

```
auth = MD5(request_body + token)
```

- The token is a UUID distributed by the device to SwitchBot cloud via MQTT
- Retrieved via ADB from device logs: `cc_mqtt.*.log`, `functionID:1021` messages
- For empty body (heartbeat): `auth = MD5(token)`

#### How Auth Was Reverse-Engineered

1. Captured iPhone app traffic via mitmproxy — confirmed auth header is MD5 format (32 hex chars)
2. Heartbeat requests (empty body) always produce the same auth — ruled out timestamp-based generation
3. Connected via ADB (port 5555 open) with root access — found token in device logs
4. Verified `auth = MD5(body + token)` matches all captured requests

#### Available Functions

| functionID | Function | Status |
|---|---|---|
| 9206 | Storage info | Verified (64MB total, 2MB used) |
| 9217 | Photo timeline (with face recognition) | Verified (176 photos retrieved) |
| 9225 | Face recognition data (registered/unregistered) | Verified (3 registered + 16 unregistered) |

Request format:
```json
{
  "version": "1",
  "code": 3,
  "deviceID": "<KATA_DEVICE_ID>",
  "payload": {
    "functionID": 9206,
    "requestID": "UUID",
    "timestamp": 1709500000000,
    "params": {}
  }
}
```

### 4. Device Internals (via ADB)

ADB debugging is always enabled on port 5555 with no authentication required. Root shell access is available directly from Mac — no SwitchBot app or proxy needed. Just requires the device to be on the same Wi-Fi network.

```bash
# Install adb if needed
brew install android-platform-tools

# Connect (no auth, root access)
adb connect <KATA_IP>:5555
adb shell    # Opens root shell directly
```

| Item | Value |
|---|---|
| OS | Linux 6.1.99 aarch64 (Debian-based) |
| Hostname | WlabRobot |
| User | root |
| Python | 3.12.3 |
| Web Framework | Flask (Werkzeug) |
| LLM Runtime | RKLLM (Rockchip NPU) |
| Chip | RK3576 (Rockchip) |

Open Ports:

| Port | Purpose |
|---|---|
| 5555 | ADB (Android Debug Bridge) |
| 5558 | ZMQ XPUB (internal IPC, subscriber port) |
| 5559 | ZMQ XSUB (internal IPC, publisher port, sensor data) |
| 8080 | LLM action server (Flask/RKLLM) |
| 8082 | LLM diary server (Flask/RKLLM) |
| 8083 | LLM unified router (auto-routes to 8080/8082) |
| 27999 | Local API (thing_model, auth required) |
| 50001 | control_center_runner (unknown purpose) |

Key directories:
```
/app/opt/wlab/sweepbot/bin/     # Main application
  flask_server_action.py        # LLM action server (port 8080)
  flask_server_diary.py         # Diary server
  route.py                      # Routing
/data/cache/log/                # Logs
  cc_main.*.log                 # Main process logs (auth verification, etc.)
  cc_mqtt.*.log                 # MQTT logs (token distribution, etc.)
  cc_bt.*.log                   # Bluetooth logs
/data/common/resource/          # Resources (eye animations, etc.)
```

### 5. Internal IPC — ZeroMQ (Discovered)

The device uses ZeroMQ for inter-process communication. The `master` process runs an XPUB/XSUB proxy.

| Socket | Address | Role |
|---|---|---|
| XPUB | `tcp://127.0.0.1:5558` | Subscribers connect here |
| XSUB | `tcp://127.0.0.1:5559` | Publishers connect here |

Messages are ZMQ multipart: Frame 1 = `#` + topic name, Frame 2 = MessagePack-encoded JSON payload.

**Sensor topics** (flowing on port 5559): `/imu`, `/tf`, `/odom`, `/curr_limb_pose`

**Control topics** (from binary analysis): `/ai/do_action`, `/ai/mood`, `/ai/sound`, `/ai/show_eyes`, `/agent/start_cc_task`, `/agent/stop_cc_task`

Publishing to these topics via XSUB port could allow direct action triggering without the LLM server or local API.

### 6. BLE Advertisement (Working)

State changes can be passively detected. The current system uses this method.

```
xxxxxxxxxxxx | 4c | 01 | 2132 | 0010 | 39 | 00
[  MAC  6B ] |seq | ?? | fixed | fixed |b12 |b13
```

| Byte | Content | Notes |
|---|---|---|
| 0-5 | BLE MAC address | Fixed |
| 6 | Sequence number | Increments per request |
| 7 | Unknown | Always 01 |
| 8-9 | Unknown | Always 2132 |
| 10-11 | Unknown | Always 0010 |
| 12 | Action counter | Decrements on action execution |
| 13 | Interaction flag | 00=idle, 03=voice responding |

### 7. BLE GATT

Has SwitchBot standard service `cba20d00-224d-11e6-9fb8-0002a5d5c51b`. Write characteristic (cba20002) and Notify characteristic (cba20003) available. No push notifications — request-response only.

Only `0x57` prefix responds (`0x56`, `0x58`, `0x01`, etc. yield nothing). Full 0x5700-0x57FF brute-force completed with `ble_brute.py`.

| Command | Response | Interpretation |
|---|---|---|
| 0x5700 | `01` | OK response only |
| 0x5701 | `05` | Not supported |
| 0x5702 | `01 64 19` | 01=OK, 64=battery 100%?, 19=state value |
| 0x5703 | `05` | Not supported |
| 0x5704 | `01 02` | Unknown (2-byte response) |
| 0x5705-0x57FF | No response | — |

## How to Trigger Actions

### Action Execution Flow

```mermaid
flowchart LR
    A["🎤 Voice / curl"] --> B["LLM :8080\nQwen3+LoRA"]
    B --> C["mood/instruction"]
    C --> D[ZMQ\n/ai/do_action]
    D --> E[master\nXPUB/XSUB]
    E --> F[control_center]
    F --> G["🤖 Execute"]
```

### Via LLM Server (No Auth Required)

```bash
# Send natural language → LLM returns mood/instruction
curl -X POST http://<KATA_IP>:8080/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "dance please"}'
# Response: happy/dance

# Japanese
curl -X POST http://<KATA_IP>:8080/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "踊って"}'
```

This returns the LLM's *decision* only. The actual action is executed by internal processes via ZMQ.

Ports 8080 (direct), 8082 (diary), 8083 (router) are all **unauthenticated**.

### Via ZMQ (Direct, WIP)

```bash
adb shell
# Publish to /ai/do_action via ZMQ XSUB (tcp://127.0.0.1:5559)
# Format: multipart [#/ai/do_action, msgpack(json_payload)]
```

### Available Actions (41)

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

### Available Emotions (7)

`happy`, `angry`, `sad`, `scared`, `disgusted`, `surprised`, `neutral`

## Current Detection Capabilities

### What Can Be Detected

- **Interaction start**: byte[13] changes 00→03 (voice command detected)
- **Interaction end**: byte[13] changes 03→00
- **Action execution**: byte[12] decrements (dance, photo, etc.)
- **Photo list**: Via local API (auth solved)
- **Face recognition data**: Registered/unregistered face list (via local API)
- **Storage info**: Usage and capacity (via local API)
- **Device internals**: Filesystem and log access via ADB

### What Cannot Be Detected

- **Voice recognition text**: Processed entirely on-device, not sent externally
- **Command type**: Unknown what was instructed (only whether it reacted)

## Architecture

```mermaid
flowchart TB
    subgraph Mac["Mac (Control)"]
        BLE[ble_watcher.py\nBLE monitor]
        API[home_api\nFastAPI :9000]
        CLI[kata_local_api.py]
        CURL[curl / scripts]
    end

    subgraph Device["Kata Friends"]
        subgraph Internal["Internal Services"]
            VOICE[pet_voice\nASR]
            LLM[LLM :8080\nAction]
            DIARY[LLM :8082\nDiary]
            ZMQ[ZMQ IPC\n:5558/:5559]
            CC[control_center\n:27999]
            MASTER[master\nprocess]
        end
        MOTOR["🤖 Motors/Eyes/Sound"]
    end

    BLE -.->|"BLE scan\nbyte[12],[13]"| Device
    BLE -->|"POST /events"| API
    CLI -->|"POST + MD5 auth"| CC
    CURL -->|"POST (no auth)"| LLM
    CURL -->|"adb shell"| Device

    VOICE --> LLM
    LLM --> ZMQ
    ZMQ --> MASTER
    MASTER --> MOTOR
    CC --> ZMQ
```

See **[Device Internals Documentation](../README.md)** for full details.

## Directory Structure

```
kata/
├── .env                      # Secrets (tokens, MACs, etc.)
├── .env.example              # .env template
├── .gitignore
├── README.md                 # Japanese documentation
├── README_en.md              # This file
├── requirements.txt          # Python dependencies
├── ble_watcher.py            # BLE advertisement monitor → API event sender
├── home_api/
│   └── main.py               # FastAPI event receiver
├── proxy/
│   ├── kata_proxy.py         # mitmproxy transparent proxy (unused: no Wi-Fi traffic)
│   └── capture_auth.py       # mitmweb auth analysis addon
├── scripts/
│   ├── 01_discover.sh        # Network device discovery
│   ├── 02_capture.sh         # tcpdump packet capture
│   ├── 03_analyze_pcap.sh    # pcap analysis
│   ├── 04_setup_proxy.sh     # mitmproxy setup
│   ├── 05_setup_routing.sh   # macOS routing setup
│   ├── 06_teardown_routing.sh # Routing teardown
│   ├── setup_webhook.py      # SwitchBot official API webhook management
│   ├── kata_local_api.py     # Local API client (auth solved, working)
│   ├── ble_monitor.py        # BLE advertisement monitor (debug)
│   ├── ble_gatt_explore.py   # GATT service explorer
│   ├── ble_command.py        # BLE command sender/tester
│   └── ble_brute.py          # BLE GATT command brute-force scanner
├── logs/                     # Event logs (auto-generated)
└── captures/                 # pcap files (auto-generated)
```

## Getting Started

### Terminal 1: API Server
```bash
pip3 install -r requirements.txt
python3 -m uvicorn home_api.main:app --host 0.0.0.0 --port 9000
```

### Terminal 2: BLE Monitor
```bash
pip3 install bleak python-dotenv
python3 ble_watcher.py
```

Talk to Kata Friends and events will be detected and sent to the API.

### Local API Usage
```bash
python3 scripts/kata_local_api.py storage    # Storage info
python3 scripts/kata_local_api.py photos     # Photo list
python3 scripts/kata_local_api.py faces      # Face recognition data
python3 scripts/kata_local_api.py discover   # Scan functionIDs
python3 scripts/kata_local_api.py raw <ID>   # Arbitrary functionID
```

### ADB Access
```bash
adb connect <KATA_IP>:5555    # Root shell access
adb shell                      # Explore device internals
```

## Dependencies

```bash
pip3 install -r requirements.txt
pip3 install bleak python-dotenv
```

## Scripts

### Main System

#### ble_watcher.py

Monitors BLE advertisements, detects Kata Friends state changes, and sends events to the API server.

```bash
python3 ble_watcher.py
```

Events sent:
- `interaction_start` — Voice command detected (byte[13]: 00→03)
- `interaction_end` — Response finished (byte[13]: 03→00)
- `action` — Action executed (byte[12] decrements)

#### home_api/main.py

FastAPI server that receives events from the BLE monitor. Events are logged to `logs/kata_events.jsonl`.

```bash
python3 -m uvicorn home_api.main:app --host 0.0.0.0 --port 9000
```

Endpoints:
- `POST /events` — Receive events
- `GET /health` — Health check

#### scripts/kata_local_api.py

Local API client for Kata Friends. Retrieves photos, face recognition data, and storage info.

Auth method: `auth = MD5(body + token)` (uses `KATA_LOCAL_TOKEN` from `.env`)

```bash
python3 scripts/kata_local_api.py storage    # Storage info
python3 scripts/kata_local_api.py photos     # Photo list (with thumbnail URLs)
python3 scripts/kata_local_api.py faces      # Face recognition data
python3 scripts/kata_local_api.py discover   # Scan functionIDs
python3 scripts/kata_local_api.py raw <ID>   # Arbitrary functionID
```

### Investigation & Debug Scripts

#### scripts/ble_monitor.py

Debug tool that displays raw BLE advertisement data. Highlights changed bytes between updates.

```bash
python3 scripts/ble_monitor.py
```

#### scripts/ble_gatt_explore.py

Enumerates BLE GATT services and characteristics. Reads available values and subscribes to notifications.

```bash
python3 scripts/ble_gatt_explore.py
```

#### scripts/ble_command.py

Sends commands to BLE GATT characteristics and checks responses. Tests SwitchBot standard commands (0x5701-0x5721) and various prefixes.

```bash
python3 scripts/ble_command.py
```

#### scripts/ble_brute.py

Brute-force scanner for BLE GATT commands. Sweeps the second byte (0x00-0xFF) for a given prefix and logs all responses. Results are saved to `logs/`.

```bash
python3 scripts/ble_brute.py              # 0x5700-0x57FF (default)
python3 scripts/ble_brute.py --prefix 01  # 0x0100-0x01FF
```

#### proxy/capture_auth.py

mitmweb addon script for capturing traffic between iPhone SwitchBot app and Kata Friends, with auth header verification.

```bash
mitmweb -s proxy/capture_auth.py -p 8888 --set connection_strategy=lazy
```

#### scripts/setup_webhook.py

SwitchBot official API (v1.1) webhook management. Requires `SWITCHBOT_TOKEN` and `SWITCHBOT_SECRET` in `.env`.

```bash
python3 scripts/setup_webhook.py setup <webhook_url>  # Register webhook
python3 scripts/setup_webhook.py query                 # Check status
python3 scripts/setup_webhook.py delete                # Delete webhook
```

### Network Investigation Scripts

```bash
# 1. Discover devices on network (ARP table)
bash scripts/01_discover.sh

# 2. Capture Kata Friends packets (sudo required)
bash scripts/02_capture.sh <KATA_IP>

# 3. Analyze pcap file (destination IPs, SNI, HTTP requests)
bash scripts/03_analyze_pcap.sh <pcap_file>

# 4. Install mitmproxy and setup CA certificate
bash scripts/04_setup_proxy.sh

# 5. macOS packet forwarding setup (redirect HTTPS to mitmproxy, sudo required)
bash scripts/05_setup_routing.sh

# 6. Tear down packet forwarding
bash scripts/06_teardown_routing.sh
```

## Next Steps

| Approach | Description | Difficulty |
|---|---|---|
| ZMQ action triggering | Publish to `/ai/do_action` via ZMQ to directly trigger actions (dance, photo, etc.) | Medium |
| ZMQ message format RE | Decode the MessagePack payloads for control topics to understand required fields | Medium |
| Mac microphone | Use BLE detection as trigger to record with Mac's mic → speech recognition via Whisper etc. | Low |
| Local API functionID scan | Use `kata_local_api.py discover` to find unknown functionIDs | Low |
| Port 50001 investigation | Determine the purpose of the externally-accessible port 50001 | Low |
| LLM server integration | Send commands directly to the RKLLM server on port 8080 | Medium |
| Firmware analysis | Detailed analysis of device binaries | High |
