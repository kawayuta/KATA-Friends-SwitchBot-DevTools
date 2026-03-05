"""
Kata Friends Developer Tools — Flask on-device backend

Device dependencies only: Flask 3.0.2, requests, jinja2
Runs directly on device at /data/devtools/, port 9001.
"""

import glob
import hashlib
import json
import os
import re
import subprocess
import time
import uuid

import requests
from flask import Flask, abort, jsonify, request, send_from_directory

# --- Config ---
# On-device: everything is localhost
LOCAL_API_PORT = int(os.environ.get("KATA_LOCAL_PORT", "27999"))
DEVICE_ID = os.environ.get("KATA_DEVICE_ID", "")
LOCAL_TOKEN = os.environ.get("KATA_LOCAL_TOKEN", "")

app = Flask(__name__, static_folder="static", static_url_path="/static")


# --- Auto-discover device ID and token from MQTT logs ---

def _read_token_from_logs():
    """Read local token from MQTT log files (functionID:1021 response)."""
    log_pattern = "/data/cache/log/cc_mqtt.*.log"
    files = sorted(glob.glob(log_pattern), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            with open(f, "r", errors="ignore") as fh:
                content = fh.read()
            # Look for token in functionID:1021 responses
            for m in re.finditer(r'"token"\s*:\s*"([0-9a-f-]{36})"', content):
                return m.group(1)
        except Exception:
            continue
    return ""


def _read_device_id_from_logs():
    """Read device ID from MQTT log files."""
    log_pattern = "/data/cache/log/cc_mqtt.*.log"
    files = sorted(glob.glob(log_pattern), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            with open(f, "r", errors="ignore") as fh:
                content = fh.read()
            for m in re.finditer(r'"deviceID"\s*:\s*"([A-F0-9]+)"', content):
                return m.group(1)
        except Exception:
            continue
    return ""


def _ensure_config():
    global DEVICE_ID, LOCAL_TOKEN
    if not DEVICE_ID:
        DEVICE_ID = _read_device_id_from_logs()
    if not LOCAL_TOKEN:
        LOCAL_TOKEN = _read_token_from_logs()


# --- ZMQ publish (import from local module) ---

from zmq_publish import publish as zmq_publish_msg


# --- Helpers ---

def make_auth(body_str):
    return hashlib.md5((body_str + LOCAL_TOKEN).encode()).hexdigest()


def build_local_payload(function_id, params=None):
    payload = {
        "version": "1",
        "code": 3,
        "deviceID": DEVICE_ID,
        "payload": {
            "functionID": function_id,
            "requestID": str(uuid.uuid4()).upper(),
            "timestamp": int(time.time() * 1000),
            "params": params or {},
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def flask_error(status, message):
    return jsonify({"detail": message}), status


# LLM instruction -> ZMQ start_cc_task payload mapping
ACTION_MAP = {
    "dance":        {"action": "RDANCE008", "task_name": "music"},
    "sing":         {"action": "RSING001",  "task_name": "music"},
    "take_photo":   {"action": "RPIC001",   "task_name": "take_photo"},
    "welcome":      {"action": "RHUG002",   "task_name": "welcome"},
    "say_hello":    {"action": "RIMGHI001", "task_name": "hello"},
    "bye":          {"action": "RTHDL50",   "task_name": "bye"},
    "good_morning": {"action": "RIMGHI001", "task_name": "good_morning"},
    "good_night":   {"action": "RTHDL50",   "task_name": "good_night"},
    "wave_hand":    {"action": "RIMGHI001", "task_name": "wave_hand"},
    "show_love":    {"action": "RHAPPY001", "task_name": "show_love"},
    "get_praise":   {"action": "RHAPPY001", "task_name": "get_praise"},
    "wake_up":      {"action": "RAWAKE001", "task_name": "wake_up"},
    "nod":          {"action": "RANodyes",  "task_name": "nod"},
    "shake_head":   {"action": "RANO",      "task_name": "shake_head"},
    "speak":        {"action": "RSAYS001",  "task_name": "speak"},
    "look_left":    {"action": "RAFL",      "task_name": "look_left"},
    "look_right":   {"action": "RAFR",      "task_name": "look_right"},
    "look_up":      {"action": "RAUP",      "task_name": "look_up"},
    "look_down":    {"action": "RADOWN",    "task_name": "look_down"},
    "spin":         {"action": "RWCIR001",  "task_name": "spin"},
    "come_over":    {"task_name": "come_over"},
    "follow_me":    {"task_name": "follow_me"},
    "go_away":      {"task_name": "go_away"},
    "go_play":      {"task_name": "go_play"},
    "go_sleep":     {"task_name": "go_sleep"},
    "go_power":     {"task_name": "go_power"},
    "go_to_kitchen":  {"task_name": "go_to_kitchen"},
    "go_to_bedroom":  {"task_name": "go_to_bedroom"},
    "go_to_balcony":  {"task_name": "go_to_balcony"},
    "move_forward": {"task_name": "move_forward"},
    "move_back":    {"task_name": "move_back"},
    "move_left":    {"task_name": "move_left"},
    "move_right":   {"task_name": "move_right"},
    "turn_left":    {"task_name": "turn_left"},
    "turn_right":   {"task_name": "turn_right"},
    "stop":         {"task_name": "stop"},
    "be_silent":    {"task_name": "be_silent"},
    "volume_up":    {"task_name": "volume_up"},
    "volume_down":  {"task_name": "volume_down"},
    "user_leave":   {"task_name": "user_leave"},
}


# --- Endpoints ---

@app.post("/api/action")
def proxy_action():
    """LLM action server (:8080) proxy."""
    data = request.get_json(force=True)
    voice_text = data.get("voiceText", "")
    if not voice_text:
        return flask_error(400, "voiceText is required")
    try:
        resp = requests.post(
            "http://127.0.0.1:8080/rkllm_action",
            json={"voiceText": voice_text},
            timeout=30,
        )
        text = resp.text.strip()
        parts = text.split("/", 1)
        return jsonify({
            "raw": text,
            "mood": parts[0] if parts else text,
            "instruction": parts[1] if len(parts) > 1 else "",
        })
    except requests.ConnectionError:
        return flask_error(502, "action server unreachable")
    except Exception as e:
        return flask_error(502, str(e))


DIARY_RECORD_PATH = "/data/control_center/db/diary_record.json"
GENERATED_DIARIES_PATH = "/data/devtools/generated_diaries.json"


def _load_generated_diaries():
    if not os.path.isfile(GENERATED_DIARIES_PATH):
        return {}
    try:
        with open(GENERATED_DIARIES_PATH, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except Exception:
        return {}


def _save_generated_diary(date, diary_data):
    diaries = _load_generated_diaries()
    diaries[date] = {**diary_data, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(GENERATED_DIARIES_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(diaries, ensure_ascii=False, indent=2))


@app.get("/api/diary/records")
def get_diary_records():
    """Return event records and generated diaries from device."""
    events = {}
    if os.path.isfile(DIARY_RECORD_PATH):
        try:
            with open(DIARY_RECORD_PATH, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            events = data.get("diary_event_records", {})
        except Exception:
            pass
    return jsonify({
        "events": events,
        "generated": _load_generated_diaries(),
    })


@app.post("/api/diary")
def proxy_diary():
    """Proxy to route server (:8083) — same pipeline as production."""
    data = request.get_json(force=True)
    events = data.get("events", [])
    language = data.get("language", "ja")
    local_date = data.get("local_date", "") or time.strftime("%Y-%m-%d")

    # Send to route server in the exact format it expects
    # Route server handles: dedup, retry (x3), format validation, translation
    payload = {
        "language": language,
        "local_date": local_date,
        "events": events,
    }
    try:
        resp = requests.post(
            "http://127.0.0.1:8083/rkllm_diary",
            json=payload,
            timeout=600,
        )
        body = resp.json()
        # Route server returns {"resultCode":100,"data":{"title":..,"diary":..,"emotion":..}}
        if body.get("resultCode") == 100 and body.get("data"):
            result = body["data"]
            _save_generated_diary(local_date, result)
            return jsonify(result)
        return jsonify({"error": body.get("message", "unknown error"), "raw": body})
    except requests.ConnectionError:
        return flask_error(502, "route server (8083) unreachable")
    except requests.Timeout:
        return flask_error(504, "route server timeout (600s)")
    except Exception as e:
        return flask_error(502, str(e))


@app.post("/api/local")
def proxy_local():
    """Local API (:27999) proxy with MD5 auth."""
    _ensure_config()
    data = request.get_json(force=True)
    function_id = data.get("function_id")
    params = data.get("params")

    body_str = build_local_payload(function_id, params)
    auth = make_auth(body_str)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "auth": auth,
    }
    try:
        resp = requests.post(
            f"http://127.0.0.1:{LOCAL_API_PORT}/thing_model/func_request",
            data=body_str,
            headers=headers,
            timeout=10,
        )
        return jsonify(resp.json())
    except requests.ConnectionError:
        return flask_error(502, "local API unreachable")
    except Exception as e:
        return flask_error(502, str(e))


@app.post("/api/zmq/publish")
def zmq_publish():
    """Publish directly to local ZMQ bus (:5558)."""
    data = request.get_json(force=True)
    topic = data.get("topic", "/ai/do_action")
    payload = data.get("payload", {})

    payload_json = json.dumps(payload, separators=(",", ":"))
    try:
        zmq_publish_msg(topic, payload_json)
        return jsonify({"status": "ok", "topic": topic, "payload": payload})
    except Exception as e:
        return flask_error(502, f"ZMQ publish failed: {e}")


CUSTOM_PROMPT_PATH = "/data/devtools/custom_prompt.txt"
CUSTOM_LLM_CONFIG_PATH = "/data/devtools/custom_llm_config.json"


@app.post("/api/execute")
def execute_action():
    """LLM classify -> ZMQ publish pipeline."""
    _ensure_config()
    data = request.get_json(force=True)
    voice_text = data.get("voiceText", "")
    if not voice_text:
        return flask_error(400, "voiceText is required")

    # Step 1: LLM classification
    try:
        resp = requests.post(
            "http://127.0.0.1:8080/rkllm_action",
            json={"voiceText": voice_text},
            timeout=30,
        )
    except requests.ConnectionError:
        return flask_error(502, "action server unreachable")

    text = resp.text.strip()
    parts = text.split("/", 1)
    mood = parts[0] if parts else text
    instruction = parts[1] if len(parts) > 1 else ""

    result = {"raw": text, "mood": mood, "instruction": instruction}

    # Step 2: Map instruction -> ZMQ publish
    if instruction and instruction != "no_action":
        mapping = ACTION_MAP.get(instruction)
        if mapping:
            ts = int(time.time() * 1e9)
            payload_dict = {"task_type": "voice", "timestamp": ts, **mapping}
            payload_json = json.dumps(payload_dict, separators=(",", ":"))
            try:
                zmq_publish_msg("/agent/start_cc_task", payload_json)
                result["payload"] = payload_dict
                result["executed"] = True
            except Exception:
                result["executed"] = False
                result["error"] = "ZMQ publish failed"
        else:
            result["executed"] = False
            result["error"] = f"unknown instruction: {instruction}"
    else:
        result["executed"] = False

    return jsonify(result)


@app.post("/api/custom-llm")
def custom_llm_call():
    """Call diary LLM (Qwen3-1.7B :8082) with custom prompt template."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return flask_error(400, "text is required")

    # Read template
    if not os.path.isfile(CUSTOM_PROMPT_PATH):
        return flask_error(404, "custom_prompt.txt not found — Prompt tab で設定してください")
    try:
        with open(CUSTOM_PROMPT_PATH, "r", encoding="utf-8") as f:
            template = f.read().strip()
    except Exception as e:
        return flask_error(500, f"template read error: {e}")
    if not template:
        return flask_error(400, "custom_prompt.txt is empty")

    # Format template
    try:
        filled = template.format(text=text)
    except (KeyError, IndexError, ValueError) as e:
        return flask_error(400, f"template format error: {e}")

    # Read config (temperature, max_new_tokens)
    config = {"temperature": 1.3, "max_new_tokens": 4096}
    if os.path.isfile(CUSTOM_LLM_CONFIG_PATH):
        try:
            with open(CUSTOM_LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
                config.update(json.loads(f.read()))
        except Exception:
            pass

    # Call diary server with retries
    payload = {
        "task": config.get("task", "diary"),
        "prompt": filled,
        "temperature": float(config.get("temperature", 1.3)),
        "max_new_tokens": int(config.get("max_new_tokens", 4096)),
    }

    MAX_RETRIES = 3
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "http://127.0.0.1:8082/rkllm_diary",
                json=payload,
                timeout=300,
            )
            if resp.status_code == 503:
                last_error = "LLM server busy"
                time.sleep(2)
                continue
            result_text = resp.text.strip()
            if result_text:
                result_text = re.sub(
                    r"<think>.*?</think>", "", result_text, flags=re.DOTALL
                ).strip()
                return jsonify({
                    "result": result_text,
                    "prompt": filled,
                    "attempt": attempt + 1,
                })
            last_error = "empty response"
        except requests.ConnectionError:
            return flask_error(502, "diary server (8082) unreachable")
        except requests.Timeout:
            last_error = "timeout"
        except Exception as e:
            last_error = str(e)
        time.sleep(2)

    return jsonify({
        "result": "",
        "error": f"no output after {MAX_RETRIES} attempts ({last_error})",
        "prompt": filled,
    })


# --- Sensors ---

_prev_cpu = None  # (idle, total) for CPU delta calculation


def _read_file(path):
    """Read a sysfs/proc file, return stripped content or None."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def _read_thermal_zones():
    """Read all thermal_zone temperatures."""
    zones = []
    for i in range(10):
        temp_str = _read_file(f"/sys/class/thermal/thermal_zone{i}/temp")
        if temp_str is None:
            break
        type_str = _read_file(f"/sys/class/thermal/thermal_zone{i}/type") or f"zone{i}"
        try:
            temp_c = int(temp_str) / 1000.0
        except ValueError:
            temp_c = 0
        zones.append({"zone": i, "type": type_str, "temp": round(temp_c, 1)})
    return zones


def _read_adc():
    """Read ADC channels from iio:device0."""
    base = "/sys/bus/iio/devices/iio:device0"
    scale_str = _read_file(f"{base}/in_voltage_scale")
    scale = float(scale_str) if scale_str else 1.0
    channels = []
    for ch in range(8):
        raw_str = _read_file(f"{base}/in_voltage{ch}_raw")
        if raw_str is None:
            continue
        try:
            raw = int(raw_str)
        except ValueError:
            raw = 0
        voltage = round(raw * scale / 1000.0, 3)
        channels.append({"channel": ch, "raw": raw, "voltage": voltage})
    return {"channels": channels, "scale": scale}


def _read_touch():
    """Read touch sensor sensitivity."""
    val = _read_file("/sys/bus/i2c/devices/6-0058/sensitivity")
    return {"sensitivity": int(val) if val else None}


def _read_leds():
    """Read LED brightness values."""
    leds = []
    led_dirs = sorted(glob.glob("/sys/class/leds/*/"))
    for d in led_dirs:
        name = os.path.basename(d.rstrip("/"))
        brightness = _read_file(os.path.join(d, "brightness"))
        max_br = _read_file(os.path.join(d, "max_brightness"))
        if brightness is not None:
            leds.append({
                "name": name,
                "brightness": int(brightness),
                "max_brightness": int(max_br) if max_br else None,
            })
    return leds


def _read_npu():
    """Read NPU load from debugfs."""
    val = _read_file("/sys/kernel/debug/rknpu/load")
    if not val:
        return {"raw": None, "cores": []}
    # Format example: "Core0: 35%, Core1: 35%"
    cores = []
    for m in re.finditer(r'Core(\d+):\s*(\d+)%', val):
        cores.append({"core": int(m.group(1)), "load": int(m.group(2))})
    return {"raw": val, "cores": cores}


def _read_cpu():
    """Read CPU usage from /proc/stat (delta between calls)."""
    global _prev_cpu
    line = _read_file("/proc/stat")
    if not line:
        return {"percent": None}
    first_line = line.split("\n")[0]  # "cpu  user nice system idle ..."
    parts = first_line.split()
    if len(parts) < 5:
        return {"percent": None}
    values = [int(x) for x in parts[1:]]
    idle = values[3]
    total = sum(values)
    if _prev_cpu:
        prev_idle, prev_total = _prev_cpu
        d_idle = idle - prev_idle
        d_total = total - prev_total
        percent = round((1 - d_idle / d_total) * 100, 1) if d_total > 0 else 0
    else:
        percent = None  # first call, no delta yet
    _prev_cpu = (idle, total)
    return {"percent": percent}


def _read_memory():
    """Read memory info from /proc/meminfo."""
    content = _read_file("/proc/meminfo")
    if not content:
        return {}
    info = {}
    for line in content.split("\n"):
        parts = line.split(":")
        if len(parts) == 2:
            key = parts[0].strip()
            val = parts[1].strip().split()[0]  # value in kB
            if key in ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal", "SwapFree"):
                try:
                    info[key] = int(val)
                except ValueError:
                    pass
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", 0)
    used = total - available
    return {
        "total_kb": total,
        "available_kb": available,
        "used_kb": used,
        "percent": round(used / total * 100, 1) if total > 0 else 0,
    }


def _read_i2c_devices():
    """List I2C device names."""
    devices = []
    for path in sorted(glob.glob("/sys/bus/i2c/devices/*/name")):
        name = _read_file(path)
        bus_addr = path.split("/")[-2]
        if name:
            devices.append({"bus_addr": bus_addr, "name": name})
    return devices


def _read_cameras():
    """List video4linux camera devices."""
    cameras = []
    for d in sorted(glob.glob("/sys/class/video4linux/*/")):
        dev_name = os.path.basename(d.rstrip("/"))
        name = _read_file(os.path.join(d, "name")) or dev_name
        cameras.append({"device": dev_name, "name": name})
    return cameras


def _read_audio():
    """Read audio/speaker info from ALSA."""
    info = {"cards": [], "volume": None}
    # Sound cards
    cards_str = _read_file("/proc/asound/cards")
    if cards_str:
        for m in re.finditer(r'^\s*(\d+)\s+\[(\w+)\s*\]:\s*(.+)$', cards_str, re.MULTILINE):
            info["cards"].append({
                "id": int(m.group(1)),
                "name": m.group(2),
                "description": m.group(3).strip(),
            })
    # Volume via amixer (best-effort)
    try:
        out = subprocess.check_output(
            ["amixer", "get", "Master"], timeout=3, stderr=subprocess.DEVNULL
        ).decode(errors="ignore")
        m = re.search(r'\[(\d+)%\]', out)
        if m:
            info["volume"] = int(m.group(1))
        m2 = re.search(r'\[(on|off)\]', out)
        if m2:
            info["mute"] = m2.group(1) == "off"
    except Exception:
        pass
    return info


def _read_battery():
    """Read battery / power supply info."""
    supplies = []
    for d in sorted(glob.glob("/sys/class/power_supply/*/")):
        name = os.path.basename(d.rstrip("/"))
        ptype = _read_file(os.path.join(d, "type")) or "Unknown"
        entry = {"name": name, "type": ptype}
        for key in ("status", "capacity", "voltage_now", "current_now", "charge_full", "charge_now", "online"):
            val = _read_file(os.path.join(d, key))
            if val is not None:
                try:
                    entry[key] = int(val)
                except ValueError:
                    entry[key] = val
        supplies.append(entry)
    return supplies


def _read_wifi():
    """Read WiFi signal info from /proc/net/wireless."""
    content = _read_file("/proc/net/wireless")
    if not content:
        return {}
    lines = content.strip().split("\n")
    # Skip header lines
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            iface = parts[0].rstrip(":")
            return {
                "interface": iface,
                "link": float(parts[2].rstrip(".")),
                "level_dbm": float(parts[3].rstrip(".")),
                "noise_dbm": float(parts[4].rstrip(".")) if len(parts) > 4 else None,
            }
    return {}


def _read_disk():
    """Read disk usage via statvfs."""
    mounts = ["/", "/data"]
    disks = []
    for mp in mounts:
        try:
            st = os.statvfs(mp)
            total = st.f_frsize * st.f_blocks
            free = st.f_frsize * st.f_bavail
            used = total - free
            disks.append({
                "mount": mp,
                "total_mb": round(total / 1048576),
                "used_mb": round(used / 1048576),
                "free_mb": round(free / 1048576),
                "percent": round(used / total * 100, 1) if total > 0 else 0,
            })
        except Exception:
            pass
    return disks


def _read_lidar():
    """Read LiDAR device info if available."""
    # Common sysfs paths for USB/serial LiDAR
    info = {}
    for path in glob.glob("/sys/class/tty/ttyUSB*/device/interface"):
        val = _read_file(path)
        if val:
            dev = path.split("/")[4]
            info[dev] = val
    # Also check for rplidar or similar
    for path in glob.glob("/dev/ttyUSB*"):
        dev = os.path.basename(path)
        if dev not in info:
            info[dev] = "serial device"
    return info


@app.get("/api/sensors")
def get_sensors():
    """Return all hardware sensor data."""
    return jsonify({
        "thermal": _read_thermal_zones(),
        "adc": _read_adc(),
        "touch": _read_touch(),
        "leds": _read_leds(),
        "npu": _read_npu(),
        "cpu": _read_cpu(),
        "memory": _read_memory(),
        "i2c_devices": _read_i2c_devices(),
        "cameras": _read_cameras(),
        "audio": _read_audio(),
        "battery": _read_battery(),
        "wifi": _read_wifi(),
        "disk": _read_disk(),
        "lidar": _read_lidar(),
        "timestamp": time.time(),
    })


@app.get("/api/sensors/camera/snapshot")
def camera_snapshot():
    """Capture a live frame from ISP selfpath and return as JPEG."""
    from flask import Response

    video_dev = "/dev/video12"  # rkisp_selfpath
    width, height = 640, 480
    try:
        # v4l2-ctl captures one NV12 frame, ffmpeg converts to JPEG
        v4l2 = subprocess.Popen(
            ["v4l2-ctl", "-d", video_dev,
             "--set-fmt-video", f"width={width},height={height},pixelformat=NV12",
             "--stream-mmap", "--stream-count=1", "--stream-to=-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "nv12",
             "-video_size", f"{width}x{height}",
             "-i", "-", "-frames:v", "1",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"],
            stdin=v4l2.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        v4l2.stdout.close()
        jpeg_data, _ = ffmpeg.communicate(timeout=5)
        v4l2.wait(timeout=2)
        if not jpeg_data:
            return flask_error(500, "empty frame")
        return Response(jpeg_data, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-cache, no-store"})
    except subprocess.TimeoutExpired:
        for p in (v4l2, ffmpeg):
            try:
                p.kill()
            except Exception:
                pass
        return flask_error(504, "capture timeout")
    except Exception as e:
        return flask_error(500, f"capture failed: {e}")


@app.post("/api/sensors/volume")
def set_volume():
    """Set master volume via amixer."""
    data = request.get_json(force=True)
    volume = data.get("volume")
    if volume is None:
        return flask_error(400, "volume is required")
    volume = max(0, min(100, int(volume)))
    try:
        subprocess.check_output(
            ["amixer", "set", "Master", f"{volume}%"],
            timeout=5, stderr=subprocess.DEVNULL,
        )
        return jsonify({"status": "ok", "volume": volume})
    except Exception as e:
        return flask_error(502, f"amixer failed: {e}")


@app.get("/api/health")
def health_check():
    """Check local service ports."""
    ports = {
        "zmq_xpub": 5558,
        "zmq_xsub": 5559,
        "llm_action": 8080,
        "llm_diary": 8082,
        "llm_router": 8083,
        "local_api": LOCAL_API_PORT,
        "control_center": 50001,
    }
    results = {}
    for name, port in ports.items():
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/", timeout=2)
            results[name] = {"port": port, "status": "up", "code": resp.status_code}
        except requests.ConnectionError:
            results[name] = {"port": port, "status": "down"}
        except requests.Timeout:
            results[name] = {"port": port, "status": "up (timeout)"}
        except Exception:
            results[name] = {"port": port, "status": "up (non-http)"}
    return jsonify({"ip": "127.0.0.1 (on-device)", "services": results})


# --- Camera / Media ---

CAMERA_DIRS = {
    "media_photo": "/media/photo/",
    "origin": "/data/cache/video_recorder/result/origin/",
    "hand": "/data/cache/video_recorder/result/hand/",
    "photos": "/data/cache/photo/",
    "video": "/data/cache/video_recorder/archive/",
    "video_archive": "/data/cache/video_recorder_archive/",
    "sensor": "/data/cache/recorder/archive/",
    "face_known": "/data/ai_brain_data/face_metadata/known/",
    "face_unknown": "/data/ai_brain_data/face_metadata/unknown/",
}

# Directories where multiple file variants exist per item (e.g. original + thumb + mini)
# Only list files matching this extension in summary/list to avoid duplicates
PHOTO_EXT_FILTER = {
    "media_photo": ".png",  # /media/photo/ has .png + _mini.jpg + _thumb.jpg per photo
}


RECURSIVE_DIRS = {"face_known", "face_unknown"}


def _remove_empty_dirs(path):
    """Remove a directory tree bottom-up if all subdirs are empty."""
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        sub = os.path.join(path, entry)
        if os.path.isdir(sub):
            _remove_empty_dirs(sub)
    # Try removing if now empty
    try:
        os.rmdir(path)
    except OSError:
        pass


def _dir_stats(path, recursive=False, ext_filter=None):
    """Return (count, total_size) for files in a directory."""
    if not os.path.isdir(path):
        return 0, 0
    count = 0
    total = 0
    if recursive:
        for root, _dirs, files in os.walk(path):
            for name in files:
                if ext_filter and not name.endswith(ext_filter):
                    continue
                fp = os.path.join(root, name)
                count += 1
                total += os.path.getsize(fp)
    else:
        for name in os.listdir(path):
            if ext_filter and not name.endswith(ext_filter):
                continue
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                count += 1
                total += os.path.getsize(fp)
    return count, total


@app.get("/api/camera/summary")
def camera_summary():
    result = {}
    for key, path in CAMERA_DIRS.items():
        ext = PHOTO_EXT_FILTER.get(key)
        count, total_size = _dir_stats(path, recursive=(key in RECURSIVE_DIRS), ext_filter=ext)
        result[key] = {"count": count, "total_size": total_size}
    return jsonify(result)


@app.get("/api/camera/list")
def camera_list():
    cat = request.args.get("type", "origin")
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 20, type=int)

    dirpath = CAMERA_DIRS.get(cat)
    if not dirpath or not os.path.isdir(dirpath):
        return jsonify({"files": [], "total": 0, "offset": offset, "limit": limit})

    ext_filter = PHOTO_EXT_FILTER.get(cat)
    entries = []
    if cat in RECURSIVE_DIRS:
        for root, _dirs, files in os.walk(dirpath):
            for name in files:
                if ext_filter and not name.endswith(ext_filter):
                    continue
                fp = os.path.join(root, name)
                relpath = os.path.relpath(fp, dirpath)
                entries.append({
                    "name": relpath,
                    "size": os.path.getsize(fp),
                    "mtime": int(os.path.getmtime(fp)),
                })
    else:
        for name in os.listdir(dirpath):
            if ext_filter and not name.endswith(ext_filter):
                continue
            fp = os.path.join(dirpath, name)
            if os.path.isfile(fp):
                entries.append({
                    "name": name,
                    "size": os.path.getsize(fp),
                    "mtime": int(os.path.getmtime(fp)),
                })

    entries.sort(key=lambda e: (e["mtime"], e["name"]), reverse=True)
    total = len(entries)
    return jsonify({
        "files": entries[offset:offset + limit],
        "total": total,
        "offset": offset,
        "limit": limit,
    })


@app.get("/api/camera/photo/<cat>/<path:filename>")
def camera_photo(cat, filename):
    dirpath = CAMERA_DIRS.get(cat)
    if not dirpath:
        abort(404)
    # Prevent path traversal
    if ".." in filename or filename.startswith("/"):
        abort(400)
    return send_from_directory(dirpath, filename)


@app.get("/api/camera/faces")
def camera_faces():
    """List face IDs with per-subfolder counts and thumbnail."""
    kind = request.args.get("kind", "known")  # known or unknown
    dirpath = CAMERA_DIRS.get(f"face_{kind}")
    if not dirpath or not os.path.isdir(dirpath):
        return jsonify({"ids": []})

    include_empty = request.args.get("include_empty", "0") == "1"
    ids = []
    empty_count = 0
    for entry in sorted(os.listdir(dirpath), reverse=True):
        id_path = os.path.join(dirpath, entry)
        if not os.path.isdir(id_path):
            continue
        info = {"id": entry, "enrolled": [], "recognized_count": 0, "features_count": 0}
        # enrolled_faces (known) or faces (unknown)
        for ef_name in ("enrolled_faces", "faces"):
            ef_path = os.path.join(id_path, ef_name)
            if os.path.isdir(ef_path):
                files_in = [f for f in sorted(os.listdir(ef_path)) if os.path.isfile(os.path.join(ef_path, f))]
                if files_in:
                    info["enrolled"] = files_in[:5]  # max 5 thumbnails
                    info["enrolled_dir"] = ef_name
                    break
        # recognized_faces
        rf_path = os.path.join(id_path, "recognized_faces")
        if os.path.isdir(rf_path):
            info["recognized_count"] = len(os.listdir(rf_path))
        # features
        ft_path = os.path.join(id_path, "features")
        if os.path.isdir(ft_path):
            info["features_count"] = len(os.listdir(ft_path))
        total_files = len(info["enrolled"]) + info["recognized_count"] + info["features_count"]
        if total_files == 0:
            empty_count += 1
            if not include_empty:
                continue
        ids.append(info)
    return jsonify({"ids": ids, "empty_count": empty_count})


@app.get("/api/camera/face_files")
def camera_face_files():
    """List files in a specific face ID subfolder."""
    kind = request.args.get("kind", "known")
    face_id = request.args.get("id", "")
    subfolder = request.args.get("sub", "recognized_faces")
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 20, type=int)

    if not face_id or ".." in face_id or ".." in subfolder:
        abort(400)
    if subfolder not in ("enrolled_faces", "recognized_faces", "features", "faces"):
        abort(400)

    dirpath = os.path.join(CAMERA_DIRS.get(f"face_{kind}", ""), face_id, subfolder)
    if not os.path.isdir(dirpath):
        return jsonify({"files": [], "total": 0})

    entries = []
    for name in os.listdir(dirpath):
        fp = os.path.join(dirpath, name)
        if os.path.isfile(fp):
            entries.append({
                "name": name,
                "size": os.path.getsize(fp),
                "mtime": int(os.path.getmtime(fp)),
            })
    entries.sort(key=lambda e: (e["mtime"], e["name"]), reverse=True)
    total = len(entries)
    return jsonify({
        "files": entries[offset:offset + limit],
        "total": total,
        "offset": offset,
        "limit": limit,
    })


@app.post("/api/camera/cleanup_empty_faces")
def cleanup_empty_faces():
    """Remove face ID directories that have no files."""
    import shutil
    kind = request.args.get("kind", "known")
    dirpath = CAMERA_DIRS.get(f"face_{kind}")
    if not dirpath or not os.path.isdir(dirpath):
        return jsonify({"removed": 0})
    removed = 0
    for entry in os.listdir(dirpath):
        id_path = os.path.join(dirpath, entry)
        if not os.path.isdir(id_path):
            continue
        # Check if any files exist recursively
        has_files = False
        for _root, _dirs, files in os.walk(id_path):
            if files:
                has_files = True
                break
        if not has_files:
            shutil.rmtree(id_path, ignore_errors=True)
            removed += 1
    return jsonify({"removed": removed})


@app.post("/api/camera/delete")
def camera_delete():
    """Delete files or entire face IDs from a camera directory."""
    import shutil
    data = request.get_json(force=True)
    cat = data.get("type", "")
    files = data.get("files", [])
    face_ids = data.get("face_ids", [])

    dirpath = CAMERA_DIRS.get(cat)
    if not dirpath:
        return flask_error(400, f"unknown type: {cat}")
    if not files and not face_ids:
        return flask_error(400, "no files or face_ids specified")

    deleted = 0
    errors = []
    is_face = cat in ("face_known", "face_unknown")
    affected_face_ids = set()

    # Delete entire face ID directories
    face_ids_deleted = 0
    for fid in face_ids:
        if ".." in fid or "/" in fid or fid.startswith("."):
            errors.append({"file": fid, "error": "invalid face_id"})
            continue
        face_dir = os.path.join(dirpath, fid)
        real = os.path.realpath(face_dir)
        if not real.startswith(os.path.realpath(dirpath)):
            errors.append({"file": fid, "error": "path traversal"})
            continue
        if not os.path.isdir(real):
            errors.append({"file": fid, "error": "not found"})
            continue
        try:
            shutil.rmtree(real)
            face_ids_deleted += 1
        except Exception as e:
            errors.append({"file": fid, "error": str(e)})

    # Delete individual files
    for fname in files:
        if ".." in fname or fname.startswith("/"):
            errors.append({"file": fname, "error": "invalid path"})
            continue
        fp = os.path.join(dirpath, fname)
        real = os.path.realpath(fp)
        if not real.startswith(os.path.realpath(dirpath)):
            errors.append({"file": fname, "error": "path traversal"})
            continue
        if not os.path.isfile(real):
            errors.append({"file": fname, "error": "not found"})
            continue
        try:
            os.remove(real)
            deleted += 1
            if is_face:
                affected_face_ids.add(fname.split("/")[0])
            # For /media/photo/: also delete _mini.jpg and _thumb.jpg variants
            if cat == "media_photo" and fname.endswith(".png"):
                base = fname[:-4]
                for suffix in ("_mini.jpg", "_thumb.jpg"):
                    variant = os.path.join(dirpath, base + suffix)
                    if os.path.isfile(variant):
                        try:
                            os.remove(variant)
                        except Exception:
                            pass
        except Exception as e:
            errors.append({"file": fname, "error": str(e)})

    # Clean up features for affected face IDs (individual file deletion)
    features_deleted = 0
    for face_id in affected_face_ids:
        feat_dir = os.path.join(dirpath, face_id, "features")
        if not os.path.isdir(feat_dir):
            continue
        for name in os.listdir(feat_dir):
            fp = os.path.join(feat_dir, name)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                    features_deleted += 1
                except Exception:
                    pass
        try:
            os.rmdir(feat_dir)
        except OSError:
            pass
        face_dir = os.path.join(dirpath, face_id)
        try:
            _remove_empty_dirs(face_dir)
        except Exception:
            pass

    return jsonify({
        "deleted": deleted,
        "face_ids_deleted": face_ids_deleted,
        "features_deleted": features_deleted,
        "errors": errors,
    })


# --- System Prompts & Launch Scripts ---

PROMPT_DIR = "/app/opt/wlab/sweepbot/share/llm_server/res"
SCRIPT_DIR = "/app/opt/wlab/sweepbot/bin"
# Mirror paths: overlay writes go here so LLM servers see changes via /opt/...
_OVERLAY_PROMPT_DIR = "/opt/wlab/sweepbot/share/llm_server/res"
_OVERLAY_SCRIPT_DIR = "/opt/wlab/sweepbot/bin"

# All editable files: key -> (directory, filename)
EDITABLE_FILES = {
    "action":             (PROMPT_DIR, "action_system_prompt.txt"),
    "action_config":      (SCRIPT_DIR, "llm_action_server.sh"),
    "diary":              (PROMPT_DIR, "system_prompt_diary.txt"),
    "diary_config":       (SCRIPT_DIR, "llm_diary_server.sh"),
    "diary_translation":  (PROMPT_DIR, "system_prompt_diary_translation.txt"),
    "custom_llm":         ("/data/devtools", "custom_prompt.txt"),
    "custom_llm_config":  ("/data/devtools", "custom_llm_config.json"),
}

# LLM services to restart after prompt edit
LLM_SERVICES = ["llm_action", "llm_diary", "llm_route"]

# Map /app/opt/... dirs to /opt/... dirs for overlay sync
_OVERLAY_MAP = {PROMPT_DIR: _OVERLAY_PROMPT_DIR, SCRIPT_DIR: _OVERLAY_SCRIPT_DIR}


def _editable_path(key):
    d, f = EDITABLE_FILES[key]
    return os.path.join(d, f)


def _init_overlay_dirs():
    """Ensure overlay directories exist and are writable on first run."""
    for overlay_dir in _OVERLAY_MAP.values():
        try:
            os.makedirs(overlay_dir, mode=0o777, exist_ok=True)
        except Exception:
            pass
    # Sync all editable files from /app/opt/... to /opt/... on startup
    for key, (d, fname) in EDITABLE_FILES.items():
        overlay_dir = _OVERLAY_MAP.get(d)
        if not overlay_dir:
            continue
        src = os.path.join(d, fname)
        dst = os.path.join(overlay_dir, fname)
        try:
            with open(src, "r", encoding="utf-8") as f:
                content = f.read()
            with open(dst, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass


def _sync_to_overlay(directory, filename, content):
    """Write content to /opt/... overlay path so LLM servers see the change."""
    overlay_dir = _OVERLAY_MAP.get(directory)
    if not overlay_dir:
        return
    try:
        dst = os.path.join(overlay_dir, filename)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass  # best-effort sync


_init_overlay_dirs()


@app.get("/api/prompts")
def get_prompts():
    """Return all editable files (prompts + launch scripts)."""
    result = {}
    for key, (d, fname) in EDITABLE_FILES.items():
        fpath = os.path.join(d, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                result[key] = {"filename": fname, "content": f.read()}
        except Exception as e:
            result[key] = {"filename": fname, "content": "", "error": str(e)}
    return jsonify(result)


@app.post("/api/prompts/save")
def save_prompt():
    """Save an editable file."""
    data = request.get_json(force=True)
    key = data.get("key", "")
    content = data.get("content", "")
    if key not in EDITABLE_FILES:
        return flask_error(400, f"unknown key: {key}")
    fpath = _editable_path(key)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        d, fname = EDITABLE_FILES[key]
        _sync_to_overlay(d, fname, content)
        return jsonify({"status": "ok", "key": key, "size": len(content)})
    except Exception as e:
        return flask_error(500, str(e))


PROMPT_BACKUP_DIR = "/data/devtools/prompt_backups"


@app.get("/api/prompts/backups")
def list_prompt_backups():
    """List available prompt backups."""
    if not os.path.isdir(PROMPT_BACKUP_DIR):
        return jsonify({"backups": []})
    backups = []
    for name in sorted(os.listdir(PROMPT_BACKUP_DIR), reverse=True):
        bp = os.path.join(PROMPT_BACKUP_DIR, name)
        if not os.path.isdir(bp):
            continue
        files = sorted(os.listdir(bp))
        backups.append({"name": name, "files": files})
    return jsonify({"backups": backups})


@app.post("/api/prompts/backup")
def backup_prompts():
    """Backup all current editable files."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(PROMPT_BACKUP_DIR, ts)
    os.makedirs(dst, exist_ok=True)
    copied = []
    for key, (d, fname) in EDITABLE_FILES.items():
        src = os.path.join(d, fname)
        if os.path.isfile(src):
            try:
                with open(src, "r", encoding="utf-8") as f:
                    content = f.read()
                with open(os.path.join(dst, fname), "w", encoding="utf-8") as f:
                    f.write(content)
                copied.append(fname)
            except Exception:
                pass
    return jsonify({"status": "ok", "name": ts, "files": copied})


@app.post("/api/prompts/restore")
def restore_prompts():
    """Restore editable files from a backup."""
    data = request.get_json(force=True)
    name = data.get("name", "")
    if not name or ".." in name or "/" in name:
        return flask_error(400, "invalid backup name")
    bp = os.path.join(PROMPT_BACKUP_DIR, name)
    if not os.path.isdir(bp):
        return flask_error(404, f"backup not found: {name}")
    restored = []
    for key, (d, fname) in EDITABLE_FILES.items():
        src = os.path.join(bp, fname)
        dst = os.path.join(d, fname)
        if os.path.isfile(src):
            try:
                with open(src, "r", encoding="utf-8") as f:
                    content = f.read()
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(content)
                _sync_to_overlay(d, fname, content)
                restored.append(fname)
            except Exception as e:
                return flask_error(500, f"restore failed for {fname}: {e}")
    return jsonify({"status": "ok", "name": name, "restored": restored})


@app.post("/api/prompts/backup/delete")
def delete_prompt_backup():
    """Delete a prompt backup."""
    import shutil
    data = request.get_json(force=True)
    name = data.get("name", "")
    if not name or ".." in name or "/" in name:
        return flask_error(400, "invalid backup name")
    bp = os.path.join(PROMPT_BACKUP_DIR, name)
    if not os.path.isdir(bp):
        return flask_error(404, f"backup not found: {name}")
    shutil.rmtree(bp)
    return jsonify({"status": "ok", "name": name})


@app.post("/api/prompts/restart")
def restart_llm_services():
    """Restart LLM services to pick up prompt changes."""
    results = {}
    for svc in LLM_SERVICES:
        try:
            rc = os.system(f"systemctl restart {svc} 2>/dev/null")
            results[svc] = "ok" if rc == 0 else f"exit code {rc}"
        except Exception as e:
            results[svc] = str(e)
    return jsonify({"status": "ok", "services": results})


@app.get("/api/events")
def get_events():
    """Return recent events from log file."""
    log_path = "/data/cache/log/kata_events.jsonl"
    if not os.path.exists(log_path):
        return jsonify({"events": [], "total": 0})
    with open(log_path, "r") as f:
        lines = f.read().strip().split("\n")
    n = request.args.get("n", 50, type=int)
    events = []
    for line in lines[-n:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return jsonify({"events": list(reversed(events)), "total": len(lines)})


# --- Static file serving ---

@app.route("/")
def index():
    resp = send_from_directory(app.static_folder, "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


if __name__ == "__main__":
    _ensure_config()
    print(f"[DevTools] Device ID: {DEVICE_ID or '(not found)'}")
    print(f"[DevTools] Token: {'***' + LOCAL_TOKEN[-4:] if LOCAL_TOKEN else '(not found)'}")
    app.run(host="0.0.0.0", port=9001, debug=False, threaded=True)
