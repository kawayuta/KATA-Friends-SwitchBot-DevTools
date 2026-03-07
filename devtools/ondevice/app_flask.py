"""
Kata Friends Developer Tools — Flask on-device backend

Device dependencies only: Flask 3.0.2, requests, jinja2
Runs directly on device at /data/devtools/, port 9001.
"""

import ctypes
import glob
import hashlib
import json
import datetime
import os
import re
import struct
import subprocess
import threading
import time
import uuid

import requests
from flask import Flask, Response, abort, jsonify, request, send_from_directory

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
LLM_BACKEND_CONFIG_PATH = "/data/devtools/llm_backend_config.json"

_LLM_BACKEND_DEFAULTS = {
    "backend": "device",
    "lmstudio_url": "http://192.168.11.xx:1234/v1",
    "lmstudio_model": "",
    "lmstudio_api_key": "",
}


def _load_llm_backend_config():
    if os.path.isfile(LLM_BACKEND_CONFIG_PATH):
        try:
            with open(LLM_BACKEND_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.loads(f.read())
            merged = dict(_LLM_BACKEND_DEFAULTS)
            merged.update(cfg)
            return merged
        except Exception:
            pass
    return dict(_LLM_BACKEND_DEFAULTS)


@app.get("/api/llm-backend")
def get_llm_backend():
    return jsonify(_load_llm_backend_config())


@app.post("/api/llm-backend")
def set_llm_backend():
    data = request.get_json(force=True)
    cfg = _load_llm_backend_config()
    for key in ("backend", "lmstudio_url", "lmstudio_model", "lmstudio_api_key"):
        if key in data:
            cfg[key] = data[key]
    # Normalize URL: ensure /v1 suffix
    ext_url = cfg.get("lmstudio_url", "").rstrip("/")
    if ext_url and not ext_url.endswith("/v1"):
        cfg["lmstudio_url"] = ext_url + "/v1"
    os.makedirs(os.path.dirname(LLM_BACKEND_CONFIG_PATH), exist_ok=True)
    with open(LLM_BACKEND_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2))
    return jsonify(cfg)


@app.post("/api/llm-backend/test")
def test_llm_backend():
    """Proxy connection test to LM Studio /models endpoint."""
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip().rstrip("/")
    api_key = (data.get("api_key") or "").strip()
    if not url:
        return flask_error(400, "url is required")
    if not url.endswith("/v1"):
        url += "/v1"
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get(f"{url}/models", headers=headers, timeout=10)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.ConnectionError:
        return flask_error(502, f"接続失敗: {url}")
    except requests.Timeout:
        return flask_error(504, "タイムアウト")
    except Exception as e:
        return flask_error(502, str(e))


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


def _lmstudio_chat(ext_url, model, messages, config, api_key=""):
    """Send chat request to LM Studio (OpenAI compatible)."""
    # Inject current date into system message
    today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y年%m月%d日 %H:%M")
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            msg["content"] = f"現在の日時: {today}\n\n{msg['content']}"
            break
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(config.get("temperature", 1.3)),
        "max_tokens": int(config.get("max_new_tokens", 4096)),
        "think": False,
        "reasoning_effort": "none",
        "store": False,
    }
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        f"{ext_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=300,
    )
    resp.raise_for_status()
    result_text = resp.json()["choices"][0]["message"].get("content", "").strip()
    result_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()
    # Strip special tokens
    result_text = re.sub(r"<\|[^>]*\|>", "", result_text).strip()
    return result_text


def _lmstudio_chat_mcp(base_url, model, messages, config, mcp_servers,
                        api_key="", store=False, previous_response_id=None):
    """LM Studio native API with MCP tool support.

    Uses /api/v1/chat endpoint (non-OpenAI format).
    base_url should be e.g. "http://x.x.x.x:1234" (without /v1).
    Returns (result_text, response_id).
    """
    # Build input text from messages (system + user)
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # VLM multipart: extract text parts only (images not supported via native API)
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(content, str):
            parts.append(content)
    input_text = "\n\n".join(parts)

    # Inject current date + tool use instruction
    today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y年%m月%d日 %H:%M")
    tool_names = ", ".join(mcp_servers)
    input_text = (
        f"<|system|>\n"
        f"現在の日時: {today}\n"
        f"You MUST call at least one tool ({tool_names}) before answering. "
        "You may call tools up to 3 times maximum. After that, give your final answer immediately. "
        "Do NOT repeat the same search query.\n"
        f"<|user|>\n{input_text}"
    )

    integrations = [{"type": "plugin", "id": s} for s in mcp_servers]
    payload = {
        "model": model,
        "input": input_text,
        "integrations": integrations,
        "temperature": min(float(config.get("temperature", 0.7)), 1.0),
        "context_length": int(config.get("max_new_tokens", 4096)),
        "store": store,
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    print(f"[MCP] payload: {json.dumps(payload, ensure_ascii=False)[:500]}")
    resp = requests.post(
        f"{base_url}/api/v1/chat",
        json=payload,
        headers=headers,
        timeout=600,
    )
    if resp.status_code != 200:
        try:
            err = resp.json().get("error", {})
            detail = err.get("message", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"LM Studio /api/v1/chat {resp.status_code}: {detail}")
    data = resp.json()
    print(f"[MCP] response keys: {list(data.keys())}")
    output = data.get("output", [])
    for item in output:
        t = item.get("type", "?")
        c = str(item.get("content", ""))[:200]
        print(f"[MCP]   output item: type={t}, content={c}")
    # Extract message content from output array
    result_parts = []
    for item in output:
        if item.get("type") == "message" and item.get("content"):
            result_parts.append(item["content"])
    result_text = "\n".join(result_parts).strip()
    result_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()
    result_text = re.sub(r"<\|[^>]*\|>", "", result_text).strip()
    response_id = data.get("id", None)
    return result_text, response_id


@app.post("/api/custom-llm")
def custom_llm_call():
    """Call diary LLM with custom prompt template. Supports VLM with image."""
    import base64 as b64mod
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    use_camera = data.get("use_camera", False)
    req_mcp_servers = list(data.get("mcp_servers", []))

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

    # Load LLM backend config
    backend_cfg = _load_llm_backend_config()
    use_lmstudio = backend_cfg.get("backend") == "lmstudio"

    # VLM mode: capture camera image
    if use_camera:
        try:
            # Capture camera snapshot
            video_dev = "/dev/video12"
            width, height = 448, 448
            v4l2 = subprocess.Popen(
                ["v4l2-ctl", "-d", video_dev,
                 "--set-fmt-video", f"width={width},height={height},pixelformat=NV12",
                 "--stream-mmap", "--stream-count=1", "--stream-to=-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "nv12",
                 "-video_size", f"{width}x{height}",
                 "-i", "-", "-frames:v", "1",
                 "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"],
                stdin=v4l2.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            v4l2.stdout.close()
            jpeg_data, _ = ffmpeg.communicate(timeout=10)
            v4l2.wait(timeout=5)
            if not jpeg_data:
                return flask_error(500, "camera capture failed")
            image_b64 = b64mod.b64encode(jpeg_data).decode()
        except Exception as e:
            return flask_error(500, f"camera capture error: {e}")

        if use_lmstudio:
            try:
                ext_url = backend_cfg["lmstudio_url"].rstrip("/")
                model = backend_cfg.get("lmstudio_model", "")
                messages = [
                    {"role": "system", "content": filled},
                    {"role": "user", "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }},
                    ]},
                ]
                api_key = backend_cfg.get("lmstudio_api_key", "")
                if req_mcp_servers:
                    base_url = re.sub(r"/v1/?$", "", ext_url)
                    result_text, _ = _lmstudio_chat_mcp(base_url, model, messages, config, req_mcp_servers, api_key)
                else:
                    result_text = _lmstudio_chat(ext_url, model, messages, config, api_key)
                return jsonify({
                    "result": result_text,
                    "prompt": filled,
                    "mode": "vlm",
                    "backend": "lmstudio",
                    "mcp": bool(req_mcp_servers),
                })
            except requests.ConnectionError:
                return flask_error(502, "LM Studio unreachable")
            except requests.Timeout:
                return flask_error(504, "LM Studio timeout")
            except Exception as e:
                return flask_error(502, f"LM Studio error: {e}")
        else:
            # Device RKLLM VLM endpoint
            try:
                resp = requests.post(
                    "http://127.0.0.1:8082/rkllm_vlm",
                    json={
                        "prompt": filled,
                        "image_base64": image_b64,
                        "max_new_tokens": int(config.get("max_new_tokens", 512)),
                    },
                    timeout=300,
                )
                if resp.status_code == 503:
                    return flask_error(503, "VLM server busy")
                result_text = resp.text.strip()
                result_text = re.sub(
                    r"<think>.*?</think>", "", result_text, flags=re.DOTALL
                ).strip()
                return jsonify({
                    "result": result_text,
                    "prompt": filled,
                    "mode": "vlm",
                })
            except requests.ConnectionError:
                return flask_error(502, "VLM server (8082) unreachable")
            except requests.Timeout:
                return flask_error(504, "VLM timeout")

    # Text-only mode
    if use_lmstudio:
        try:
            ext_url = backend_cfg["lmstudio_url"].rstrip("/")
            model = backend_cfg.get("lmstudio_model", "")
            messages = [
                {"role": "system", "content": filled},
                {"role": "user", "content": text},
            ]
            api_key = backend_cfg.get("lmstudio_api_key", "")
            print(f"[Custom LLM] mcp_servers={req_mcp_servers}")
            if req_mcp_servers:
                base_url = re.sub(r"/v1/?$", "", ext_url)
                result_text, _ = _lmstudio_chat_mcp(base_url, model, messages, config, req_mcp_servers, api_key)
            else:
                result_text = _lmstudio_chat(ext_url, model, messages, config, api_key)
            return jsonify({
                "result": result_text,
                "prompt": filled,
                "backend": "lmstudio",
                "mcp": bool(req_mcp_servers),
            })
        except requests.ConnectionError:
            return flask_error(502, "LM Studio unreachable")
        except requests.Timeout:
            return flask_error(504, "LM Studio timeout")
        except Exception as e:
            return flask_error(502, f"LM Studio error: {e}")

    # Device RKLLM text mode
    payload = {
        "task": config.get("task", "custom"),
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


def _read_amixer_control(name):
    """Read a single amixer control, return {value, min, max, percent} or None."""
    try:
        out = subprocess.check_output(
            ["amixer", "get", name], timeout=3, stderr=subprocess.DEVNULL
        ).decode(errors="ignore")
        # Parse limits: "Limits: 0 - 14" or "Limits: Capture 0 - 255"
        m_range = re.search(r'Limits:.*?(\d+)\s*-\s*(\d+)', out)
        # Parse raw value: "Mono: 14 [100%]" or "Mono: Capture 191 [75%]"
        m_raw = re.search(r'Mono:.*?(\d+)\s*\[', out)
        m_pct = re.search(r'\[(\d+)%\]', out)
        if m_raw and m_range:
            val = int(m_raw.group(1))
            vmin, vmax = int(m_range.group(1)), int(m_range.group(2))
            pct = int(m_pct.group(1)) if m_pct else (
                round((val - vmin) / (vmax - vmin) * 100) if vmax > vmin else 0)
            return {"value": val, "min": vmin, "max": vmax, "percent": pct}
        # Fallback: numid style "values=N"
        m_val = re.search(r':\s*values=(\d+)', out)
        m_range2 = re.search(r'min=(\d+),max=(\d+)', out)
        if m_val and m_range2:
            val = int(m_val.group(1))
            vmin, vmax = int(m_range2.group(1)), int(m_range2.group(2))
            pct = round((val - vmin) / (vmax - vmin) * 100) if vmax > vmin else 0
            return {"value": val, "min": vmin, "max": vmax, "percent": pct}
        if m_pct:
            return {"value": int(m_pct.group(1)), "percent": int(m_pct.group(1))}
    except Exception:
        pass
    return None


def _read_audio():
    """Read audio/speaker info from ALSA."""
    info = {"cards": [], "volume": None, "mic": {}}
    # Sound cards
    cards_str = _read_file("/proc/asound/cards")
    if cards_str:
        for m in re.finditer(r'^\s*(\d+)\s+\[(\w+)\s*\]:\s*(.+)$', cards_str, re.MULTILINE):
            info["cards"].append({
                "id": int(m.group(1)),
                "name": m.group(2),
                "description": m.group(3).strip(),
            })
    # Master volume
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
    # Microphone controls
    mic = {}
    for name in ("ADCL", "ADCR", "ADCL PGA", "ADCR PGA",
                 "Main Mic", "Headset Mic"):
        ctrl = _read_amixer_control(name)
        if ctrl is not None:
            mic[name] = ctrl
    # Switches (on/off)
    for name in ("Main Mic", "Headset Mic"):
        try:
            out = subprocess.check_output(
                ["amixer", "get", name], timeout=3, stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            m = re.search(r'\[(on|off)\]', out)
            if m:
                mic.setdefault(name, {})["switch"] = m.group(1) == "on"
        except Exception:
            pass
    if mic:
        info["mic"] = mic
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


@app.post("/api/sensors/mic")
def set_mic():
    """Set microphone gain via amixer."""
    data = request.get_json(force=True)
    control = data.get("control")
    value = data.get("value")
    if not control or value is None:
        return flask_error(400, "control and value are required")
    allowed = {"ADCL", "ADCR", "ADCL PGA", "ADCR PGA",
               "Main Mic", "Headset Mic"}
    if control not in allowed:
        return flask_error(400, f"unknown control: {control}")
    value = int(value)
    try:
        # For switch controls (Main Mic, Headset Mic), treat as on/off toggle
        if control in ("Main Mic", "Headset Mic"):
            state = "on" if value else "off"
            subprocess.check_output(
                ["amixer", "set", control, state],
                timeout=5, stderr=subprocess.DEVNULL,
            )
            return jsonify({"status": "ok", "control": control, "switch": state})
        # For gain controls, set raw value
        subprocess.check_output(
            ["amixer", "set", control, str(value)],
            timeout=5, stderr=subprocess.DEVNULL,
        )
        return jsonify({"status": "ok", "control": control, "value": value})
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


# --- TTS (edge-tts) ---

import asyncio

TTS_VOICES = [
    "ja-JP-NanamiNeural",
    "ja-JP-KeitaNeural",
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-BrianNeural",
    "en-US-AvaNeural",
    "en-US-AndrewNeural",
    "en-US-EmmaNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-US-MichelleNeural",
    "en-US-RogerNeural",
]

TTS_CONFIG_PATH = "/data/devtools/tts_config.json"


def _load_tts_config():
    try:
        with open(TTS_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_tts_config(cfg):
    with open(TTS_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


@app.get("/api/tts/options")
def tts_options():
    cfg = _load_tts_config()
    return jsonify({"voices": TTS_VOICES, "config": cfg})


@app.post("/api/tts/config")
def tts_config_save():
    data = request.get_json(force=True)
    cfg = {
        "voice": data.get("voice", "ja-JP-NanamiNeural"),
        "rate": data.get("rate", "+0%"),
        "pitch": data.get("pitch", "+0Hz"),
        "playback": data.get("playback", "browser"),
    }
    _save_tts_config(cfg)
    return jsonify({"status": "ok"})


@app.post("/api/tts")
def tts_synthesize():
    import edge_tts

    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return flask_error(400, "text is required")
    voice = data.get("voice", "ja-JP-NanamiNeural")
    rate = data.get("rate", "+0%")
    pitch = data.get("pitch", "+0Hz")

    mp3_path = "/tmp/tts_edge.mp3"
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        asyncio.run(comm.save(mp3_path))
    except Exception as e:
        return flask_error(502, f"edge-tts failed: {e}")

    if data.get("browser"):
        with open(mp3_path, "rb") as f:
            audio_data = f.read()
        return Response(audio_data, mimetype="audio/mpeg")

    subprocess.Popen(
        ["mpg123", "-q", "-a", "tts_out", mp3_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return jsonify({"status": "ok"})


# --- Device control ---

@app.post("/api/device/reboot")
def device_reboot():
    """Reboot the device."""
    subprocess.Popen(["reboot"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify({"status": "ok", "message": "rebooting"})


@app.post("/api/service/restart")
def service_restart():
    """Restart a specific systemd service."""
    data = request.get_json(force=True)
    service = data.get("service", "")
    allowed = ["kata-devtools", "llm_action", "llm_diary", "llm_route", "master", "pet_voice"]
    if service not in allowed:
        return flask_error(400, f"service not allowed: {service}")
    try:
        rc = os.system(f"systemctl restart {service} 2>/dev/null")
        return jsonify({"status": "ok" if rc == 0 else "error", "service": service, "rc": rc})
    except Exception as e:
        return flask_error(500, str(e))


# --- Auto-talk (server-side loop) ---

AUTO_TALK_CONFIG_PATH = "/data/devtools/auto_talk_config.json"

_auto_talk_lock = threading.Lock()
_auto_talk_state = {
    "running": False,
    "thread": None,
    "stop_event": None,
    "last_result": None,
    "last_time": None,
    "count": 0,
}


def _auto_talk_loop(text, interval, stop_event):
    """Background thread: periodically capture camera + call VLM."""
    import base64 as b64mod

    while not stop_event.is_set():
        result_entry = {"time": time.strftime("%H:%M:%S"), "text": text}
        try:
            # Read template
            template = ""
            if os.path.isfile(CUSTOM_PROMPT_PATH):
                with open(CUSTOM_PROMPT_PATH, "r", encoding="utf-8") as f:
                    template = f.read().strip()
            filled = template.format(text=text) if template else text

            # Read config
            config = {"temperature": 1.3, "max_new_tokens": 4096}
            if os.path.isfile(CUSTOM_LLM_CONFIG_PATH):
                try:
                    with open(CUSTOM_LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
                        config.update(json.loads(f.read()))
                except Exception:
                    pass

            # Capture camera
            video_dev = "/dev/video12"
            w, h = 448, 448
            v4l2 = subprocess.Popen(
                ["v4l2-ctl", "-d", video_dev,
                 "--set-fmt-video", f"width={w},height={h},pixelformat=NV12",
                 "--stream-mmap", "--stream-count=1", "--stream-to=-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "nv12",
                 "-video_size", f"{w}x{h}",
                 "-i", "-", "-frames:v", "1",
                 "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"],
                stdin=v4l2.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            v4l2.stdout.close()
            jpeg_data, _ = ffmpeg.communicate(timeout=10)
            v4l2.wait(timeout=5)
            if not jpeg_data:
                result_entry["error"] = "camera capture failed"
            else:
                image_b64 = b64mod.b64encode(jpeg_data).decode()

                # Check backend
                backend_cfg = _load_llm_backend_config()
                if backend_cfg.get("backend") == "lmstudio":
                    ext_url = backend_cfg["lmstudio_url"].rstrip("/")
                    messages = [
                        {"role": "system", "content": filled},
                        {"role": "user", "content": [
                            {"type": "text", "text": text},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }},
                        ]},
                    ]
                    result_text = _lmstudio_chat(
                        ext_url, backend_cfg.get("lmstudio_model", ""),
                        messages, config, backend_cfg.get("lmstudio_api_key", ""))
                    result_entry["result"] = result_text
                    result_entry["mode"] = "vlm"
                else:
                    resp = requests.post(
                        "http://127.0.0.1:8082/rkllm_vlm",
                        json={
                            "prompt": filled,
                            "image_base64": image_b64,
                            "max_new_tokens": int(config.get("max_new_tokens", 512)),
                        },
                        timeout=300,
                    )
                    if resp.status_code == 503:
                        result_entry["error"] = "VLM server busy"
                    else:
                        result_text = resp.text.strip()
                        result_text = re.sub(
                            r"<think>.*?</think>", "", result_text,
                            flags=re.DOTALL).strip()
                        result_entry["result"] = result_text
                        result_entry["mode"] = "vlm"
        except Exception as e:
            result_entry["error"] = str(e)

        with _auto_talk_lock:
            _auto_talk_state["last_result"] = result_entry
            _auto_talk_state["last_time"] = time.time()
            _auto_talk_state["count"] += 1

        # Wait for interval (check stop_event every second)
        for _ in range(interval):
            if stop_event.is_set():
                break
            time.sleep(1)



@app.get("/api/auto-talk/config")
def auto_talk_config_get():
    try:
        with open(AUTO_TALK_CONFIG_PATH, "r") as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({"text": "今何が見える？", "interval": 60})


@app.post("/api/auto-talk/config")
def auto_talk_config_save():
    data = request.get_json(force=True)
    cfg = {
        "text": data.get("text", "今何が見える？"),
        "interval": int(data.get("interval", 60)),
    }
    with open(AUTO_TALK_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok"})


@app.post("/api/auto-talk/start")
def auto_talk_start():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    interval = int(data.get("interval", 60))
    if not text:
        return flask_error(400, "text is required")
    with _auto_talk_lock:
        if _auto_talk_state["running"]:
            return jsonify({"status": "already_running"})
        stop_event = threading.Event()
        t = threading.Thread(
            target=_auto_talk_loop, args=(text, interval, stop_event),
            daemon=True)
        _auto_talk_state.update({
            "running": True,
            "thread": t,
            "stop_event": stop_event,
            "last_result": None,
            "last_time": None,
            "count": 0,
        })
        t.start()
    return jsonify({"status": "started"})


@app.post("/api/auto-talk/stop")
def auto_talk_stop():
    with _auto_talk_lock:
        if not _auto_talk_state["running"]:
            return jsonify({"status": "not_running"})
        _auto_talk_state["stop_event"].set()
        _auto_talk_state["running"] = False
    return jsonify({"status": "stopped"})


@app.get("/api/auto-talk/status")
def auto_talk_status():
    with _auto_talk_lock:
        return jsonify({
            "running": _auto_talk_state["running"],
            "count": _auto_talk_state["count"],
            "last_result": _auto_talk_state["last_result"],
            "last_time": _auto_talk_state["last_time"],
        })


# --- Conversation mode (wake word → Custom LLM chat) ---

CONVERSATION_CONFIG_PATH = "/data/devtools/conversation_config.json"

_conversation_lock = threading.Lock()
_conversation_state = {
    "enabled": False,
    "phase": "disabled",  # disabled | waiting | listening | processing | speaking
    "thread": None,
    "stop_event": None,
    "auto_talk_was_running": False,
    "timeout": 30,
    "conversation_log": [],  # [{role, text, time}, ...] max 50
    "turn_count": 0,
    "last_response_id": None,  # LM Studio response ID for conversation continuation
    "last_llm_time": 0,        # timestamp of last LLM response
}


# ZMQ SUB constants (ctypes)
ZMQ_SUB = 2
ZMQ_SUBSCRIBE = 6
ZMQ_RCVTIMEO = 27
ZMQ_RCVMORE = 13
ZMQ_DONTWAIT = 1


def _zmq_recv_multipart(zmq_lib, socket):
    """Receive a multipart ZMQ message. Returns list of bytes frames or None on timeout."""
    frames = []
    while True:
        buf = ctypes.create_string_buffer(4096)
        rc = zmq_lib.zmq_recv(socket, buf, 4096, 0)
        if rc < 0:
            return None  # timeout or error
        frames.append(buf.raw[:rc])
        # Check ZMQ_RCVMORE
        more = ctypes.c_int(0)
        more_size = ctypes.c_size_t(ctypes.sizeof(more))
        zmq_lib.zmq_getsockopt(socket, ZMQ_RCVMORE, ctypes.byref(more), ctypes.byref(more_size))
        if not more.value:
            break
    return frames


def _zmq_flush(zmq_lib, socket, duration=3.0):
    """Drain all pending/incoming messages for `duration` seconds after TTS.

    ASR pipeline has latency — echo messages arrive after TTS finishes.
    Keep draining for a few seconds to catch them all.
    """
    buf = ctypes.create_string_buffer(4096)
    flushed = 0
    deadline = time.time() + duration
    while time.time() < deadline:
        rc = zmq_lib.zmq_recv(socket, buf, 4096, ZMQ_DONTWAIT)
        if rc < 0:
            time.sleep(0.05)  # 50ms sleep to avoid busy-wait
            continue
        flushed += 1
        more = ctypes.c_int(0)
        ms = ctypes.c_size_t(ctypes.sizeof(more))
        zmq_lib.zmq_getsockopt(socket, ZMQ_RCVMORE, ctypes.byref(more), ctypes.byref(ms))
        while more.value:
            zmq_lib.zmq_recv(socket, buf, 4096, ZMQ_DONTWAIT)
            zmq_lib.zmq_getsockopt(socket, ZMQ_RCVMORE, ctypes.byref(more), ctypes.byref(ms))
    if flushed:
        print(f"[Conversation] flushed {flushed} echo messages ({duration}s window)")


def _msgpack_decode_str(data):
    """Decode a msgpack fixstr/str8/str16 from bytes."""
    if not data:
        return ""
    b0 = data[0]
    if (b0 & 0xE0) == 0xA0:  # fixstr
        length = b0 & 0x1F
        return data[1:1 + length].decode("utf-8", errors="replace")
    elif b0 == 0xD9:  # str8
        length = data[1]
        return data[2:2 + length].decode("utf-8", errors="replace")
    elif b0 == 0xDA:  # str16
        length = struct.unpack(">H", data[1:3])[0]
        return data[3:3 + length].decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")


_tts_process = None  # mpg123 subprocess, set during TTS playback


def _tts_speak_on_device(text):
    """Synthesize with edge-tts and play on device via mpg123 (blocking)."""
    global _tts_process
    import edge_tts

    cfg = _load_tts_config()
    voice = cfg.get("voice", "ja-JP-NanamiNeural")
    rate = cfg.get("rate", "+0%")
    pitch = cfg.get("pitch", "+0Hz")

    mp3_path = "/tmp/tts_conversation.mp3"
    try:
        loop = asyncio.new_event_loop()
        comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        loop.run_until_complete(comm.save(mp3_path))
        loop.close()
    except Exception as e:
        print(f"[Conversation] TTS synthesis error: {e}")
        return

    try:
        _tts_process = subprocess.Popen(
            ["mpg123", "-q", "-a", "tts_out", mp3_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _tts_process.wait(timeout=60)
    except Exception as e:
        print(f"[Conversation] mpg123 error: {e}")
    finally:
        _tts_process = None


def _tts_speak_on_device_start(text):
    """Synthesize TTS and start mpg123 playback (non-blocking).

    After calling this, use _tts_wait_with_zmq() to wait for completion
    while monitoring ZMQ for wake word interrupts.
    """
    global _tts_process
    import edge_tts

    cfg = _load_tts_config()
    voice = cfg.get("voice", "ja-JP-NanamiNeural")
    rate = cfg.get("rate", "+0%")
    pitch = cfg.get("pitch", "+0Hz")

    mp3_path = "/tmp/tts_conversation.mp3"
    try:
        loop = asyncio.new_event_loop()
        comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        loop.run_until_complete(comm.save(mp3_path))
        loop.close()
    except Exception as e:
        print(f"[Conversation] TTS synthesis error: {e}")
        return

    try:
        _tts_process = subprocess.Popen(
            ["mpg123", "-q", "-a", "tts_out", mp3_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[Conversation] mpg123 start error: {e}")
        _tts_process = None


def _tts_wait_with_zmq(zmq_lib, sock, wake_texts):
    """Wait for TTS to finish while checking ZMQ for wake word.

    Returns extracted question text (str) if wake word interrupted,
    or None if TTS completed normally.
    """
    global _tts_process
    while _tts_process and _tts_process.poll() is None:
        frames = _zmq_recv_multipart(zmq_lib, sock)
        if frames is None:
            continue
        if len(frames) < 2:
            continue
        payload_str = _msgpack_decode_str(frames[1])
        try:
            vad_data = json.loads(payload_str)
        except (json.JSONDecodeError, ValueError):
            continue
        if vad_data.get("is_wake_word"):
            # ウェイクワード検出 → TTS 中断
            _tts_process.kill()
            _tts_process = None
            text = vad_data.get("text", "").strip()
            extra = text
            for wt in wake_texts:
                extra = extra.replace(wt, "")
            extra = extra.strip()
            print(f"[Conversation] wake interrupt during TTS, extra='{extra}'")
            return extra if extra else ""
    _tts_process = None
    return None


def _tts_cancel():
    """Kill mpg123 if playing. Returns True if cancelled."""
    global _tts_process
    if _tts_process and _tts_process.poll() is None:
        _tts_process.kill()
        _tts_process = None
        print("[Conversation] TTS cancelled by wake word")
        return True
    return False


def _conv_log_append(role, text):
    """Append to conversation log (max 50 entries)."""
    entry = {"role": role, "text": text, "time": time.strftime("%H:%M:%S")}
    with _conversation_lock:
        _conversation_state["conversation_log"].append(entry)
        if len(_conversation_state["conversation_log"]) > 50:
            _conversation_state["conversation_log"] = _conversation_state["conversation_log"][-50:]


def _pause_auto_talk_for_conversation():
    """Pause auto-talk mode if running, record state for later resume."""
    with _auto_talk_lock:
        if _auto_talk_state["running"]:
            _auto_talk_state["stop_event"].set()
            _auto_talk_state["running"] = False
            with _conversation_lock:
                _conversation_state["auto_talk_was_running"] = True
            print("[Conversation] auto-talk paused")
        else:
            with _conversation_lock:
                _conversation_state["auto_talk_was_running"] = False


def _resume_auto_talk_after_conversation():
    """Resume auto-talk mode if it was running before conversation."""
    with _conversation_lock:
        was_running = _conversation_state["auto_talk_was_running"]
        _conversation_state["auto_talk_was_running"] = False
    if not was_running:
        return
    # Load config and restart
    try:
        with open(AUTO_TALK_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {"text": "今何が見える？", "interval": 60}
    text = cfg.get("text", "今何が見える？")
    interval = int(cfg.get("interval", 60))
    with _auto_talk_lock:
        if _auto_talk_state["running"]:
            return  # already running
        stop_event = threading.Event()
        t = threading.Thread(
            target=_auto_talk_loop, args=(text, interval, stop_event),
            daemon=True)
        _auto_talk_state.update({
            "running": True,
            "thread": t,
            "stop_event": stop_event,
            "last_result": None,
            "last_time": None,
            "count": 0,
        })
        t.start()
    print("[Conversation] auto-talk resumed")


def _conversation_call_llm(text):
    """Camera capture + Custom LLM call. Returns response text or error string."""
    import base64 as b64mod

    # Read template
    template = ""
    if os.path.isfile(CUSTOM_PROMPT_PATH):
        try:
            with open(CUSTOM_PROMPT_PATH, "r", encoding="utf-8") as f:
                template = f.read().strip()
        except Exception:
            pass
    filled = template.format(text=text) if template else text

    # Read config
    config = {"temperature": 1.3, "max_new_tokens": 4096}
    if os.path.isfile(CUSTOM_LLM_CONFIG_PATH):
        try:
            with open(CUSTOM_LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
                config.update(json.loads(f.read()))
        except Exception:
            pass

    # Capture camera
    video_dev = "/dev/video12"
    w, h = 448, 448
    try:
        v4l2 = subprocess.Popen(
            ["v4l2-ctl", "-d", video_dev,
             "--set-fmt-video", f"width={w},height={h},pixelformat=NV12",
             "--stream-mmap", "--stream-count=1", "--stream-to=-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "nv12",
             "-video_size", f"{w}x{h}",
             "-i", "-", "-frames:v", "1",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"],
            stdin=v4l2.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        v4l2.stdout.close()
        jpeg_data, _ = ffmpeg.communicate(timeout=10)
        v4l2.wait(timeout=5)
    except Exception as e:
        return f"[camera error: {e}]"
    if not jpeg_data:
        return "[camera capture failed]"
    image_b64 = b64mod.b64encode(jpeg_data).decode()

    # LLM call
    backend_cfg = _load_llm_backend_config()
    if backend_cfg.get("backend") == "lmstudio":
        try:
            ext_url = backend_cfg["lmstudio_url"].rstrip("/")
            model = backend_cfg.get("lmstudio_model", "")
            messages = [
                {"role": "system", "content": filled},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }},
                ]},
            ]
            # MCP tool support: use active servers for conversation mode
            conv_cfg = _load_conversation_config()
            active_servers = conv_cfg.get("conv_active_servers", [])
            api_key = backend_cfg.get("lmstudio_api_key", "")
            # Conversation continuation: reuse response_id if within 60 seconds
            now = time.time()
            prev_id = None
            use_store = False
            if _conversation_state["last_response_id"] and (now - _conversation_state["last_llm_time"]) < 60:
                prev_id = _conversation_state["last_response_id"]
                use_store = True
                print(f"[Conv] Continuing conversation (prev_id={prev_id})")
            else:
                print("[Conv] Starting fresh conversation")
            if active_servers:
                base_url = re.sub(r"/v1/?$", "", ext_url)
                result_text, resp_id = _lmstudio_chat_mcp(
                    base_url, model, messages, config, active_servers, api_key,
                    store=use_store, previous_response_id=prev_id)
                _conversation_state["last_response_id"] = resp_id
                _conversation_state["last_llm_time"] = time.time()
                return result_text
            result_text = _lmstudio_chat(ext_url, model, messages, config, api_key)
            _conversation_state["last_response_id"] = None
            _conversation_state["last_llm_time"] = time.time()
            return result_text
        except Exception as e:
            return f"[LM Studio error: {e}]"
    else:
        try:
            resp = requests.post(
                "http://127.0.0.1:8082/rkllm_vlm",
                json={
                    "prompt": filled,
                    "image_base64": image_b64,
                    "max_new_tokens": int(config.get("max_new_tokens", 512)),
                },
                timeout=300,
            )
            if resp.status_code == 503:
                return "[VLM server busy]"
            result_text = resp.text.strip()
            result_text = re.sub(
                r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()
            return result_text
        except Exception as e:
            return f"[VLM error: {e}]"


def _conversation_thread(stop_event):
    """Main conversation thread: ZMQ SUB → wake word → LLM → TTS loop."""
    zmq_lib = ctypes.CDLL("libzmq.so.5")
    zmq_lib.zmq_ctx_new.restype = ctypes.c_void_p
    zmq_lib.zmq_socket.restype = ctypes.c_void_p
    zmq_lib.zmq_socket.argtypes = [ctypes.c_void_p, ctypes.c_int]
    zmq_lib.zmq_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    zmq_lib.zmq_recv.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
    zmq_lib.zmq_recv.restype = ctypes.c_int
    zmq_lib.zmq_setsockopt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    zmq_lib.zmq_getsockopt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    zmq_lib.zmq_close.argtypes = [ctypes.c_void_p]
    zmq_lib.zmq_ctx_destroy.argtypes = [ctypes.c_void_p]

    ctx = zmq_lib.zmq_ctx_new()
    sock = zmq_lib.zmq_socket(ctx, ZMQ_SUB)

    # Subscribe to /voice/vad_data
    sub_filter = b"#/voice/vad_data"
    zmq_lib.zmq_setsockopt(sock, ZMQ_SUBSCRIBE, sub_filter, len(sub_filter))

    # Set receive timeout 1000ms so we can check stop_event
    timeout_val = ctypes.c_int(1000)
    zmq_lib.zmq_setsockopt(sock, ZMQ_RCVTIMEO, ctypes.byref(timeout_val), ctypes.sizeof(timeout_val))

    zmq_lib.zmq_connect(sock, b"tcp://127.0.0.1:5559")

    with _conversation_lock:
        _conversation_state["phase"] = "waiting"
        _conversation_state["conversation_log"] = []
        _conversation_state["turn_count"] = 0
        _conversation_state["last_response_id"] = None
        _conversation_state["last_llm_time"] = 0

    print("[Conversation] ZMQ SUB started, waiting for wake word...")

    last_utterance_time = 0
    last_response_time = 0  # TTS再生完了時刻 (ウェイクワード省略判定用)
    WAKE_SKIP_WINDOW = 10   # 秒以内ならウェイクワード不要

    # ウェイクワードテキスト一覧を読み込み（テキスト除去用）
    _wake_texts = set()
    try:
        src = KWS_PERSIST_FILE if os.path.exists(KWS_PERSIST_FILE) else KWS_BINARY_FILE
        with open(src, "r") as f:
            for line in f:
                t = _detokenize_keyword(line.strip())
                if t:
                    _wake_texts.add(t)
        print(f"[Conversation] wake word texts: {_wake_texts}")
    except Exception:
        pass

    try:
        while not stop_event.is_set():
            frames = _zmq_recv_multipart(zmq_lib, sock)
            if frames is None:
                # Timeout — check if conversation should end
                with _conversation_lock:
                    phase = _conversation_state["phase"]
                    timeout = _conversation_state["timeout"]
                if phase == "listening" and last_utterance_time > 0:
                    elapsed = time.time() - last_utterance_time
                    if elapsed > timeout:
                        _conv_log_append("system", "タイムアウト — 会話終了")
                        with _conversation_lock:
                            _conversation_state["phase"] = "waiting"
                            _conversation_state["turn_count"] = 0
                        _resume_auto_talk_after_conversation()
                        print("[Conversation] timeout, back to waiting")
                        last_utterance_time = 0
                continue

            # Parse vad_data: frames[0]=topic, frames[1]=msgpack payload
            if len(frames) < 2:
                continue

            payload_str = _msgpack_decode_str(frames[1])
            try:
                vad_data = json.loads(payload_str)
            except (json.JSONDecodeError, ValueError):
                continue

            is_wake = vad_data.get("is_wake_word", False)
            text = vad_data.get("text", "").strip()

            with _conversation_lock:
                phase = _conversation_state["phase"]

            # Speaking: ウェイクワードのみ受付 → TTS中断して新会話開始
            if phase == "speaking":
                if is_wake:
                    _tts_cancel()
                    extra_text = text
                    if extra_text:
                        for wt in _wake_texts:
                            extra_text = extra_text.replace(wt, "")
                        extra_text = extra_text.strip()
                    _zmq_flush(zmq_lib, sock, duration=0)
                    with _conversation_lock:
                        _conversation_state["phase"] = "listening"
                        _conversation_state["turn_count"] = 0
                    _conv_log_append("system", "ウェイクワード割り込み — TTS中断")
                    print(f"[Conversation] wake interrupt during TTS, extra='{extra_text}'")
                    last_utterance_time = time.time()
                continue

            if phase == "waiting":
                # ウェイクワード or 直前の応答から10秒以内の発話で会話開始
                wake_skip = (last_response_time > 0
                             and (time.time() - last_response_time) < WAKE_SKIP_WINDOW
                             and text and not is_wake)
                if is_wake or wake_skip:
                    # ウェイクワード部分をテキストから除去して質問部分を抽出
                    extra_text = text
                    if is_wake and extra_text:
                        for wt in _wake_texts:
                            extra_text = extra_text.replace(wt, "")
                        extra_text = extra_text.strip()
                        print(f"[Conversation] wake text='{text}', extra='{extra_text}'")

                    _pause_auto_talk_for_conversation()
                    with _conversation_lock:
                        _conversation_state["phase"] = "listening"
                        _conversation_state["turn_count"] = 0

                    # 質問部分があればそのまま処理（ウェイクワード+質問を一息で言った場合）
                    user_text = extra_text if is_wake else text
                    if user_text:
                        _conv_log_append("system", "ウェイクワード検出" if is_wake else "会話継続 (ウェイクワード省略)")
                        last_utterance_time = time.time()
                        _conv_log_append("user", user_text)
                        print(f"[Conversation] user: {user_text}")
                        with _conversation_lock:
                            _conversation_state["phase"] = "processing"
                        response = _conversation_call_llm(user_text)
                        _conv_log_append("robot", response)
                        print(f"[Conversation] robot: {response}")
                        with _conversation_lock:
                            _conversation_state["turn_count"] += 1
                            _conversation_state["phase"] = "speaking"
                        _tts_speak_on_device(response)
                        _zmq_flush(zmq_lib, sock)  # にこの声のエコーを破棄
                        last_response_time = time.time()
                        with _conversation_lock:
                            _conversation_state["phase"] = "listening"
                        last_utterance_time = time.time()
                        continue
                    else:
                        _conv_log_append("system", "ウェイクワード検出")
                        last_utterance_time = time.time()
                        print("[Conversation] wake word detected, listening...")

            elif phase == "listening":
                if is_wake:
                    # ウェイクワード → 会話リセット
                    extra_text = text
                    if extra_text:
                        for wt in _wake_texts:
                            extra_text = extra_text.replace(wt, "")
                        extra_text = extra_text.strip()
                    with _conversation_lock:
                        _conversation_state["turn_count"] = 0
                    _conv_log_append("system", "ウェイクワード — 会話リセット")
                    print("[Conversation] wake word during listening — reset")
                    last_utterance_time = time.time()
                    if extra_text:
                        _conv_log_append("user", extra_text)
                        print(f"[Conversation] user: {extra_text}")
                        with _conversation_lock:
                            _conversation_state["phase"] = "processing"
                        response = _conversation_call_llm(extra_text)
                        _conv_log_append("robot", response)
                        print(f"[Conversation] robot: {response}")
                        with _conversation_lock:
                            _conversation_state["turn_count"] += 1
                            _conversation_state["phase"] = "speaking"
                        _tts_speak_on_device(response)
                        _zmq_flush(zmq_lib, sock)
                        last_response_time = time.time()
                        with _conversation_lock:
                            _conversation_state["phase"] = "listening"
                        last_utterance_time = time.time()
                    continue
                if not text:
                    continue
                # ウェイクワードだけのテキストは無視
                cleaned = text
                for wt in _wake_texts:
                    cleaned = cleaned.replace(wt, "")
                if not cleaned.strip():
                    print(f"[Conversation] ignoring wake-word-only text: '{text}'")
                    continue
                text = cleaned.strip() if cleaned.strip() != text else text
                last_utterance_time = time.time()
                _conv_log_append("user", text)
                print(f"[Conversation] user: {text}")

                # Process: camera + LLM
                with _conversation_lock:
                    _conversation_state["phase"] = "processing"

                response = _conversation_call_llm(text)
                _conv_log_append("robot", response)
                print(f"[Conversation] robot: {response}")

                with _conversation_lock:
                    _conversation_state["turn_count"] += 1

                # TTS playback (non-blocking: ZMQ監視でウェイクワード割り込み可能)
                with _conversation_lock:
                    _conversation_state["phase"] = "speaking"

                _tts_speak_on_device_start(response)
                # mpg123 完了まで待ちつつ ZMQ を監視
                while _tts_process and _tts_process.poll() is None:
                    frames = _zmq_recv_multipart(zmq_lib, sock)
                    if frames and len(frames) >= 2:
                        ps = _msgpack_decode_str(frames[1])
                        try:
                            vd = json.loads(ps)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if vd.get("is_wake_word"):
                            _tts_cancel()
                            et = vd.get("text", "").strip()
                            for wt in _wake_texts:
                                et = et.replace(wt, "")
                            et = et.strip()
                            _zmq_flush(zmq_lib, sock, duration=0)
                            with _conversation_lock:
                                _conversation_state["phase"] = "listening"
                                _conversation_state["turn_count"] = 0
                            _conv_log_append("system", "ウェイクワード割り込み — TTS中断")
                            print(f"[Conversation] wake interrupt during TTS, extra='{et}'")
                            last_utterance_time = time.time()
                            break
                else:
                    # TTS 正常完了
                    _zmq_flush(zmq_lib, sock)
                    last_response_time = time.time()
                    with _conversation_lock:
                        _conversation_state["phase"] = "listening"
                    last_utterance_time = time.time()

    except Exception as e:
        print(f"[Conversation] thread error: {e}")
    finally:
        zmq_lib.zmq_close(sock)
        zmq_lib.zmq_ctx_destroy(ctx)
        with _conversation_lock:
            _conversation_state["phase"] = "disabled"
            _conversation_state["enabled"] = False
        print("[Conversation] thread stopped")


def _load_conversation_config():
    try:
        with open(CONVERSATION_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"timeout": 30}


def _save_conversation_config(cfg):
    os.makedirs(os.path.dirname(CONVERSATION_CONFIG_PATH), exist_ok=True)
    with open(CONVERSATION_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


@app.get("/api/conversation/config")
def conversation_config_get():
    return jsonify(_load_conversation_config())


@app.post("/api/conversation/config")
def conversation_config_save():
    data = request.get_json(force=True)
    cfg = {
        "timeout": int(data.get("timeout", 30)),
        "mcp_servers": list(data.get("mcp_servers", [])),
        "conv_active_servers": list(data.get("conv_active_servers", [])),
    }
    _save_conversation_config(cfg)
    with _conversation_lock:
        _conversation_state["timeout"] = cfg["timeout"]
    return jsonify({"status": "ok"})


@app.post("/api/conversation/enable")
def conversation_enable():
    with _conversation_lock:
        if _conversation_state["enabled"]:
            return jsonify({"status": "already_enabled"})
        cfg = _load_conversation_config()
        _conversation_state["timeout"] = cfg.get("timeout", 30)
        stop_event = threading.Event()
        t = threading.Thread(
            target=_conversation_thread, args=(stop_event,),
            daemon=True)
        _conversation_state.update({
            "enabled": True,
            "thread": t,
            "stop_event": stop_event,
        })
        t.start()
    return jsonify({"status": "enabled"})


@app.post("/api/conversation/disable")
def conversation_disable():
    with _conversation_lock:
        if not _conversation_state["enabled"]:
            return jsonify({"status": "not_enabled"})
        _conversation_state["stop_event"].set()
        _conversation_state["enabled"] = False
        _conversation_state["phase"] = "disabled"
    return jsonify({"status": "disabled"})


@app.get("/api/conversation/status")
def conversation_status():
    conv_cfg = _load_conversation_config()
    mcp_active = bool(conv_cfg.get("conv_active_servers"))
    with _conversation_lock:
        return jsonify({
            "enabled": _conversation_state["enabled"],
            "phase": _conversation_state["phase"],
            "turn_count": _conversation_state["turn_count"],
            "conversation_log": _conversation_state["conversation_log"],
            "auto_talk_was_running": _conversation_state["auto_talk_was_running"],
            "mcp_active": mcp_active,
        })


# --- Wake words ---

KWS_DIR = "/opt/wlab/sweepbot/share/ai_brain/model/voice/kws"
KWS_BINARY_FILE = os.path.join(KWS_DIR, "keywords.txt")  # pet_voice reads this
KWS_PERSIST_FILE = "/data/devtools/keywords.txt"           # survives reboot
KWS_TOKENS_FILE = os.path.join(KWS_DIR, "tokens.txt")

def _sync_keywords_on_boot():
    """On startup, restore persistent keywords to the path pet_voice reads."""
    if os.path.exists(KWS_PERSIST_FILE):
        try:
            import shutil
            shutil.copy2(KWS_PERSIST_FILE, KWS_BINARY_FILE)
        except Exception as e:
            print(f"[KWS] sync failed: {e}")
    elif os.path.exists(KWS_BINARY_FILE):
        # First run: seed persistent copy from binary's file
        try:
            import shutil
            shutil.copy2(KWS_BINARY_FILE, KWS_PERSIST_FILE)
        except Exception:
            pass


_sync_keywords_on_boot()

_bpe_tokens = None  # lazy-loaded


def _load_bpe_tokens():
    """Load BPE token vocabulary: token -> id mapping."""
    global _bpe_tokens
    if _bpe_tokens is not None:
        return _bpe_tokens
    _bpe_tokens = {}
    try:
        with open(KWS_TOKENS_FILE, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    _bpe_tokens[parts[0]] = int(parts[1])
    except Exception:
        pass
    return _bpe_tokens


def _tokenize_keyword(text):
    """Convert plain English text (e.g. 'HELLO KATA') to BPE token sequence.

    Uses greedy longest-match from the token vocabulary.
    Words are prefixed with ▁ (U+2581) as per sentencepiece convention.
    """
    tokens = _load_bpe_tokens()
    if not tokens:
        return None
    result = []
    words = text.strip().upper().split()
    for i, word in enumerate(words):
        remaining = "\u2581" + word  # ▁ prefix
        while remaining:
            matched = None
            for length in range(len(remaining), 0, -1):
                candidate = remaining[:length]
                if candidate in tokens:
                    matched = candidate
                    break
            if matched:
                result.append(matched)
                remaining = remaining[len(matched):]
            else:
                # Single char fallback
                result.append(remaining[0])
                remaining = remaining[1:]
    return " ".join(result)


def _detokenize_keyword(token_line):
    """Convert BPE token sequence back to readable text.

    e.g. '▁HE LL O ▁K AT A' -> 'HELLO KATA'
    """
    text = token_line.replace(" ", "").replace("\u2581", " ").strip()
    return text


@app.get("/api/wakewords")
def get_wakewords():
    # Read from persistent copy (canonical), fallback to binary path
    src = KWS_PERSIST_FILE if os.path.exists(KWS_PERSIST_FILE) else KWS_BINARY_FILE
    try:
        with open(src, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        keywords = []
        for line in lines:
            keywords.append({
                "tokens": line,
                "text": _detokenize_keyword(line),
            })
        return jsonify({"keywords": keywords})
    except FileNotFoundError:
        return jsonify({"keywords": [], "error": "file not found"})
    except Exception as e:
        return jsonify({"keywords": [], "error": str(e)})


@app.post("/api/wakewords")
def save_wakewords():
    data = request.get_json(force=True)
    keywords = data.get("keywords", [])
    restart = data.get("restart", False)
    try:
        lines = []
        for kw in keywords:
            if kw.get("tokens"):
                lines.append(kw["tokens"])
            elif kw.get("text"):
                tok = _tokenize_keyword(kw["text"])
                if tok:
                    lines.append(tok)
        content = "\n".join(lines) + "\n"
        # Write to persistent location (/data/) + binary's path
        with open(KWS_PERSIST_FILE, "w") as f:
            f.write(content)
        with open(KWS_BINARY_FILE, "w") as f:
            f.write(content)
        result = {"status": "saved", "count": len(lines)}
        if restart:
            subprocess.run(
                ["systemctl", "restart", "pet_voice"],
                capture_output=True, timeout=15,
            )
            result["restarted"] = "pet_voice"
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.post("/api/wakewords/tokenize")
def tokenize_keyword():
    """Preview BPE tokenization for a given text."""
    data = request.get_json(force=True)
    text = data.get("text", "")
    tok = _tokenize_keyword(text)
    if tok is None:
        return jsonify({"error": "tokens.txt not found"}), 500
    return jsonify({"text": text.strip().upper(), "tokens": tok})


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
