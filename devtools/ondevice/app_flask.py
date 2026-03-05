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


@app.post("/api/diary")
def proxy_diary():
    """LLM diary server (:8082) proxy."""
    data = request.get_json(force=True)
    events = data.get("events", [])
    language = data.get("language", "ja")
    local_date = data.get("local_date", "") or time.strftime("%Y-%m-%d")

    lang_map = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
    lang_name = lang_map.get(language, language)
    events_str = "\n".join(events)
    prompt = f"language:{lang_name}\nlocal_date:{local_date}\nevents:\n{events_str}"

    try:
        resp = requests.post(
            "http://127.0.0.1:8082/rkllm_diary",
            json={"task": "diary", "prompt": prompt},
            timeout=30,
        )
        try:
            return jsonify(resp.json())
        except Exception:
            return jsonify({"raw": resp.text.strip()})
    except requests.ConnectionError:
        return flask_error(502, "diary server unreachable")
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

    if not instruction or instruction == "no_action":
        return jsonify({"raw": text, "mood": mood, "instruction": instruction, "executed": False})

    # Step 2: Map instruction -> ZMQ publish
    mapping = ACTION_MAP.get(instruction)
    if not mapping:
        return jsonify({"raw": text, "mood": mood, "instruction": instruction, "executed": False,
                        "error": f"unknown instruction: {instruction}"})

    ts = int(time.time() * 1e9)
    payload_dict = {"task_type": "voice", "timestamp": ts, **mapping}
    payload_json = json.dumps(payload_dict, separators=(",", ":"))

    try:
        zmq_publish_msg("/agent/start_cc_task", payload_json)
    except Exception:
        return jsonify({"raw": text, "mood": mood, "instruction": instruction, "executed": False,
                        "error": "ZMQ publish failed"})

    return jsonify({"raw": text, "mood": mood, "instruction": instruction, "payload": payload_dict, "executed": True})


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
    "origin": "/data/cache/video_recorder/result/origin/",
    "hand": "/data/cache/video_recorder/result/hand/",
    "photos": "/data/cache/photo/",
    "video": "/data/cache/video_recorder/archive/",
    "video_archive": "/data/cache/video_recorder_archive/",
    "sensor": "/data/cache/recorder/archive/",
    "face_known": "/data/ai_brain_data/face_metadata/known/",
    "face_unknown": "/data/ai_brain_data/face_metadata/unknown/",
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


def _dir_stats(path, recursive=False):
    """Return (count, total_size) for files in a directory."""
    if not os.path.isdir(path):
        return 0, 0
    count = 0
    total = 0
    if recursive:
        for root, _dirs, files in os.walk(path):
            for name in files:
                fp = os.path.join(root, name)
                count += 1
                total += os.path.getsize(fp)
    else:
        for name in os.listdir(path):
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                count += 1
                total += os.path.getsize(fp)
    return count, total


@app.get("/api/camera/summary")
def camera_summary():
    result = {}
    for key, path in CAMERA_DIRS.items():
        count, total_size = _dir_stats(path, recursive=(key in RECURSIVE_DIRS))
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

    entries = []
    if cat in RECURSIVE_DIRS:
        for root, _dirs, files in os.walk(dirpath):
            for name in files:
                fp = os.path.join(root, name)
                relpath = os.path.relpath(fp, dirpath)
                entries.append({
                    "name": relpath,
                    "size": os.path.getsize(fp),
                    "mtime": int(os.path.getmtime(fp)),
                })
    else:
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
        # enrolled_faces
        ef_path = os.path.join(id_path, "enrolled_faces")
        if os.path.isdir(ef_path):
            info["enrolled"] = sorted(os.listdir(ef_path))[:5]  # max 5 thumbnails
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
    if subfolder not in ("enrolled_faces", "recognized_faces", "features"):
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
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    _ensure_config()
    print(f"[DevTools] Device ID: {DEVICE_ID or '(not found)'}")
    print(f"[DevTools] Token: {'***' + LOCAL_TOKEN[-4:] if LOCAL_TOKEN else '(not found)'}")
    app.run(host="0.0.0.0", port=9001, debug=False)
