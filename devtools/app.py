"""
Kata Friends Developer Tools — FastAPI バックエンド

起動:
  python3 -m uvicorn devtools.app:app --host 0.0.0.0 --port 9001 --reload
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

KATA_IP = os.environ.get("KATA_IP", "192.168.11.17")
KATA_PORT = int(os.environ.get("KATA_LOCAL_PORT", "27999"))
KATA_DEVICE_ID = os.environ.get("KATA_DEVICE_ID", "")
KATA_LOCAL_TOKEN = os.environ.get("KATA_LOCAL_TOKEN", "")
EVENTS_LOG = Path(__file__).resolve().parent.parent / "logs" / "kata_events.jsonl"

app = FastAPI(title="Kata Friends DevTools")


# --- Models ---

class ActionRequest(BaseModel):
    voiceText: str

class DiaryRequest(BaseModel):
    events: list[str]
    language: str = "ja"
    local_date: str = ""

class LocalAPIRequest(BaseModel):
    function_id: int
    params: Optional[dict] = None

class ZmqPublishRequest(BaseModel):
    topic: str = "/ai/do_action"
    payload: dict

class ExecuteActionRequest(BaseModel):
    voiceText: str


# --- Helpers ---

def make_auth(body_str: str) -> str:
    return hashlib.md5((body_str + KATA_LOCAL_TOKEN).encode()).hexdigest()


def build_local_payload(function_id: int, params: dict | None = None) -> str:
    payload = {
        "version": "1",
        "code": 3,
        "deviceID": KATA_DEVICE_ID,
        "payload": {
            "functionID": function_id,
            "requestID": str(uuid.uuid4()).upper(),
            "timestamp": int(time.time() * 1000),
            "params": params or {},
        },
    }
    return json.dumps(payload, separators=(",", ":"))


# --- Endpoints ---

@app.post("/api/action")
async def proxy_action(req: ActionRequest):
    """LLM アクションサーバー (:8080) へ転送"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"http://{KATA_IP}:8080/rkllm_action",
                json={"voiceText": req.voiceText},
            )
            text = resp.text.strip()
            # レスポンスは "mood/instruction" 形式のプレーンテキスト
            parts = text.split("/", 1)
            return {
                "raw": text,
                "mood": parts[0] if parts else text,
                "instruction": parts[1] if len(parts) > 1 else "",
            }
        except httpx.ConnectError:
            raise HTTPException(502, "アクションサーバーに接続できません")
        except Exception as e:
            raise HTTPException(502, str(e))


@app.post("/api/diary")
async def proxy_diary(req: DiaryRequest):
    """LLM 日記サーバー (:8082) へ転送"""
    # 言語マッピング
    lang_map = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
    lang_name = lang_map.get(req.language, req.language)
    date_str = req.local_date or time.strftime("%Y-%m-%d")
    events_str = "\n".join(req.events)
    prompt = f"language:{lang_name}\nlocal_date:{date_str}\nevents:\n{events_str}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"http://{KATA_IP}:8082/rkllm_diary",
                json={"task": "diary", "prompt": prompt},
            )
            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text.strip()}
        except httpx.ConnectError:
            raise HTTPException(502, "日記サーバーに接続できません")
        except Exception as e:
            raise HTTPException(502, str(e))


@app.post("/api/local")
async def proxy_local(req: LocalAPIRequest):
    """ローカルAPI (:27999) へ転送 (MD5 auth 付き)"""
    body_str = build_local_payload(req.function_id, req.params)
    auth = make_auth(body_str)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "auth": auth,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"http://{KATA_IP}:{KATA_PORT}/thing_model/func_request",
                content=body_str,
                headers=headers,
            )
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(502, "ローカルAPIに接続できません")
        except Exception as e:
            raise HTTPException(502, str(e))


_adb_connected = False

async def adb_shell(cmd: str, timeout: float = 10.0) -> str:
    """ADB経由でデバイス上のコマンドを実行"""
    global _adb_connected
    if not _adb_connected:
        p = await asyncio.create_subprocess_exec(
            "adb", "connect", f"{KATA_IP}:5555",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await p.communicate()
        _adb_connected = True
    proc = await asyncio.create_subprocess_exec(
        "adb", "-s", f"{KATA_IP}:5555", "shell", cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(504, "ADBコマンドがタイムアウトしました")
    if proc.returncode != 0:
        raise HTTPException(502, f"ADBエラー: {stderr.decode().strip()}")
    return stdout.decode().strip()


ZMQ_SCRIPT_PATH = "/data/pylib/zmq_publish.py"

# LLM instruction → ZMQ start_cc_task ペイロード
# ai_brainバイナリ + ログから抽出したマッピング
# action付き = モーションアクション、actionなし = 移動/制御系 (doaフィールド)
ACTION_MAP = {
    # モーションアクション (action + task_name)
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
    # 移動系 (actionなし、task_nameのみ)
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


@app.post("/api/zmq/publish")
async def zmq_publish(req: ZmqPublishRequest):
    """ADB経由でデバイスのZMQバスにpublish"""
    msg = json.dumps({"topic": req.topic, "payload": req.payload}, separators=(",", ":"), ensure_ascii=False)
    cmd = f"python3 {ZMQ_SCRIPT_PATH} {shlex.quote(msg)}"
    result = await adb_shell(cmd, timeout=15.0)
    return {"status": "ok", "topic": req.topic, "payload": req.payload, "device_output": result}


@app.post("/api/execute")
async def execute_action(req: ExecuteActionRequest):
    """LLM判定 → ZMQ publish の一気通貫実行"""
    # Step 1: LLMアクションサーバーで判定
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"http://{KATA_IP}:8080/rkllm_action",
                json={"voiceText": req.voiceText},
            )
        except httpx.ConnectError:
            raise HTTPException(502, "アクションサーバーに接続できません")
    text = resp.text.strip()
    parts = text.split("/", 1)
    mood = parts[0] if parts else text
    instruction = parts[1] if len(parts) > 1 else ""

    if not instruction or instruction == "no_action":
        return {"raw": text, "mood": mood, "instruction": instruction, "executed": False}

    # Step 2: instructionをZMQ start_cc_taskペイロードに変換してpublish
    mapping = ACTION_MAP.get(instruction)
    if not mapping:
        return {"raw": text, "mood": mood, "instruction": instruction, "executed": False, "error": f"未知のinstruction: {instruction}"}

    ts = int(time.time() * 1e9)
    payload_dict = {"task_type": "voice", "timestamp": ts, **mapping}
    msg = json.dumps({"topic": "/agent/start_cc_task", "payload": payload_dict}, separators=(",", ":"))
    cmd = f"python3 {ZMQ_SCRIPT_PATH} {shlex.quote(msg)}"
    try:
        device_output = await adb_shell(cmd, timeout=15.0)
    except HTTPException:
        return {"raw": text, "mood": mood, "instruction": instruction, "executed": False, "error": "ZMQ publish失敗 (ADB接続確認)"}
    return {"raw": text, "mood": mood, "instruction": instruction, "payload": payload_dict, "executed": True, "device_output": device_output}


@app.get("/api/health")
async def health_check():
    """デバイスの各ポート生存確認"""
    ports = {
        "adb": 5555,
        "zmq_xpub": 5558,
        "zmq_xsub": 5559,
        "llm_action": 8080,
        "llm_diary": 8082,
        "llm_router": 8083,
        "local_api": KATA_PORT,
        "control_center": 50001,
    }
    results = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for name, port in ports.items():
            try:
                resp = await client.get(f"http://{KATA_IP}:{port}/")
                results[name] = {"port": port, "status": "up", "code": resp.status_code}
            except httpx.ConnectError:
                results[name] = {"port": port, "status": "down"}
            except httpx.ReadTimeout:
                results[name] = {"port": port, "status": "up (timeout)"}
            except Exception:
                results[name] = {"port": port, "status": "up (non-http)"}
    return {"ip": KATA_IP, "services": results}


@app.get("/api/events")
async def get_events(n: int = 50):
    """イベントログの最新N件"""
    if not EVENTS_LOG.exists():
        return {"events": [], "total": 0}
    lines = EVENTS_LOG.read_text().strip().split("\n")
    events = []
    for line in lines[-n:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {"events": list(reversed(events)), "total": len(lines)}


# Static files (HTML frontend)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
