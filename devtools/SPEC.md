# Kata Friends DevTools 仕様書

## 概要

Kata Friends (SwitchBot AI Pet) のデバイス内部APIを操作するための開発者向けWebツール。
単一HTMLフロントエンド (vanilla JS + TailwindCSS CDN) のSPA構成。

**2つの動作モード:**

| モード | バックエンド | 通信方式 | 用途 |
|---|---|---|---|
| **オンデバイス** | Flask (デバイス上で直接実行) | localhost直接 + ZMQ直接 | 常時稼働、本番 |
| **Mac開発** | FastAPI (Docker or ローカル) | ADB経由リモート | 開発・デバッグ |

## ファイル構成

```
devtools/
├── app.py                 # FastAPI バックエンド — Mac開発用 (ポート 9001)
├── Dockerfile             # Docker イメージ定義
├── docker-compose.yml     # Docker Compose 設定
├── requirements.txt       # Python 依存 (FastAPI版)
├── zmq_publish.py         # ZMQ publish スクリプト (Mac版: ADB経由で実行)
├── setup_vlm.sh           # VLM自動セットアップスクリプト (adb経由)
├── static/
│   └── index.html         # SPA フロントエンド (Mac版)
├── ondevice/
│   ├── app_flask.py       # Flask バックエンド — オンデバイス版
│   ├── flask_server_diary.py  # LLM日記サーバー (persistent model版, VLM対応)
│   ├── zmq_publish.py     # ZMQ publish (デバイス上で直接import)
│   └── static/
│       └── index.html     # SPA フロントエンド (オンデバイス版)
├── SPEC.md                # 本仕様書
scripts/
└── deploy_devtools.sh     # ADB デプロイスクリプト
```

### デバイス上の配置 (`/data/devtools/`)

デプロイスクリプトが `devtools/ondevice/` の内容を `/data/devtools/` にコピーする。

```
/data/devtools/
├── app_flask.py
├── zmq_publish.py
├── start.sh                    # 起動ラッパースクリプト
├── custom_prompt.txt           # カスタム LLM プロンプトテンプレート
├── custom_llm_config.json      # カスタム LLM 設定 (temperature, max_new_tokens, task)
├── auto_talk_config.json       # 語りかけモード設定 (text, interval)
├── generated_diaries.json      # 生成済み日記の永続保存
├── prompt_backups/             # プロンプトバックアップ (自動作成)
│   └── <timestamp>/
│       ├── action_system_prompt.txt
│       ├── system_prompt_diary.txt
│       └── system_prompt_diary_translation.txt
└── static/
    └── index.html
```

systemd サービスファイル: `/data/overlay_upper/etc/systemd/system/kata-devtools.service`
overlay 永続化フラグ: `/overlay/overlay_upper` (このファイルが存在しないと起動時に overlay_upper が全削除される)

## 起動方法

### A. オンデバイス版 (推奨)

```bash
# デプロイ (Mac から ADB 経由)
bash scripts/deploy_devtools.sh [KATA_IP]

# ブラウザで http://<KATA_IP>:9001 を開く
```

デバイス再起動後も systemd により自動起動する。
Flask は `threaded=True` で起動し、並行リクエストを処理する。
HTML は `Cache-Control: no-cache, no-store, must-revalidate` ヘッダー付きで配信され、ブラウザキャッシュによる古い表示を防止する。
初回デプロイ時は overlayfs への反映のため **reboot が1回必要**。

**前提条件:**
- ADB接続済み (`adb connect <KATA_IP>:5555`)
- デバイスの `/data/` に書き込み可能 (root shell)

**デバイス上の依存 (すべてプリインストール済み):**
- Python 3.12.3, Flask 3.0.2, requests, jinja2, werkzeug
- libzmq.so.5 (ctypes経由で使用)

### B. Mac開発版 (Docker)

```bash
cd devtools
docker compose up --build

# ブラウザで http://localhost:9001 を開く
```

**前提条件:**
- `.env` に `KATA_IP`, `KATA_LOCAL_PORT`, `KATA_DEVICE_ID`, `KATA_LOCAL_TOKEN` が設定済み
- デバイスと同一ネットワーク上にいること
- デバイス上に `/data/pylib/zmq_publish.py` が配置済み (ZMQ機能使用時)

**Docker版の注意点:**
- Dockerfile に `android-tools-adb` をインストール済み
- ADB接続はアプリ側で初回リクエスト時に自動実行 (`adb connect` 不要)
- macOS では `network_mode: host` が動作しないため、ポートマッピング (`ports: 9001:9001`) を使用
- コンテナ内からデバイスIPへの TCP 接続でADB通信する

### C. Mac開発版 (ローカル)

```bash
# プロジェクトルートから
pip install -r devtools/requirements.txt
python3 -m uvicorn devtools.app:app --host 0.0.0.0 --port 9001 --reload
```

### 依存パッケージ (Mac版)

`requirements.txt`: `fastapi`, `uvicorn`, `httpx`, `python-dotenv`

## オンデバイス版の特徴

### FastAPI版との主な違い

| 項目 | Mac (FastAPI) | デバイス (Flask) |
|---|---|---|
| HTTP クライアント | `httpx.AsyncClient` | `requests` (同期) |
| ZMQ publish | ADB shell 経由 | `zmq_publish.py` を直接 import |
| API アクセス先 | `http://<KATA_IP>:<port>` | `http://127.0.0.1:<port>` |
| 認証情報 | `.env` から読み取り | MQTT ログから自動検出 |
| health check | ADB ポート含む全ポート | ローカルサービスのみ |
| 非同期 | async/await | 同期 |

### 認証情報の自動検出

オンデバイス版は `.env` が不要。以下を MQTT ログから自動取得:

- **DEVICE_ID**: `/data/cache/log/cc_mqtt.*.log` 内の `"deviceID":"<hex>"` パターン
- **LOCAL_TOKEN**: 同ログ内の `"token":"<uuid>"` パターン (functionID:1021 レスポンス)

取得タイミング: アプリ起動時 + `/api/local` 初回呼び出し時

### systemd サービス

```ini
[Unit]
Description=Kata Friends DevTools
After=network.target master.service

[Service]
Type=simple
User=wlab
WorkingDirectory=/data/devtools
ExecStart=/data/devtools/start.sh
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

`start.sh` の中身:
```bash
#!/bin/bash
cd /data/devtools
exec python3 app_flask.py
```

**管理コマンド (ADB shell内):**
```bash
systemctl status kata-devtools    # 状態確認
systemctl restart kata-devtools   # 再起動
journalctl -u kata-devtools -f    # ログ確認
```

### デプロイ手順

#### 初回セットアップ

```bash
# 1. ADB 接続
adb connect <KATA_IP>:5555

# 2. デプロイ実行
bash scripts/deploy_devtools.sh [KATA_IP]

# 3. 初回は overlayfs 反映のため reboot が必要
adb -s <KATA_IP>:5555 reboot

# 4. 再起動後、自動起動を確認 (30-60秒待つ)
adb connect <KATA_IP>:5555
adb -s <KATA_IP>:5555 shell systemctl status kata-devtools

# 5. ブラウザでアクセス
open http://<KATA_IP>:9001
```

#### アプリ更新時 (2回目以降)

overlayfs の設定は済んでいるので reboot 不要。

```bash
# 1. ファイルを push
bash scripts/deploy_devtools.sh [KATA_IP]

# 2. サービス再起動
adb -s <KATA_IP>:5555 shell systemctl restart kata-devtools
```

#### トラブルシューティング

```bash
# ログ確認
adb -s <KATA_IP>:5555 shell journalctl -u kata-devtools -f

# サービス状態
adb -s <KATA_IP>:5555 shell systemctl status kata-devtools

# overlay フラグ確認 (なければ overlay_upper が毎回消える)
adb -s <KATA_IP>:5555 shell ls -la /overlay/overlay_upper

# サービスファイル確認
adb -s <KATA_IP>:5555 shell ls -la /etc/systemd/system/kata-devtools.service

# 手動起動テスト
adb -s <KATA_IP>:5555 shell "cd /data/devtools && python3 app_flask.py"
```

### デプロイスクリプト (`scripts/deploy_devtools.sh`)

```bash
bash scripts/deploy_devtools.sh [KATA_IP]
# デフォルト IP: $KATA_IP 環境変数 or 192.168.11.17
```

**処理内容:**
1. ADB 接続確認
2. `/data/devtools/` ディレクトリ作成 + アプリファイル push + `start.sh` 作成
3. `/overlay/overlay_upper` フラグファイル作成 (overlay 永続化)
4. systemd サービスファイルを `/data/overlay_upper/etc/systemd/system/` に配置
5. `multi-user.target.wants` にシンボリックリンクで自動起動有効化
6. `systemctl enable --now` で即時起動を試行 (初回は要 reboot)

### overlayfs 永続化の仕組み

デバイスの rootfs は overlayfs で構成されている:
- **lowerdir**: `/app:/` (読み取り専用のベースイメージ)
- **upperdir**: `/data/overlay_upper/` (変更レイヤー)

`/sbin/init` が起動時に以下の分岐を行う:

| `/overlay/overlay_upper` ファイル | 動作 |
|---|---|
| **存在する** | overlay を **rw** マウント、upper の内容を保持 |
| **存在しない** | `overlay_upper/*` を**全削除**、overlay を **ro** マウント |

デプロイスクリプトは `touch /overlay/overlay_upper` でフラグを作成し、overlay upper への変更が再起動後も維持されるようにする。

> **注意:** overlay upper に新しいファイルを追加した場合、overlayfs の仕様上 **reboot が必要** (マウント後に upper に追加されたファイルはマージビューに即時反映されない)。

### OverlayFS 二重書き込み同期

デバイスの OverlayFS では `/app/opt/...`（tmpfs）と `/opt/...`（overlay）が別のファイルシステムとしてマウントされている。LLM サーバー群は `/opt/...` パスからプロンプトを読み込むため、`/app/opt/...` だけに書き込んでも反映されない。

DevTools はこの問題を以下の仕組みで解決する:

- **`_init_overlay_dirs()`**: アプリ起動時に実行。`EDITABLE_FILES` に登録された全ファイルについて `/app/opt/...` から `/opt/...` へコピーし、両パスの内容を同期する。
- **`_sync_to_overlay()`**: プロンプトの保存 (`/api/prompts/save`) および復元 (`/api/prompts/restore`) のたびに呼び出され、変更されたファイルを `/opt/...` 側にも書き込む。

これにより、LLM サーバーを再起動すれば常に最新のプロンプトが反映される。

## API エンドポイント一覧

### POST /api/action — LLM アクション判定

デバイスの LLM アクションサーバー (:8080) にテキストを送り、mood/instruction を取得する。
デバイスは動かない (判定のみ)。

**リクエスト:**
```json
{"voiceText": "踊って"}
```

**デバイスへの転送:**
```
POST http://<KATA_IP>:8080/rkllm_action
{"voiceText": "踊って"}
```

**レスポンス (パース済み):**
```json
{"raw": "neutral/dance", "mood": "neutral", "instruction": "dance"}
```

> デバイスの LLM サーバーはプレーンテキスト (`mood/instruction`) を返す。バックエンドで分解して返却。

---

### POST /api/execute — LLM判定 + ZMQ実行 (一気通貫)

LLM で instruction を判定した後、ACTION_MAP で ZMQ ペイロードに変換し、
ADB 経由でデバイスの ZMQ バスに publish してアクションを実際に実行する。

**リクエスト:**
```json
{"voiceText": "踊って"}
```

**処理フロー:**
1. `POST :8080/rkllm_action` → `neutral/dance`
2. `instruction=dance` → ACTION_MAP → `{"action":"RDANCE008","task_name":"music"}`
3. `adb shell python3 /data/pylib/zmq_publish.py '{"topic":"/agent/start_cc_task","payload":{...}}'`

**レスポンス:**
```json
{
  "raw": "neutral/dance",
  "mood": "neutral",
  "instruction": "dance",
  "payload": {"task_type":"voice","timestamp":1772711000000000000,"action":"RDANCE008","task_name":"music"},
  "executed": true,
  "device_output": "{\"ok\": true, \"topic_rc\": 21, \"payload_rc\": 96}"
}
```

**instruction が `no_action` の場合:**
```json
{"raw": "neutral/no_action", "mood": "neutral", "instruction": "no_action", "executed": false}
```

---

### POST /api/zmq/publish — ZMQ 直接 publish

任意のトピック/ペイロードをデバイスの ZMQ バスに直接 publish する。

**リクエスト:**
```json
{
  "topic": "/agent/start_cc_task",
  "payload": {"action": "RDANCE008", "task_name": "music", "task_type": "voice"}
}
```

**レスポンス:**
```json
{
  "status": "ok",
  "topic": "/agent/start_cc_task",
  "payload": {"action": "RDANCE008", "task_name": "music", "task_type": "voice"},
  "device_output": "{\"ok\": true, \"topic_rc\": 21, \"payload_rc\": 96}"
}
```

---

### POST /api/diary — LLM 日記生成

ルートサーバー (:8083) を経由してイベントリストから日記を生成する。ルートサーバーが重複排除、リトライ (最大3回)、フォーマットバリデーション、中国語→対象言語への翻訳を処理する。
生成成功時は結果を `/data/devtools/generated_diaries.json` に自動保存する。

**リクエスト:**
```json
{
  "events": [{"event": "被摸了脸", "time": "10:39"}, {"event": "被抱起来了", "time": "11:15"}],
  "language": "ja",
  "local_date": "2026-03-05"
}
```

> `events` はオブジェクト配列。各オブジェクトは `event` (中国語イベント名) と `time` (HH:MM) を持つ。

**デバイスへの転送:**
```
POST http://127.0.0.1:8083/rkllm_diary
```

ルートサーバーが内部で日記サーバー (:8082) を呼び出し、翻訳・バリデーションを行う。

**レスポンス:** ルートサーバーからの JSON を返却。生成成功時は `/data/devtools/generated_diaries.json` にも永続保存される。JSON パース失敗時は `{"raw": "..."}` で返す。

---

### GET /api/diary/records — イベント履歴・生成済み日記取得

デバイスの `diary_record.json` からイベント履歴を、`/data/devtools/generated_diaries.json` から生成済み日記を取得して返す。

**レスポンス:**
```json
{
  "events": {
    "2026-03-05": [{"event": "被摸了脸", "time": "10:39"}, {"event": "被抱起来了", "time": "11:15"}],
    "2026-03-04": [{"event": "被摸了头", "time": "09:00"}]
  },
  "generated": {
    "2026-03-05": {
      "title": "...",
      "diary": "...",
      "emotion": "...",
      "generated_at": "..."
    }
  }
}
```

> `events` は日付ごとにグループ化されたイベント配列。`generated` は日付をキーとする生成済み日記。

---

### POST /api/custom-llm — カスタム LLM 呼び出し

日記サーバー (:8082) をカスタムプロンプトテンプレートで呼び出す独立エンドポイント。

**リクエスト:**
```json
{"text": "今日はいい天気でした"}
```

**動作:**
1. `/data/devtools/custom_prompt.txt` をテンプレートとして読み込み、`{text}` プレースホルダーをリクエストの `text` で置換
2. `/data/devtools/custom_llm_config.json` から設定 (`temperature`, `max_new_tokens`, `task`) を読み込み
3. 日記サーバー (:8082) の `/rkllm_diary` に送信
4. 最大3回リトライ、タイムアウト 300秒

> **`task: "custom"` サポート:** 設定 JSON で `"task": "custom"` を指定すると、日記サーバーはシステムプロンプトを読み込まない（空文字列を返す）。これにより日記・翻訳用のシステムプロンプトを完全にバイパスし、カスタムプロンプトテンプレートのみで LLM を呼び出せる。

**レスポンス:** 日記サーバーからの応答をそのまま返却。

---

### GET /api/auto-talk/config — 語りかけモード設定取得

語りかけモードの設定を返す。ファイルが存在しない場合はデフォルト値を返す。

**レスポンス:**
```json
{"text": "今何が見える？", "interval": 60}
```

---

### POST /api/auto-talk/config — 語りかけモード設定保存

設定を `/data/devtools/auto_talk_config.json` に保存する。

**リクエスト:**
```json
{"text": "今何が見える？", "interval": 60}
```

---

### POST /api/auto-talk/start — 語りかけモード開始

バックエンドで daemon スレッドを起動し、定期的にカメラ撮影 + VLM 推論を実行する。

**リクエスト:**
```json
{"text": "今何が見える？", "interval": 60}
```

**レスポンス:**
```json
{"status": "started"}
```

> 既に実行中の場合は `{"status": "already_running"}` を返す。

---

### POST /api/auto-talk/stop — 語りかけモード停止

実行中のスレッドに停止シグナルを送る。`running` フラグは即座に `false` になる。

**レスポンス:**
```json
{"status": "stopped"}
```

---

### GET /api/auto-talk/status — 語りかけモード状態取得

実行状態・実行回数・最新の推論結果を返す。フロントエンドが 3秒ポーリングで使用する。

**レスポンス:**
```json
{
  "running": true,
  "count": 5,
  "last_result": {"time": "12:34:56", "text": "今何が見える？", "result": "...", "mode": "vlm"},
  "last_time": 1772711000.0
}
```

---

### GET /api/wakewords — ウェイクワード一覧取得

KWS の `keywords.txt` を読み込み、各行を BPE トークン列と可読テキストに変換して返す。

**レスポンス:**
```json
{
  "keywords": [
    {"text": "HELLO KATA", "tokens": "▁HE LL O ▁K AT A"},
    {"text": "HI NOAH", "tokens": "▁HI ▁NO A H"}
  ]
}
```

---

### POST /api/wakewords — ウェイクワード保存

ウェイクワードリストを `keywords.txt` に保存する。`restart: true` で `pet_voice` サービスも再起動。

**リクエスト:**
```json
{
  "keywords": [
    {"text": "HELLO KATA", "tokens": "▁HE LL O ▁K AT A"},
    {"text": "HELLO BUDDY"}
  ],
  "restart": true
}
```

> `tokens` が指定されていない場合、`text` から BPE トークン列を自動生成する。

**レスポンス:**
```json
{"status": "saved", "count": 2, "restarted": "pet_voice"}
```

---

### POST /api/wakewords/tokenize — BPE トークン化プレビュー

英語テキストを BPE トークン列に変換する（保存はしない）。

**リクエスト:**
```json
{"text": "HELLO BUDDY"}
```

**レスポンス:**
```json
{"text": "HELLO BUDDY", "tokens": "▁HE LL O ▁BU D D Y"}
```

---

### POST /api/device/reboot — デバイス再起動

`reboot` コマンドを実行してデバイスを再起動する。

**レスポンス:**
```json
{"status": "ok", "message": "rebooting"}
```

---

### POST /api/service/restart — サービス再起動

指定した systemd サービスを `systemctl restart` で再起動する。

**リクエスト:**
```json
{"service": "llm_diary"}
```

**許可されたサービス:** `kata-devtools`, `llm_action`, `llm_diary`, `llm_route`, `master`, `pet_voice`

**レスポンス:**
```json
{"status": "ok", "service": "llm_diary", "rc": 0}
```

---

### GET /api/tts/options — TTS オプション一覧

Voice 一覧と保存済み設定を返す。

**レスポンス:**
```json
{"voices": ["ja-JP-NanamiNeural","ja-JP-KeitaNeural","en-US-AriaNeural",...], "config": {}}
```

---

### POST /api/tts/config — TTS 設定保存

Voice・速度・ピッチ・再生先のデフォルト設定を `/data/devtools/tts_config.json` に保存する。

**リクエスト:**
```json
{"voice": "ja-JP-NanamiNeural", "rate": "+0%", "pitch": "+0Hz", "playback": "browser"}
```

---

### POST /api/tts — TTS 音声合成 (edge-tts)

Microsoft Edge TTS (edge-tts) を使用してテキストを音声に変換する。

**リクエスト:**
```json
{"text": "こんにちは", "voice": "ja-JP-NanamiNeural", "rate": "+0%", "pitch": "+0Hz", "browser": true}
```

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `text` | (必須) | 読み上げテキスト |
| `voice` | `"ja-JP-NanamiNeural"` | 音声 (ja-JP-NanamiNeural, ja-JP-KeitaNeural, en-US-* 等) |
| `rate` | `"+0%"` | 速度 (-50% 〜 +100%) |
| `pitch` | `"+0Hz"` | ピッチ (-100Hz 〜 +100Hz) |
| `browser` | `false` | `true`: MP3 バイナリを返却 (ブラウザ再生用)、`false`: デバイスで mpg123 再生 |

**動作:**
1. edge-tts (`edge_tts.Communicate`) でテキストを音声合成
2. `browser: true` の場合 → MP3 を `audio/mpeg` で返却
3. `browser: false` の場合 → `/tmp/tts_edge.mp3` に保存 → `mpg123` で非同期再生 → `{"status":"ok"}` を返却

---

### POST /api/local — ローカル API 呼び出し

デバイスのローカル API (:27999) に MD5 認証付きでリクエストを送る。

**リクエスト:**
```json
{"function_id": 9206, "params": {}}
```

**認証方式:** `auth` ヘッダー = `MD5(compact_json_body + KATA_LOCAL_TOKEN)`

**デバイスへの転送:**
```
POST http://<KATA_IP>:27999/thing_model/func_request
Content-Type: application/json
auth: <md5_hash>

{"version":"1","code":3,"deviceID":"...","payload":{"functionID":9206,"requestID":"...","timestamp":...,"params":{}}}
```

> JSON は `separators=(",",":")` でコンパクト化必須 (MD5計算と一致させるため)。

**既知の functionID:**

| ID | 機能 | params 例 |
|---|---|---|
| 9206 | ストレージ情報 | `{}` |
| 9217 | 写真タイムライン | `{"0":{"is_pic":true,"startTime":...,"endTime":...,"face_ids":[]}}` |
| 9225 | 顔認識データ | `{"0":["stranger","familiar"]}` |

---

### GET /api/sensors — ハードウェアセンサー情報

全ハードウェアセンサーとシステム情報を一括取得する。

**レスポンス:**
```json
{
  "thermal": [{"zone": 0, "type": "soc_thermal", "temp": 70.2}],
  "adc": {"channels": [{"channel": 0, "raw": 1024, "voltage": 1.8}], "scale": 1.758},
  "touch": {"sensitivity": 15},
  "leds": [{"name": "lcd", "brightness": 5, "max_brightness": 100}],
  "npu": {"raw": "Core0: 35%, Core1: 35%", "cores": [{"core": 0, "load": 35}]},
  "cpu": {"percent": 45.2},
  "memory": {"total_kb": 8000000, "available_kb": 4000000, "used_kb": 4000000, "percent": 50.0},
  "i2c_devices": [{"bus_addr": "6-0058", "name": "gtx315l"}],
  "cameras": [{"device": "video11", "name": "rkisp_mainpath"}],
  "audio": {"cards": [{"id": 0, "name": "rockchipes8388", "description": "..."}], "volume": 30, "mute": false},
  "battery": [{"name": "usb", "type": "USB", "online": 1}],
  "wifi": {"interface": "wlan0", "link": 70, "level_dbm": -40, "noise_dbm": -95},
  "disk": [{"mount": "/", "total_mb": 28000, "used_mb": 8000, "free_mb": 20000, "percent": 28.6}],
  "lidar": {"ttyUSB0": "CP2102 USB to UART"},
  "timestamp": 1709654400.0
}
```

> CPU 使用率はデルタ計算のため、初回呼び出しは `null` を返す。2回目以降で有効な値を返す。

---

### GET /api/sensors/camera/snapshot — ライブカメラフレーム取得

ISP selfpath (`/dev/video12`) から NV12 フレームをキャプチャし、JPEG に変換して返す。

**動作:**
1. `v4l2-ctl` で `/dev/video12` から 640x480 NV12 フレームを1枚取得
2. パイプで `ffmpeg` に渡し JPEG にエンコード
3. `image/jpeg` で返却（`Cache-Control: no-cache`）

**レスポンス:** JPEG 画像バイナリ (`Content-Type: image/jpeg`)

> `rkaiq_3A_server` が ISP mainpath (`/dev/video11`) を占有するため、selfpath を使用する。

---

### POST /api/sensors/volume — 音量変更

`amixer set Master` で Master 音量を変更する。

**リクエスト:**
```json
{"volume": 50}
```

**レスポンス:**
```json
{"status": "ok", "volume": 50}
```

> `volume` は 0〜100 の整数。範囲外はクランプされる。

---

### GET /api/health — デバイス生存確認

デバイスの各ポートに HTTP GET を送り、生存状態を返す。

**チェック対象:**

| 名前 | ポート | プロセス |
|---|---|---|
| adb | 5555 | adbd |
| zmq_xpub | 5558 | master (PUB接続先) |
| zmq_xsub | 5559 | master (SUB接続先) |
| llm_action | 8080 | flask_server_action.py |
| llm_diary | 8082 | flask_server_diary.py |
| llm_router | 8083 | route.py |
| local_api | 27999 | control_center_runner |
| control_center | 50001 | control_center_runner |

**レスポンス:**
```json
{
  "ip": "192.168.11.17",
  "services": {
    "llm_action": {"port": 8080, "status": "up", "code": 200},
    "adb": {"port": 5555, "status": "up (non-http)"},
    "zmq_xpub": {"port": 5558, "status": "down"}
  }
}
```

> ステータス: `up` (HTTP応答あり), `up (timeout)` (接続成功・応答なし), `up (non-http)` (接続成功・例外), `down` (接続不可)

---

### GET /api/events?n=50 — BLE イベントログ

`logs/kata_events.jsonl` から最新 N 件を取得する (新しい順)。

**レスポンス:**
```json
{
  "events": [
    {
      "received_at": "2026-03-04T21:25:24.457626",
      "timestamp": "2026-03-04T21:25:24.382587",
      "source": "kata_friends_ble",
      "type": "action",
      "data": {"action_counter": 51, "rssi": -66, "raw": "..."}
    }
  ],
  "total": 1234
}
```

**イベントタイプ:** `action` (アクション実行検知), `interaction_start`, `interaction_end`

---

### GET /api/camera/summary — カメラディレクトリ統計

全カメラディレクトリのファイル数・合計サイズを返す。

**レスポンス:**
```json
{
  "media_photo": {"count": 191, "total_size": 123456789},
  "origin": {"count": 50, "total_size": 1234567},
  ...
}
```

---

### GET /api/camera/list — ファイル一覧

**パラメータ:**

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `type` | カメラサブタイプ (例: `media_photo`, `origin`) | `origin` |
| `offset` | オフセット | `0` |
| `limit` | 取得件数 | `20` |

**レスポンス:**
```json
{
  "files": [{"name": "20260305_120000.png", "size": 1234567, "mtime": 1772711000}],
  "total": 191,
  "offset": 0,
  "limit": 20
}
```

> `media_photo` は `.png` のみ返す (`_mini.jpg`, `_thumb.jpg` は除外)。

---

### GET /api/camera/photo/<cat\>/<filename\> — ファイル取得

カメラディレクトリ内のファイルを直接配信する。パストラバーサル防止あり。

---

### GET /api/camera/faces — 顔ID一覧

**パラメータ:** `kind` (`known` or `unknown`), `include_empty` (`0` or `1`)

**レスポンス:**
```json
{
  "ids": [
    {
      "id": "ID_1234567890",
      "enrolled": ["face1.jpg", "face2.jpg"],
      "enrolled_dir": "enrolled_faces",
      "recognized_count": 15,
      "features_count": 3
    }
  ],
  "empty_count": 2
}
```

> `enrolled_dir` は既知顔では `enrolled_faces`、未知顔では `faces` になる。

---

### GET /api/camera/face_files — 顔サブフォルダ内ファイル一覧

**パラメータ:** `kind`, `id` (顔ID), `sub` (`enrolled_faces`/`recognized_faces`/`features`/`faces`), `offset`, `limit`

---

### POST /api/camera/cleanup_empty_faces — 空の顔IDを削除

ファイルが1つも含まれない顔IDディレクトリを削除する。

---

### POST /api/camera/delete — ファイル/顔ID削除

**リクエスト:**
```json
{
  "type": "media_photo",
  "files": ["20260305_120000.png"]
}
```

`media_photo` の `.png` 削除時は `_mini.jpg`, `_thumb.jpg` も自動削除。
`face_*` タイプで個別ファイル削除時は対応する `features/` も自動削除。

---

### GET /api/prompts — システムプロンプト取得

全3ファイルの内容を返す。

**レスポンス:**
```json
{
  "action": {"filename": "action_system_prompt.txt", "content": "..."},
  "diary": {"filename": "system_prompt_diary.txt", "content": "..."},
  "diary_translation": {"filename": "system_prompt_diary_translation.txt", "content": "..."}
}
```

---

### POST /api/prompts/save — プロンプト保存

**リクエスト:**
```json
{"key": "action", "content": "新しいプロンプト内容"}
```

---

### POST /api/prompts/restart — LLMサービス再起動

`llm_action`, `llm_diary`, `llm_route` の3サービスを `systemctl restart` で再起動する。

---

### GET /api/prompts/backups — バックアップ一覧

`/data/devtools/prompt_backups/` 内のバックアップを返す。

---

### POST /api/prompts/backup — バックアップ作成

全プロンプトファイルを `/data/devtools/prompt_backups/<timestamp>/` にコピーする。

---

### POST /api/prompts/restore — バックアップから復元

**リクエスト:** `{"name": "20260305_120000"}`

---

### POST /api/prompts/backup/delete — バックアップ削除

**リクエスト:** `{"name": "20260305_120000"}`

---

## ZMQ 通信仕様

### アーキテクチャ

**オンデバイス版 (直接):**
```
[app_flask.py (Flask)]
    ↓ import
[zmq_publish.py]
    ↓ PUB socket → tcp://127.0.0.1:5558
[master process (XPUB/XSUB proxy)]
    ↓ 5559 (XPUB)
[control_center_runner / ai_brain (SUB)]
    ↓
[モーター / 目 / サウンド 実行]
```

**Mac版 (ADB経由):**
```
[app.py (FastAPI on Mac)]
    ↓ ADB shell
[zmq_publish.py on device]
    ↓ PUB socket → tcp://127.0.0.1:5558
[master process (XPUB/XSUB proxy)]
    ↓ ...
```

### メッセージフォーマット

ZMQ multipart 2フレーム:
- **Frame 1 (トピック):** `#/agent/start_cc_task` (UTF-8 bytes, `#` プレフィックス必須)
- **Frame 2 (ペイロード):** msgpack str8/str16 エンコードされた JSON 文字列

msgpack str エンコード:
- `\xA0-\xBF` + data: fixstr (0-31 bytes)
- `\xD9` + 1byte len + data: str8 (32-255 bytes)
- `\xDA` + 2byte len (big-endian) + data: str16 (256-65535 bytes)

### ポート役割 (実測確認済み)

| ポート | ソケット種別 | 接続方向 | 用途 |
|---|---|---|---|
| 5558 | XSUB (bind) | **PUBがconnect** | publish側 (アクション送信) |
| 5559 | XPUB (bind) | **SUBがconnect** | subscribe側 (センサーデータ受信) |

> IPC ソケット (`/dev/shm/ipc.*`) は本デバイスでは存在しない。TCP のみ。

### トピック一覧

**制御トピック (publish用):**

| トピック | 用途 |
|---|---|
| `/agent/start_cc_task` | アクション実行 (メイン) |
| `/agent/stop_cc_task` | アクション停止 |
| `/ai/do_action` | (ai_brain内部用) |
| `/ai/mood` | 感情設定 |
| `/ai/sound` | サウンド再生 |
| `/ai/show_eyes` | 目アニメーション変更 |

**センサートピック (subscribe用, 5559から受信):**

`/agent/person_info`, `/agent/focal_posi`, `/line_laser_map`, `/line_point_cloud_above`, `/line_point_cloud_below` 等

---

## ACTION_MAP (instruction → ZMQ ペイロード)

LLM が返す `instruction` を ZMQ `start_cc_task` ペイロードに変換するマッピング。
`ai_brain` バイナリおよびログから抽出。

### モーションアクション (action フィールドあり)

| instruction | action | task_name | 説明 |
|---|---|---|---|
| `dance` | `RDANCE008` | `music` | 踊る |
| `sing` | `RSING001` | `music` | 歌う |
| `take_photo` | `RPIC001` | `take_photo` | 写真撮影 |
| `welcome` | `RHUG002` | `welcome` | 出迎え |
| `say_hello` | `RIMGHI001` | `hello` | 挨拶 |
| `bye` | `RTHDL50` | `bye` | バイバイ |
| `good_morning` | `RIMGHI001` | `good_morning` | おはよう |
| `good_night` | `RTHDL50` | `good_night` | おやすみ |
| `wave_hand` | `RIMGHI001` | `wave_hand` | 手を振る |
| `show_love` | `RHAPPY001` | `show_love` | 愛情表現 |
| `get_praise` | `RHAPPY001` | `get_praise` | 褒められた |
| `wake_up` | `RAWAKE001` | `wake_up` | 起きる |
| `nod` | `RANodyes` | `nod` | うなずき |
| `shake_head` | `RANO` | `shake_head` | 首振り |
| `speak` | `RSAYS001` | `speak` | 話す |
| `look_left` | `RAFL` | `look_left` | 左を見る |
| `look_right` | `RAFR` | `look_right` | 右を見る |
| `look_up` | `RAUP` | `look_up` | 上を見る |
| `look_down` | `RADOWN` | `look_down` | 下を見る |
| `spin` | `RWCIR001` | `spin` | 回転 |

**ペイロード形式:** `{"action":"<action>","task_name":"<task_name>","task_type":"voice","timestamp":<ns>}`

### 移動・制御系 (action フィールドなし)

| instruction | task_name | 説明 |
|---|---|---|
| `come_over` | `come_over` | こっち来て |
| `follow_me` | `follow_me` | ついてきて |
| `go_away` | `go_away` | あっち行って |
| `go_play` | `go_play` | 遊びに行く |
| `go_sleep` | `go_sleep` | 寝る |
| `go_power` | `go_power` | 充電しに行く |
| `go_to_kitchen` | `go_to_kitchen` | キッチンに行く |
| `go_to_bedroom` | `go_to_bedroom` | 寝室に行く |
| `go_to_balcony` | `go_to_balcony` | バルコニーに行く |
| `move_forward` | `move_forward` | 前進 |
| `move_back` | `move_back` | 後退 |
| `move_left` | `move_left` | 左移動 |
| `move_right` | `move_right` | 右移動 |
| `turn_left` | `turn_left` | 左旋回 |
| `turn_right` | `turn_right` | 右旋回 |
| `stop` | `stop` | 停止 |
| `be_silent` | `be_silent` | 静かにする |
| `volume_up` | `volume_up` | 音量上げ |
| `volume_down` | `volume_down` | 音量下げ |
| `user_leave` | `user_leave` | ユーザー退出 |

**ペイロード形式:** `{"task_name":"<task_name>","task_type":"voice","timestamp":<ns>}`

> 実際の `ai_brain` では移動系に `doa` (direction of arrival, 度数) フィールドが付くが、DevTools からは省略。

---

## デバイス上の zmq_publish.py 仕様

**配置場所:** `/data/pylib/zmq_publish.py`

**実行方法:**
```bash
python3 /data/pylib/zmq_publish.py '<json>'
```

**引数 JSON:**
```json
{
  "topic": "/agent/start_cc_task",
  "payload": {"action": "RDANCE008", "task_name": "music", "task_type": "voice", "timestamp": 1772711000000000000}
}
```

**動作:**
1. `libzmq.so.5` を ctypes で直接ロード (pyzmq 不要)
2. PUB ソケットを作成し `tcp://127.0.0.1:5558` に connect
3. 1秒待機 (subscription propagation)
4. multipart 送信: `[#<topic>, msgpack_str(<payload_json>)]`
5. 0.5秒待機後クローズ

**制約:**
- デバイスの rootfs は read-only のため pip/apt でのインストール不可
- `libzmq.so.5` は apt パッケージ `libzmq5` としてインストール済み
- Python 3.12.3 (`/usr/bin/python3`)
- `/data/` パーティションのみ書き込み可能

---

## フロントエンド タブ構成 (10タブ)

### 1. Action タブ

テキスト入力でデバイスにアクションを実行させる。

- **テキスト入力** + 「判定のみ」/「実行」ボタン
- **プリセットボタン:** 踊って / 写真撮って / こっち来て / おすわり / お手 / ハイタッチ (クリックで即実行)
- 「判定のみ」→ `POST /api/action` (LLM応答確認のみ)
- 「実行」→ `POST /api/execute` (LLM判定 → ZMQ publish)
- Enter キーで「実行」
- **ウェイクワード設定** (details パネル): KWS のウェイクワードを管理
  - 現在のウェイクワード一覧（テキスト + BPE トークン列）を表示
  - 英語テキスト入力で追加（自動 BPE トークン化、`POST /api/wakewords/tokenize`）
  - 個別削除ボタン (✕)
  - 「保存」→ `POST /api/wakewords`、「保存 + 再起動」→ 保存後に `pet_voice` を再起動して反映
  - 「再読込」→ `GET /api/wakewords` でデバイスから再取得

### 2. ZMQ タブ

ZMQ トピックとペイロードを直接指定して publish する上級者向け機能。

- **トピック選択:** `/agent/start_cc_task`, `/agent/stop_cc_task`, `/ai/do_action`, `/ai/mood`, `/ai/sound`, `/ai/show_eyes`
- **ペイロード入力:** JSON テキストエリア
- 「Publish」ボタン → `POST /api/zmq/publish` (オンデバイス版: 直接、Mac版: ADB経由)

### 3. Diary タブ

デバイスのイベント履歴から日記を生成・閲覧する。

**イベント履歴セクション:**
- デバイスの `diary_record.json` から取得したイベントを日付ごとにグループ表示 (新しい順)
- **イベント言語切り替え:** 日本語 / English / 中文 — 19種の中国語イベントタイプを i18n 翻訳して表示
- **日付ごとの「日記生成」ボタン:** ルートサーバー経由で日記を生成 (`POST /api/diary`)
- **生成済み日記:** 各日付の下にカードとして永続表示 (`/data/devtools/generated_diaries.json` から読み込み)

**手動生成セクション:**
- カスタムイベント入力による日記生成 (下部に配置)
- **言語選択:** 日本語 / English / 中文
- 「日記生成」→ `POST /api/diary`

### 4. Local API タブ

ローカル API (functionID) を直接呼び出す。

- **functionID:** ドロップダウン (Storage/Photos/Faces) + 直接入力
- **Params:** JSON テキストエリア (省略可)
- 「実行」→ `POST /api/local`

### 5. Camera タブ

デバイスのカメラ・メディアファイルを閲覧・管理する。

**サブタブ (9種):**

| サブタブ | ラベル | ソースディレクトリ | 説明 |
|---|---|---|---|
| `media_photo` | 📸 撮影写真 | `/media/photo/` | `take_photo` で撮影した写真 (SDカード) |
| `face_known` | 😀 顔(既知) | `/data/ai_brain_data/face_metadata/known/` | 登録済み顔データ |
| `face_unknown` | ❓ 顔(未知) | `/data/ai_brain_data/face_metadata/unknown/` | 未登録顔データ |
| `origin` | 📷 録画フレーム(cache) | `/data/cache/video_recorder/result/origin/` | 録画からのフレーム |
| `hand` | ✋ 手検出(cache) | `/data/cache/video_recorder/result/hand/` | 手検出結果 |
| `photos` | 🖼 写真(cache) | `/data/cache/photo/` | キャッシュ写真 |
| `video` | 🎬 録画(cache) | `/data/cache/video_recorder/archive/` | 録画アーカイブ |
| `video_archive` | 📼 録画AR(cache) | `/data/cache/video_recorder_archive/` | 録画ARアーカイブ |
| `sensor` | 📡 センサー(cache) | `/data/cache/recorder/archive/` | ROS bag (.db3.zip) |

**機能:**
- **無限スクロール:** IntersectionObserver で自動読み込み (200px手前で発火)
- **写真グリッド表示:** `media_photo` はサムネイル (`_thumb.jpg`) をグリッド表示、モーダルで中サイズ (`_mini.jpg`) 表示、原本 (`.png`) へのリンク
- **拡張子フィルタ:** `media_photo` は `.png` のみリスト (同名の `_mini.jpg`, `_thumb.jpg` を除外)
- **削除:** ファイル選択 → 一括削除。`media_photo` 削除時は関連する `_mini.jpg`, `_thumb.jpg` も自動削除
- **顔管理:** 顔カード表示、個別ファイル閲覧、空の顔ID一括削除
- **顔ディレクトリ:** 既知顔は `enrolled_faces/`、未知顔は `faces/` サブディレクトリを使用

### 6. Prompt タブ

LLMシステムプロンプトの閲覧・編集・バックアップ・復元。

**編集可能ファイル (EDITABLE_FILES):**

| キー | ファイルパス | 用途 |
|---|---|---|
| `action` | `/app/opt/wlab/sweepbot/share/llm_server/res/action_system_prompt.txt` | アクション判定用プロンプト |
| `diary` | `/app/opt/wlab/sweepbot/share/llm_server/res/system_prompt_diary.txt` | 日記生成用プロンプト |
| `diary_translation` | `/app/opt/wlab/sweepbot/share/llm_server/res/system_prompt_diary_translation.txt` | 翻訳用プロンプト |
| `custom_llm` | `/data/devtools/custom_prompt.txt` | カスタム LLM プロンプトテンプレート |
| `custom_llm_config` | `/data/devtools/custom_llm_config.json` | カスタム LLM 設定 (temperature, max_new_tokens, task) |
| `action_config` | `/opt/wlab/sweepbot/bin/llm_action_server.sh` | アクションサーバー起動設定 |
| `diary_config` | `/opt/wlab/sweepbot/bin/llm_diary_server.sh` | 日記サーバー起動設定 |

プロンプトファイル保存先: `/app/opt/wlab/sweepbot/share/llm_server/res/`

**機能:**
- **ドロップダウンでプロンプト選択** → テキストエリアで編集
- **保存ボタン:** ファイルを保存 (LLM再起動なし)
- **保存+再起動ボタン:** 保存後に `llm_action`, `llm_diary`, `llm_route` サービスを再起動
- **バックアップ:** 全プロンプトを `/data/devtools/prompt_backups/<timestamp>/` にコピー
- **復元:** バックアップからプロンプトファイルを復元
- **バックアップ削除:** 不要なバックアップを削除
- **遅延読み込み:** タブ初回表示時にプロンプトを読み込み

### 7. Custom LLM タブ

カスタムプロンプトテンプレートを使用した独立 LLM 呼び出し。

- **テキスト入力:** 自由テキストを入力
- **設定編集:** `temperature`, `max_new_tokens` をインラインで変更可能 (Prompt タブの `custom_llm_config` から読み込み)
- **プロンプトテンプレート:** `/data/devtools/custom_prompt.txt` の `{text}` プレースホルダーを入力テキストで置換
- **実行:** 日記サーバー (:8082) に直接送信 (`POST /api/custom-llm`)
- リトライ最大3回、タイムアウト 300秒
- **読み上げ (TTS):** チェックボックスで有効化。LLM 応答を edge-tts (Microsoft Edge TTS) で音声変換
  - **Voice:** 日本語2種 + 英語11種から選択 (デフォルト: ja-JP-NanamiNeural)
  - **速度:** 0.5x 〜 2x
  - **ピッチ:** -100Hz 〜 +100Hz (スライダーで5Hz刻み)
  - **再生先:** ブラウザ / デバイス (mpg123) / 両方
  - **設定保存:** 「デフォルトに保存」で `/data/devtools/tts_config.json` に永続化
- **語りかけモード:** 定期的にカメラ撮影 + LLM 推論を繰り返す自動実行モード (サーバーサイド実行)
  - **テキスト:** LLM に送信するテキスト (デフォルト: 「今何が見える？」、textarea で複数行入力可)
  - **間隔:** 実行間隔を秒数で指定 (デフォルト: 60秒、最小5秒)
  - **サーバーサイド実行:** バックエンドの daemon スレッドで動作。ページを閉じても推論は継続する
  - **開始/停止:** `POST /api/auto-talk/start` / `POST /api/auto-talk/stop` でスレッド制御。二重開始は `already_running` で拒否
  - **カメラ常時ON:** 開始時に VLM (カメラ) チェックボックスを自動的に有効化
  - **TTS 連携:** 読み上げ (TTS) がONの場合、ポーリングで新結果を検出し TTS 再生 (ブラウザ側)
  - **状態同期:** フロントエンドは 3秒ポーリング (`GET /api/auto-talk/status`) でサーバー状態を監視。ページリロード時も実行中なら UI を復元
  - **結果表示:** タイムスタンプ付きで `custom-result` に表示
  - **設定永続化:** `/data/devtools/auto_talk_config.json` に保存、リロード時に復元

### 8. Sensors タブ

ハードウェアセンサーとシステム情報をリアルタイム表示する。

**センサーデータ:**

| セクション | データソース | 表示内容 |
|---|---|---|
| 温度 | `/sys/class/thermal/thermal_zone{0-N}/temp` | 各ゾーンの温度（色グラデーション: <60℃ 緑、60-80℃ 黄、>80℃ 赤） |
| NPU | `/sys/kernel/debug/rknpu/load` | コアごとの負荷バー |
| CPU | `/proc/stat` | 使用率バー（デルタ計算） |
| メモリ | `/proc/meminfo` | 使用量 / 合計、バー表示 |
| ADC | `/sys/bus/iio/devices/iio:device0/in_voltage{0-7}_raw` | 8ch 電圧値 |
| タッチ | `/sys/bus/i2c/devices/6-0058/sensitivity` | gtx315l 感度 |
| LED | `/sys/class/leds/*/brightness` | 各 LED の輝度 ON/OFF |
| ストレージ | `os.statvfs()` | `/` と `/data` の使用量バー |
| WiFi | `/proc/net/wireless` | RSSI (dBm)、リンク品質バー |
| バッテリー | `/sys/class/power_supply/*/` | 残量、電圧、充電状態 |
| カメラ | `/sys/class/video4linux/*/name` + ISP selfpath | デバイス一覧 + ライブプレビュー |
| オーディオ | `/proc/asound/cards` + `amixer` | サウンドカード情報 + 音量スライダー（変更可能） |
| LiDAR | `/dev/ttyUSB*` | シリアルデバイス一覧 |
| I2C | `/sys/bus/i2c/devices/*/name` | バスアドレスとデバイス名 |

**機能:**
- 「更新」ボタン + 自動更新チェックボックス (2秒ポーリング)
- カメラ: 「スナップ」ボタンで ISP selfpath (`/dev/video12`) からリアルタイムフレーム取得、「ライブ」で 2秒間隔自動更新
- オーディオ: Master 音量スライダーでリアルタイム変更 (`amixer set Master`)

### 9. Events タブ

BLE イベントログの表示。

- 「更新」ボタン + 自動更新チェックボックス (5秒ポーリング)
- イベントタイプ別のバッジ表示
- `GET /api/events?n=50`

### 10. Status タブ

デバイスの各ポート生存状況とサービス・デバイスの再起動。

- 「チェック」ボタン → `GET /api/health`
- ポートごとに up/down をインジケーター表示
- **サービス再起動** (details パネル): DevTools / LLM Action / LLM Diary / LLM Route / Master の個別再起動ボタン (`POST /api/service/restart`)
- **デバイス再起動** (details パネル): 確認ダイアログ付き。`POST /api/device/reboot` でデバイスを再起動

### Log Panel (右側固定パネル)

全 API リクエスト/レスポンスをリアルタイム表示する固定パネル。

- **表示内容:** HTTP メソッド、URL、ステータスコード、レスポンス時間、リクエスト/レスポンスボディ
- **色分け:** 青 = リクエスト、緑 = 成功、赤 = エラー
- **トグルボタン:** パネルの表示/非表示を切り替え
- **自動スクロール:** 新しいログが追加されると自動的に最下部にスクロール
- **最大 200 エントリ:** 超過分は古い順に削除
- **クリアボタン:** ログを全消去

---

## セキュリティ

### アクセス範囲

| 接続方法 | 認証 | 備考 |
|---|---|---|
| ADB (:5555) | なし | LAN内のみ |
| LLM (:8080, :8082) | なし | LAN内のみ |
| ZMQ (:5558) | なし | LAN内のみ |
| ローカルAPI (:27999) | `MD5(body + token)` | トークンはデバイスごとに動的生成 |

## 環境変数 (.env)

| 変数名 | 説明 | 例 |
|---|---|---|
| `KATA_IP` | デバイスの IP アドレス | `192.168.11.17` |
| `KATA_LOCAL_PORT` | ローカル API ポート | `27999` |
| `KATA_DEVICE_ID` | デバイス ID | (MACアドレス由来の hex 文字列) |
| `KATA_LOCAL_TOKEN` | MD5 認証トークン (UUID) | (MQTT経由で動的取得) |

---

## 音声認識→アクション実行パイプライン (デバイス側)

全てローカル処理。クラウド不使用。端から端まで約2〜3秒。

```
[マイク] → pet_voice (VAD→KWS→ASR)
    → ZMQ /voice/vad_data
    → ai_brain (voice_component)
    → ZMQ /agent/llm_request
    → control_center (ai_proxy)
    → HTTP route.py (:8083) → flask_server_action (:8080)
    → RKLLM 推論 → mood/instruction
    → ai_brain → ZMQ /agent/start_cc_task
    → control_center → .act ファイル実行 (サーボ/目/音声)
```

### Step 1: 常時音声検知 (`pet_voice`)

| ステージ | モデル | サイズ | 処理時間 |
|---|---|---|---|
| VAD (音声検出) | Silero VAD (ONNX) | 643KB | 常時 |
| KWS (ウェイクワード) | sherpa-onnx KeywordSpotter | 14MB | 126〜342ms |
| ASR (テキスト化) | SenseVoice (RKNN) | 490MB | 298〜449ms |

- **プロセス:** `/opt/wlab/sweepbot/bin/pet_voice` (ネイティブバイナリ、CPU 47%、メモリ ~1GB)
- **サービス:** `pet_voice.service`
- **VAD:** マイクからの音声を常時監視、人の音声区間を検出
- **KWS:** `target_flag=1` 時にウェイクワードをチェック
- **ASR:** SenseVoice (sherpa-onnx OfflineRecognizer) でテキスト化
- **DOA:** マイクアレイから音源方向 (角度) を算出
- **出力:** ZMQ `/voice/vad_data` に JSON を publish

```json
{"doa_angle": 79.0, "is_wake_word": false, "segment_id": 1690, "text": "こないのって"}
```

**ウェイクワード** (`/opt/wlab/sweepbot/share/ai_brain/model/voice/kws/keywords.txt`):

sherpa-onnx KWS 形式（1行=1ウェイクワード、BPE トークン列）。英語のみ対応（学習データ: GigaSpeech-L + TTS）。
BPE 語彙 500 トークン（`tokens.txt`）で任意の英単語を合成可能。`keywords_threshold: 0.1`。

```
▁HE LL O ▁K AT A
▁HE LL O ▁NO A H
▁HE LL O ▁ N IC O
▁HI ▁K AT A
▁HI ▁NO A H
▁HI ▁ N IC O
```

DevTools の Action タブからウェイクワードの追加・削除・保存が可能。英語テキストを入力すると自動で BPE トークン化される。

**モデルファイル:**

| モデル | パス |
|---|---|
| VAD | `/opt/wlab/sweepbot/share/ai_brain/model/voice/silero_vad.onnx` |
| KWS | `/opt/wlab/sweepbot/share/ai_brain/model/voice/kws/{encoder,decoder,joiner}.onnx` |
| ASR | `/data/ai_brain/voice/sensevoice/model.rknn` |
| ASR 語彙 | `/data/ai_brain/voice/sensevoice/tokens.txt` |

### Step 2: LLM アクション判定

1. `ai_brain` (voice_component) が `/voice/vad_data` を subscribe
2. ASR テキストを `/agent/llm_request` に publish
3. `control_center` (ai_proxy) が受信し、`route.py` (:8083) → `flask_server_action` (:8080) に HTTP POST
4. RKLLM (NPU) が `mood/instruction` を推論（temperature=0.0, max_new_tokens=64, ~1秒）

```
"哥個唱歌" → "happy/sing"
"こないのって" → "neutral/no_action"
```

### Step 3: アクション実行

5. `route.py` がレスポンスをパース: `{"resultCode":100, "data":{"mood":"happy","instruction":"sing"}}`
6. `control_center` → `/cc/llm_response` → `ai_brain` (voice_component)
7. instruction → action ID に変換 (`sing` → `RSING001`)
8. `/agent/start_cc_task` に ZMQ publish
9. `control_center` が `.act` ファイルをロードしてサーボ・目・音声を制御

### 主要プロセス一覧

| プロセス | パス | 役割 |
|---|---|---|
| pet_voice | `/opt/wlab/sweepbot/bin/pet_voice` | VAD + KWS + ASR |
| ai_brain | `/opt/wlab/sweepbot/lib/ai_brain/ai_brain` | 音声→LLM連携、タスク発行 |
| control_center | `/opt/wlab/sweepbot/lib/control_center/control_center_runner` | タスク実行 (サーボ/目/音声) |
| master | (ZMQ XPUB/XSUB プロキシ) | 全ノード間メッセージ中継 |

### ログファイル

| ログ | パス |
|---|---|
| pet_voice | `/data/cache/log/pet_voice.log` |
| ai_brain | `/data/cache/log/ai_brain.log` |
| LLM Action | `/data/cache/log/rkllm_action_server.log` |
| Router | `/data/cache/log/rkllm_server.log` |
| control_center | `/data/cache/log/cc_main.latest.log` |

---

## LLM サーバー仕様 (デバイス側)

### アクションサーバー (:8080)

- **エンドポイント:** `POST /rkllm_action`
- **モデル:** Qwen3 + LoRA (`/data/ai_brain/actionmodel.rkllm`)
- **入力:** `{"voiceText": "踊って"}`
- **出力:** プレーンテキスト `mood/instruction` (例: `happy/dance`)
- **認証:** 不要
- **排他制御:** 同時リクエスト不可 (ビジー時 503)

### 日記サーバー (:8082)

- **エンドポイント:** `POST /rkllm_diary` (テキスト日記), `POST /rkllm_vlm` (VLM画像+テキスト)
- **モデル:** Qwen3-VL-2B (`/data/ai_brain/vlm/`) — persistent model方式 (起動時に1回ロード)
- **入力 (diary):** `{"task": "diary", "prompt": "language:Japanese\nlocal_date:...\nevents:\n..."}`
- **入力 (vlm):** `{"prompt": "この画像を説明して", "image_base64": "...", "max_new_tokens": 512}`
- **認証:** 不要
- **`task: "custom"` サポート:** `task` に `"custom"` を指定すると、システムプロンプトは読み込まれない (空文字列を返す)。Custom LLM 機能で使用。
- **librkllmrt.so:** VLMモデルは toolkit 1.2.3 で変換されているため、runtime 1.2.3 以上が必須 (`/data/rknn-llm/` から取得)
- **ソース:** `devtools/ondevice/flask_server_diary.py` → `setup_vlm.sh` でデプロイ

### LLM の instruction 集合 (システムプロンプトで定義)

```
wave_hand, come_over, go_power, go_play, take_photo, be_silent, nod,
shake_head, dance, look_left, look_right, look_up, look_down, go_away,
move_forward, move_back, move_left, move_right, spin, turn_left,
turn_right, go_to_kitchen, go_to_bedroom, go_to_balcony, good_morning,
bye, good_night, follow_me, stop, go_sleep, volume_up, volume_down,
sing, speak, welcome, user_leave, no_action, say_hello, show_love,
wake_up, get_praise
```

### LLM の mood 集合

```
happy, angry, sad, scared, disgusted, surprised, neutral
```

---

## アクションファイル一覧 (デバイス上)

**パス:** `/data/common/resource/pink/actions/*.act`

全 147 ファイル。ACTION_MAP で使用しているもの:

```
RADOWN, RAFL, RAFR, RANodyes, RANO, RAUP, RAWAKE001,
RDANCE008, RHAPPY001, RHUG002, RIMGHI001, RPIC001,
RSAYS001, RSING001, RTHDL50, RWCIR001
```

---

## 既知の問題 / トラブルシューティング

### Qwen3 モデルの `<think>` タグ問題

Qwen3 モデルは応答の先頭に `<think>...</think>` タグを出力し、思考トークンを消費することがある。これを防止するため、日記用システムプロンプトの末尾に `/no_think` を記述する必要がある。

### `max_context_len` 設定

`/opt/wlab/sweepbot/bin/llm_diary_server.sh` の `max_context_len` は **4096** に設定する必要がある (8192 ではない)。モデルの `max_context_limit` と一致させる必要があるため。

### 翻訳プロンプトのフォーマットバリデーション失敗

元の翻訳システムプロンプトに含まれる `Title / Body / Emotion` ラベルをモデルがヘッダーとしてエコーし、複数行出力になることでフォーマットバリデーション (`title/body/emotion` の3パート分割) が失敗していた。

**修正内容:**
- プロンプトを更新: 単一行出力を明示、`/` 区切り3パート、具体的な出力例を追加、末尾に `/no_think` を付加
- `route.py` にフォールバックを追加: 翻訳結果が2パート (感情タグ欠落) の場合、中国語生成時の感情タグを自動的に補完する

### ルートサーバーのハング

日記サーバーがスタックした場合、ルートサーバー (:8083) もハングすることがある。以下のコマンドで再起動する:

```bash
systemctl restart llm_route
```
