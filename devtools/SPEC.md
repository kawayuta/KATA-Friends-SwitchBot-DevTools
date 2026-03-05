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
├── static/
│   └── index.html         # SPA フロントエンド (Mac版)
├── ondevice/
│   ├── app_flask.py       # Flask バックエンド — オンデバイス版
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
├── start.sh          # 起動ラッパースクリプト
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

### デプロイスクリプト (`scripts/deploy_devtools.sh`)

```bash
bash scripts/deploy_devtools.sh [KATA_IP]
# デフォルト IP: $KATA_IP 環境変数 or 192.168.11.17
```

**処理内容:**
1. ADB 接続確認
2. `/data/devtools/` ディレクトリ作成 + ファイル push
3. `/overlay/overlay_upper` フラグファイル作成 (overlay 永続化)
4. systemd サービスファイルを `/data/overlay_upper/etc/systemd/system/` に配置
5. `multi-user.target.wants` にシンボリックリンクで自動起動有効化
6. `systemctl enable --now` で即時起動 (初回は要 reboot)

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

デバイスの LLM 日記サーバー (:8082) にイベントリストを送り、日記を生成する。

**リクエスト:**
```json
{
  "events": ["散歩に行った", "ご飯を食べた"],
  "language": "ja",
  "local_date": "2026-03-05"
}
```

**デバイスへの転送:**
```
POST http://<KATA_IP>:8082/rkllm_diary
{
  "task": "diary",
  "prompt": "language:Japanese\nlocal_date:2026-03-05\nevents:\n散歩に行った\nご飯を食べた"
}
```

**レスポンス:** デバイスからの JSON をそのまま返却。JSON パース失敗時は `{"raw": "..."}` で返す。

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

## フロントエンド タブ構成

### 1. Action タブ

テキスト入力でデバイスにアクションを実行させる。

- **テキスト入力** + 「判定のみ」/「実行」ボタン
- **プリセットボタン:** 踊って / 写真撮って / こっち来て / おすわり / お手 / ハイタッチ (クリックで即実行)
- 「判定のみ」→ `POST /api/action` (LLM応答確認のみ)
- 「実行」→ `POST /api/execute` (LLM判定 → ZMQ publish)
- Enter キーで「実行」

### 2. ZMQ タブ

ZMQ トピックとペイロードを直接指定して publish する上級者向け機能。

- **トピック選択:** `/agent/start_cc_task`, `/agent/stop_cc_task`, `/ai/do_action`, `/ai/mood`, `/ai/sound`, `/ai/show_eyes`
- **ペイロード入力:** JSON テキストエリア
- 「Publish」ボタン → `POST /api/zmq/publish` (オンデバイス版: 直接、Mac版: ADB経由)

### 3. Diary タブ

イベントリストから日記を生成する。

- **イベントリスト:** 1行ずつ入力
- **言語選択:** 日本語 / English / 中文
- 「日記生成」→ `POST /api/diary`

### 4. Local API タブ

ローカル API (functionID) を直接呼び出す。

- **functionID:** ドロップダウン (Storage/Photos/Faces) + 直接入力
- **Params:** JSON テキストエリア (省略可)
- 「実行」→ `POST /api/local`

### 5. Events タブ

BLE イベントログの表示。

- 「更新」ボタン + 自動更新チェックボックス (5秒ポーリング)
- イベントタイプ別のバッジ表示
- `GET /api/events?n=50`

### 6. Status タブ

デバイスの各ポート生存状況。

- 「チェック」ボタン → `GET /api/health`
- ポートごとに up/down をインジケーター表示

---

## セキュリティ

### 認証情報の管理

- 実際のトークン・シークレットは `.env` に格納 (`.gitignore` 済み、リポジトリに含まれない)
- `.env.example` にはプレースホルダーのみ記載
- ローカルAPI認証方式 (`MD5(body + token)`) はコード内に公開しているが、トークン自体はデバイスごとにMQTT経由で動的に発行されるUUIDのため、アルゴリズム公開による影響なし

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

## LLM サーバー仕様 (デバイス側)

### アクションサーバー (:8080)

- **エンドポイント:** `POST /rkllm_action`
- **モデル:** Qwen3 + LoRA (`/data/ai_brain/actionmodel.rkllm`)
- **入力:** `{"voiceText": "踊って"}`
- **出力:** プレーンテキスト `mood/instruction` (例: `happy/dance`)
- **認証:** 不要
- **排他制御:** 同時リクエスト不可 (ビジー時 503)

### 日記サーバー (:8082)

- **エンドポイント:** `POST /rkllm_diary`
- **入力:** `{"task": "diary", "prompt": "language:Japanese\nlocal_date:...\nevents:\n..."}`
- **認証:** 不要

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
