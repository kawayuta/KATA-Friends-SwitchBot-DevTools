**[English](api_integration.md)** | 日本語

# Kata Friends 自宅API連携プロジェクト

SwitchBot Kata Friendsのイベントを検知し、自宅APIへ転送するシステム。
ローカルAPIの認証を解明し、写真・顔認識データへのアクセスも実現済み。

## デバイス情報

| 項目 | 値 |
|---|---|
| デバイス名 | KATAフレンズ |
| BLE名 | WoAIPE (WonderLabs AI Pet) |
| メーカー | Woan Technology (Shenzhen) |
| BLE メーカーID | 2409 (SwitchBot/WonderLabs) |
| SwitchBot API deviceId | .envに記載 |
| Wi-Fi IP | .envに記載 |
| Wi-Fi | 2.4GHz帯に接続 |
| OS | Linux 6.1.99 aarch64 (Android系、ホスト名: WlabRobot) |
| QRコード (背面) | あり（製造シリアル番号） |

## 調査結果

### 1. Wi-Fi通信

インターネットへの通信はほぼなし。音声認識・カメラの結果をクラウドに送信していない。

### 2. SwitchBot公式API（v1.1）

| エンドポイント | 結果 |
|---|---|
| `GET /v1.1/devices` | デバイスリストに「KATAフレンズ」として表示される。ただし`deviceType`フィールドが欠落（他デバイスにはある） |
| `GET /v1.1/devices/{id}/status` | `statusCode: 100`（成功）だがbodyが空 `{}` |
| Webhook登録 | 登録は可能だがKata Friendsからのイベントは送信されない |

認証方式: HMAC-SHA256署名（`setup_webhook.py`に実装済み）

### 3. ローカルAPI（認証解明済み）

Kata FriendsはLAN上でHTTPサーバーを稼働。ポートはMQTT経由で動的に通知される。

| 項目 | 値 |
|---|---|
| エンドポイント | `POST /thing_model/func_request` |
| ヘルスチェック | `POST /heartbeat` |
| ポート | 27999（MQTTで動的配布、以前は22090だった） |
| 認証 | `auth: MD5(body + token)` |
| トークン | MQTTで配布（.envに保存） |
| 現状 | **動作中** |

#### 認証方式

```
auth = MD5(リクエストボディ + トークン)
```

- トークンはデバイスがMQTT経由でSwitchBotクラウドに通知するUUID形式の値
- ADB経由でデバイスログから取得: `cc_mqtt.*.log` の `functionID:1021` メッセージ内
- bodyが空（heartbeat）の場合: `auth = MD5(token)`

#### 解明の経緯

1. mitmproxyでiPhoneアプリの通信をキャプチャし、authヘッダーがMD5形式（32文字hex）であることを確認
2. heartbeatリクエスト（body空）で毎回同じauthになることから、タイムスタンプベースではないと判明
3. ADB（ポート5555が開放）でデバイスにroot接続し、ログからトークンを発見
4. `auth = MD5(body + token)` で全キャプチャ済みリクエストが一致

#### 利用可能な機能

| functionID | 機能 | 状態 |
|---|---|---|
| 9206 | ストレージ情報 | 動作確認済み |
| 9217 | 写真タイムライン（顔認識付き） | 動作確認済み（176枚取得） |
| 9225 | 顔認識データ（登録者・未登録者） | 動作確認済み（登録3人+未登録16人） |

リクエスト形式:
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

### 4. デバイス内部（ADB経由）

ADBデバッグがポート5555で常時有効。認証不要・root権限でシェルアクセス可能。
SwitchBotアプリやプロキシは不要で、Kata FriendsがWi-Fiに接続されていればMacから直接アクセスできる。

```bash
# adbがなければインストール
brew install android-platform-tools

# 接続（認証なし・root権限）
adb connect <KATA_IP>:5555
adb shell    # そのままrootシェルが開く
```

| 項目 | 値 |
|---|---|
| OS | Linux 6.1.99 aarch64 (Debian系) |
| ホスト名 | WlabRobot |
| ユーザー | root |
| Python | 3.12.3 |
| Webフレームワーク | Flask (Werkzeug) |
| LLMランタイム | RKLLM (Rockchip NPU) |
| チップ | RK3576 (Rockchip) |

開放ポート:

| ポート | 用途 |
|---|---|
| 5555 | ADB (Android Debug Bridge) |
| 5558 | ZMQ XPUB（内部IPC、サブスクライバーポート） |
| 5559 | ZMQ XSUB（内部IPC、パブリッシャーポート、センサーデータ） |
| 8080 | LLMアクションサーバー (Flask/RKLLM) |
| 8082 | LLM日記サーバー (Flask/RKLLM) |
| 8083 | LLM統合ルーター（8080/8082に自動振り分け） |
| 27999 | ローカルAPI (thing_model、auth必要) |
| 50001 | control_center_runner（用途不明） |

主要ディレクトリ:
```
/app/opt/wlab/sweepbot/bin/     # メインアプリケーション
  flask_server_action.py        # LLMアクションサーバー (port 8080)
  flask_server_diary.py         # 日記サーバー
  route.py                      # ルーティング
/data/cache/log/                # ログ
  cc_main.*.log                 # メインプロセスログ（auth検証等）
  cc_mqtt.*.log                 # MQTT通信ログ（トークン配布等）
  cc_bt.*.log                   # Bluetoothログ
/data/common/resource/          # リソース（目のアニメーション等）
```

### 5. 内部IPC — ZeroMQ（発見済み）

デバイス内部のプロセス間通信にZeroMQを使用。`master`プロセスがXPUB/XSUBプロキシとして動作。

| ソケット | アドレス | 役割 |
|---|---|---|
| XPUB | `tcp://127.0.0.1:5558` | サブスクライバー接続先 |
| XSUB | `tcp://127.0.0.1:5559` | パブリッシャー接続先 |

メッセージはZMQマルチパート形式: フレーム1 = `#` + トピック名、フレーム2 = MessagePackエンコードされたJSONペイロード。

**センサートピック**（ポート5559で流通）: `/imu`, `/tf`, `/odom`, `/curr_limb_pose`

**制御トピック**（バイナリ解析から判明）: `/ai/do_action`, `/ai/mood`, `/ai/sound`, `/ai/show_eyes`, `/agent/start_cc_task`, `/agent/stop_cc_task`

XSUBポートにパブリッシュすることで、LLMサーバーやローカルAPIを経由せずに直接アクションをトリガーできる可能性がある。

### 6. BLEアドバタイズ（動作中）

状態変化をパッシブに検知可能。現在のシステムはこの方式で動作。

```
xxxxxxxxxxxx | 4c | 01 | 2132 | 0010 | 39 | 00
[  MAC  6B ] |seq | ?? | 固定  | 固定  |b12 |b13
```

| バイト | 内容 | 備考 |
|---|---|---|
| 0-5 | BLE MACアドレス | 固定 |
| 6 | シーケンス番号 | リクエストごとにインクリメント |
| 7 | 不明 | 常に01 |
| 8-9 | 不明 | 常に2132 |
| 10-11 | 不明 | 常に0010 |
| 12 | アクションカウンタ | アクション実行ごとにデクリメント |
| 13 | インタラクションフラグ | 00=待機中, 03=音声応答中 |

### 7. BLE GATT

SwitchBot標準サービス `cba20d00-224d-11e6-9fb8-0002a5d5c51b` を持つ。Writeキャラ（cba20002）とNotifyキャラ（cba20003）あり。通知のプッシュはなくリクエスト-レスポンス型。

`0x57`プレフィクスのみ応答あり（`0x56`, `0x58`, `0x01`等は全滅）。`ble_brute.py`で0x5700〜0x57FFを総当たり済み。

| コマンド | 応答 | 解釈 |
|---|---|---|
| 0x5700 | `01` | OK応答のみ |
| 0x5701 | `05` | 非対応 |
| 0x5702 | `01 64 19` | 01=OK, 64=バッテリー100%?, 19=状態値 |
| 0x5703 | `05` | 非対応 |
| 0x5704 | `01 02` | 不明（2バイト応答） |
| その他 0x5705〜0x57FF | 応答なし | — |

## アクションの実行方法

### アクション実行フロー

```mermaid
flowchart LR
    A["🎤 音声 / curl"] --> B["LLM :8080\nQwen3+LoRA"]
    B --> C["mood/instruction"]
    C --> D[ZMQ\n/ai/do_action]
    D --> E[master\nXPUB/XSUB]
    E --> F[control_center]
    F --> G["🤖 実行"]
```

### LLMサーバー経由（認証不要）

```bash
# 自然言語を送信 → LLMが mood/instruction を返す
curl -X POST http://<KATA_IP>:8080/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "踊って"}'
# レスポンス: happy/dance

# 英語もOK
curl -X POST http://<KATA_IP>:8080/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "dance please"}'
```

これはLLMの*判定結果*のみ。実際のアクション実行は内部のZMQ経由で行われる。

ポート8080（直接）、8082（日記）、8083（ルーター）はすべて**認証不要**。

### ZMQ経由（直接制御、調査中）

```bash
adb shell
# /ai/do_action に ZMQ XSUB (tcp://127.0.0.1:5559) 経由でパブリッシュ
# 形式: マルチパート [#/ai/do_action, msgpack(json_payload)]
```

### 利用可能なアクション（全41種）

| カテゴリ | アクション |
|---|---|
| 移動 | `move_forward`, `move_back`, `move_left`, `move_right`, `spin`, `turn_left`, `turn_right`, `come_over`, `go_away`, `follow_me`, `stop` |
| ナビゲーション | `go_to_kitchen`, `go_to_bedroom`, `go_to_balcony` |
| 表現 | `dance`, `sing`, `nod`, `shake_head`, `wave_hand` |
| 視線 | `look_left`, `look_right`, `look_up`, `look_down` |
| 挨拶 | `good_morning`, `bye`, `good_night`, `say_hello`, `welcome` |
| 感情表現 | `show_love`, `get_praise` |
| 機能 | `take_photo`, `go_power`, `go_play`, `go_sleep`, `wake_up` |
| 音量 | `volume_up`, `volume_down`, `be_silent`, `speak` |
| その他 | `user_leave`, `no_action` |

### 利用可能な感情（全7種）

`happy`, `angry`, `sad`, `scared`, `disgusted`, `surprised`, `neutral`

## 現在の検知能力

### 検知できること

- **インタラクション開始**: byte[13]が00→03に変化（呼びかけ検知）
- **インタラクション終了**: byte[13]が03→00に変化
- **アクション実行**: byte[12]がデクリメント（ダンス・写真撮影等）
- **写真一覧**: ローカルAPI経由（認証解明済み）
- **顔認識データ**: 登録者・未登録者の一覧（ローカルAPI経由）
- **ストレージ情報**: 使用量・総容量（ローカルAPI経由）
- **デバイス内部**: ADB経由でファイルシステム・ログにアクセス可能

### 検知できないこと

- **音声認識テキスト**: デバイス内で完結、外部に送信されない
- **認識コマンドの種類**: 何を指示したかは不明（反応したかどうかのみ）

## アーキテクチャ

```mermaid
flowchart TB
    subgraph Mac["Mac (制御側)"]
        BLE[ble_watcher.py\nBLE監視]
        API[home_api\nFastAPI :9000]
        CLI[kata_local_api.py]
        CURL[curl / スクリプト]
    end

    subgraph Device["Kata Friends"]
        subgraph Internal["内部サービス"]
            VOICE[pet_voice\n音声認識]
            LLM[LLM :8080\nアクション]
            DIARY[LLM :8082\n日記]
            ZMQ[ZMQ IPC\n:5558/:5559]
            CC[control_center\n:27999]
            MASTER[master\nプロセス]
        end
        MOTOR["🤖 モーター/目/音声"]
    end

    BLE -.->|"BLEスキャン\nbyte[12],[13]"| Device
    BLE -->|"POST /events"| API
    CLI -->|"POST + MD5認証"| CC
    CURL -->|"POST (認証不要)"| LLM
    CURL -->|"adb shell"| Device

    VOICE --> LLM
    LLM --> ZMQ
    ZMQ --> MASTER
    MASTER --> MOTOR
    CC --> ZMQ
```

詳細は **[デバイス内部構造ドキュメント](../README_ja.md)** を参照。

## ディレクトリ構成

```
kata/
├── .env                      # 秘密情報（トークン、MAC等）
├── .env.example              # .envのテンプレート
├── .gitignore
├── README.md                 # このファイル
├── README_en.md              # English version
├── requirements.txt          # Python依存関係
├── ble_watcher.py            # BLEアドバタイズ監視 → APIイベント送信
├── home_api/
│   └── main.py               # FastAPIイベント受信サーバー
├── proxy/
│   ├── kata_proxy.py         # mitmproxy透過プロキシ（未使用: Wi-Fi通信なし）
│   └── capture_auth.py       # mitmweb auth解析用スクリプト
├── scripts/
│   ├── 01_discover.sh        # ネットワークデバイス探索
│   ├── 02_capture.sh         # tcpdumpパケットキャプチャ
│   ├── 03_analyze_pcap.sh    # pcap解析
│   ├── 04_setup_proxy.sh     # mitmproxyセットアップ
│   ├── 05_setup_routing.sh   # macOSルーティング設定
│   ├── 06_teardown_routing.sh # ルーティング解除
│   ├── setup_webhook.py      # SwitchBot公式API Webhook登録・管理
│   ├── kata_local_api.py     # ローカルAPIクライアント（認証済み・動作中）
│   ├── ble_monitor.py        # BLEアドバタイズ監視（デバッグ用）
│   ├── ble_gatt_explore.py   # GATT サービス探索
│   ├── ble_command.py        # BLEコマンド送信テスト
│   └── ble_brute.py          # BLE GATTコマンド総当たり探索
├── logs/                     # イベントログ（自動生成）
└── captures/                 # pcapファイル（自動生成）
```

## 起動方法

### ターミナル1: APIサーバー
```bash
cd ~/Documents/kata
python3 -m uvicorn home_api.main:app --host 0.0.0.0 --port 9000
```

### ターミナル2: BLE監視
```bash
cd ~/Documents/kata
python3 ble_watcher.py
```

Kata Friendsに話しかけるとイベントが検知され、APIに送信される。

### ローカルAPI利用
```bash
python3 scripts/kata_local_api.py storage    # ストレージ情報
python3 scripts/kata_local_api.py photos     # 写真一覧
python3 scripts/kata_local_api.py faces      # 顔認識データ
python3 scripts/kata_local_api.py discover   # functionID探索
python3 scripts/kata_local_api.py raw <ID>   # 任意のfunctionID
```

### ADB接続
```bash
adb connect <KATA_IP>:5555    # root権限でシェルアクセス
adb shell                      # デバイス内部を探索
```

## 依存関係

```bash
pip3 install -r requirements.txt
pip3 install bleak python-dotenv
```

## スクリプト

### メインシステム

#### ble_watcher.py

BLEアドバタイズを監視し、Kata Friendsの状態変化を検知してAPIサーバーにイベントを送信する。

```bash
python3 ble_watcher.py
```

送信されるイベント:
- `interaction_start` — 呼びかけ検知（byte[13]: 00→03）
- `interaction_end` — 応答終了（byte[13]: 03→00）
- `action` — アクション実行（byte[12]がデクリメント）

#### home_api/main.py

BLE監視からのイベントを受信するFastAPIサーバー。イベントは `logs/kata_events.jsonl` に記録される。

```bash
python3 -m uvicorn home_api.main:app --host 0.0.0.0 --port 9000
```

エンドポイント:
- `POST /events` — イベント受信
- `GET /health` — ヘルスチェック

#### scripts/kata_local_api.py

Kata FriendsのローカルAPIクライアント。写真・顔認識データ・ストレージ情報を取得できる。

認証方式: `auth = MD5(body + token)`（`.env`の`KATA_LOCAL_TOKEN`を使用）

```bash
python3 scripts/kata_local_api.py storage    # ストレージ情報
python3 scripts/kata_local_api.py photos     # 写真一覧（サムネイルURL付き）
python3 scripts/kata_local_api.py faces      # 顔認識データ
python3 scripts/kata_local_api.py discover   # functionID探索
python3 scripts/kata_local_api.py raw <ID>   # 任意のfunctionID
```

### 調査・デバッグ用スクリプト

#### scripts/ble_monitor.py

BLEアドバタイズの生データを表示するデバッグツール。変化したバイトをハイライト表示する。

```bash
python3 scripts/ble_monitor.py
```

#### scripts/ble_gatt_explore.py

BLE GATTサービスとキャラクタリスティックを列挙する。読み取り可能な値の取得と通知のサブスクライブも行う。

```bash
python3 scripts/ble_gatt_explore.py
```

#### scripts/ble_command.py

BLE GATTキャラクタリスティックにコマンドを送信して応答を確認する。SwitchBot標準コマンド（0x5701〜0x5721）と各種プレフィクスを試行する。

```bash
python3 scripts/ble_command.py
```

#### scripts/ble_brute.py

BLE GATTコマンドの総当たり探索。指定プレフィクスの2バイト目を0x00〜0xFFで全探索し、応答のあるコマンドを発見する。結果は `logs/` に保存される。

```bash
python3 scripts/ble_brute.py              # 0x5700〜0x57FF（デフォルト）
python3 scripts/ble_brute.py --prefix 01  # 0x0100〜0x01FF
```

#### proxy/capture_auth.py

mitmweb用のアドオンスクリプト。iPhone SwitchBotアプリとKata Friends間の通信をキャプチャし、authヘッダーの検証を行う。

```bash
mitmweb -s proxy/capture_auth.py -p 8888 --set connection_strategy=lazy
```

#### scripts/setup_webhook.py

SwitchBot公式API（v1.1）のWebhook管理。`.env`の`SWITCHBOT_TOKEN`と`SWITCHBOT_SECRET`が必要。

```bash
python3 scripts/setup_webhook.py setup <webhook_url>  # Webhook登録
python3 scripts/setup_webhook.py query                 # 登録状況確認
python3 scripts/setup_webhook.py delete                # Webhook削除
```

### ネットワーク調査スクリプト

```bash
# 1. ネットワーク上のデバイス探索（ARPテーブル表示）
bash scripts/01_discover.sh

# 2. Kata Friendsのパケットキャプチャ（sudo必要）
bash scripts/02_capture.sh <KATA_IP>

# 3. pcapファイルの解析（宛先IP、SNI、HTTPリクエスト抽出）
bash scripts/03_analyze_pcap.sh <pcapファイル>

# 4. mitmproxyのインストールとCA証明書セットアップ
bash scripts/04_setup_proxy.sh

# 5. macOSパケットフォワーディング設定（HTTPS通信をmitmproxyへ転送、sudo必要）
bash scripts/05_setup_routing.sh

# 6. パケットフォワーディング解除
bash scripts/06_teardown_routing.sh
```

## 次のステップ

| アプローチ | 概要 | 難易度 |
|---|---|---|
| ZMQアクショントリガー | `/ai/do_action`にZMQ経由でパブリッシュし、直接アクションを実行 | 中 |
| ZMQメッセージ形式のRE | 制御トピックのMessagePackペイロードを解読し、必要なフィールドを把握 | 中 |
| Macマイク併用 | BLE検知をトリガーにMacのマイクで録音→Whisper等で音声認識 | 低 |
| ローカルAPI functionID探索 | `kata_local_api.py discover` で未知のfunctionIDを発見する | 低 |
| ポート50001の調査 | 外部アクセス可能なポート50001の用途を特定 | 低 |
| LLMサーバー連携 | ポート8080のRKLLMサーバーに直接コマンドを送信 | 中 |
| ファームウェア解析 | デバイス内部のバイナリを詳細に解析 | 高 |
