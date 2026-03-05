**[English](README.md)** | 日本語

# Kata Friends デバイス内部構造

ADB経由でデバイス内部を調査した結果をまとめる。

## 接続方法

```bash
# adbのインストール（初回のみ）
brew install android-platform-tools

# 接続（認証不要・root権限）
adb connect <KATA_IP>:5555

# シェルを開く
adb shell
```

Wi-Fiに繋がっていればいつでもアクセス可能。SwitchBotアプリ不要。

## ハードウェア

| 項目 | 値 |
|---|---|
| CPU | ARM Cortex-A53 x4 (ARMv8-A) |
| チップ | Rockchip RK3576 |
| NPU | RKNN (Rockchip Neural Network) |
| RAM | 7.7GB |
| ストレージ | 28GB (/data) + SDカードスロット (/media/mmcblk1p1) |
| OS | Linux 6.1.99 aarch64 (Debian系) |
| ホスト名 | WlabRobot |
| Python | 3.12.3 |

## ファイルシステム概要

```
/
├── app/          196MB  アプリケーション（tmpfsオーバーレイ）
├── data/         8.5GB  ユーザーデータ・キャッシュ・AIモデル
├── rom/          1.5GB  読み取り専用ファイルシステム
├── usr/          1.3GB  システムバイナリ
├── media/        517MB  SDカード
├── opt/          195MB  追加パッケージ
└── overlay/      229MB  オーバーレイFS
```

## アプリケーション構造

### メインアプリ: `/app/opt/wlab/sweepbot/`

```
sweepbot/
├── bin/              # 実行ファイル (69個)
│   ├── master        # メインプロセス (395KB)
│   ├── media         # メディア処理 (1.2MB)
│   ├── pet_voice     # 音声処理 (985KB)
│   ├── recorder      # 録画サービス (591KB)
│   ├── rknn_server   # ニューラルネットワーク推論 (455KB)
│   ├── uart_ota      # OTAアップデート
│   │
│   │   # Python/Flaskサーバー
│   ├── flask_server_action.py  # LLMアクションサーバー (port 8080)
│   ├── flask_server_diary.py   # LLM日記サーバー (port 8082)
│   ├── route.py                # 統合ルーター (port 8083)
│   │
│   │   # シェルスクリプト (35個)
│   ├── rknn_server.sh
│   ├── llm_action_server.sh
│   ├── llm_diary_server.sh
│   ├── ai_brain.sh
│   ├── slam.sh
│   ├── media.sh
│   ├── pet_voice.sh
│   ├── petbot_eye.sh
│   └── ...
│
├── config/           # デバイスモデル別設定
│   ├── K20/          # MCUパラメータ
│   ├── K20Pro/
│   ├── S1/ S1+/ S10/ S20/ S20mini/ A01/
│   └── *.lua         # SLAM設定
│
├── lib/              # 共有ライブラリ
│   ├── libonnxruntime.so   # ML推論 (13MB)
│   ├── libmosquitto.so     # MQTTクライアント
│   ├── librkllmrt.so       # RKLLM推論ランタイム
│   └── ai_brain/ bt_bridge/ control_center/ lds_slam/
│
└── share/            # リソース・モデル設定
    ├── llm_server/res/
    │   ├── action_system_prompt.txt     # アクション用システムプロンプト
    │   ├── system_prompt_diary.txt      # 日記用システムプロンプト
    │   └── system_prompt_diary_translation.txt  # 翻訳用プロンプト
    ├── ai_brain/     # AI設定
    ├── bt_bridge/    # Bluetooth設定
    └── control_center/
```

## AIモデル

### LLM (大規模言語モデル)

`/data/ai_brain/` に格納。

| モデル | パス | サイズ | 用途 |
|---|---|---|---|
| Qwen3-1.7B | `Qwen3-1.7B_w8a8_RK3576_v3.rkllm` | 2.2GB | 日記生成 |
| Action Model (Qwen3 LoRA SFT) | `qwen3_v7.0.2_lora_sft_nothink_*.rkllm` | 900MB | アクション判定 |
| Action Model v1.1 | `actionmodel_w8a8_RK3576_v1.1.rkllm` | 900MB | 旧アクションモデル |

シンボリックリンク:
- `actionmodel.rkllm` → 最新のアクションモデル
- `diarymodel.rkllm` → Qwen3-1.7B

### 音声認識モデル

`/data/ai_brain/voice/` に格納。

| モデル | ファイル | 用途 |
|---|---|---|
| VAD | `vad/silero_vad.onnx` | 音声区間検出 (Voice Activity Detection) |
| KWS | `kws/encoder.onnx`, `decoder.onnx`, `joiner.onnx` | ウェイクワード検出 (Keyword Spotting) |
| SenseVoice | `sensevoice/model.rknn` | 音声認識 (ASR) |

ウェイクワード: `kws/keywords.txt` に定義

### 顔認識

バイナリベース。`/data/ai_brain_data/face_metadata/` に格納。

## データストレージ

### `/data/` ディレクトリ (8.5GB)

```
data/
├── ai_brain/              # AIモデル (5GB+)
│   ├── *.rkllm            # LLMモデル
│   ├── voice/             # 音声モデル (VAD, KWS, SenseVoice)
│   ├── llm_server/        # LLMサーバー設定
│   └── model_version.json # モデルバージョン管理
│
├── ai_brain_data/         # AI実行時データ (19MB)
│   └── face_metadata/
│       ├── known/         # 登録済み顔 (ID_*/)
│       │   └── ID_xxx/
│       │       ├── enrolled_faces/   # 登録時の顔写真 (.jpg)
│       │       ├── features/         # 顔特徴量ベクタ (.bin, 2KB each)
│       │       └── recognized_faces/ # 認識された顔写真 (.jpg)
│       └── unknown/       # 未登録の顔
│           └── timestamp/
│               ├── enrolled_faces/
│               └── features/
│
├── control_center/        # メイン制御データ
│   ├── db/sqlite.db       # SQLiteデータベース
│   ├── maps/              # ナビゲーションマップ
│   ├── snapshots/         # マップスナップショット
│   ├── ai_images/         # AI生成画像
│   │   ├── current/       # 最新
│   │   └── history/       # 履歴
│   └── task/              # タスク管理
│       ├── current_task
│       └── check_task
│
├── cache/                 # キャッシュ (835MB)
│   ├── log/               # ログファイル (40+)
│   │   ├── cc_main.*.log      # メインプロセス
│   │   ├── cc_mqtt.*.log      # MQTT通信
│   │   ├── cc_bt.*.log        # Bluetooth
│   │   ├── rkllm_*.log        # LLM推論
│   │   ├── wpa_supplicant.log # WiFi
│   │   └── ...
│   ├── image_recorder_archive/  # 撮影写真
│   ├── video_recorder_archive/  # 録画動画
│   └── vad/               # VADキャッシュ
│
├── common/                # 共有リソース (2.1GB)
│   └── resource/
│       ├── pink/          # デフォルトテーマ
│       │   ├── actions/   # アクションファイル (169個, .act)
│       │   └── eyes/      # 目のアニメーション (PNG, L/R)
│       ├── blue/          # 青テーマ
│       ├── black/         # 黒テーマ
│       ├── base_eye/      # ベース目データ
│       ├── limbs/         # 手足のデータ
│       ├── wheels/        # 車輪データ
│       └── sounds/        # サウンドエフェクト
│
├── map_server/            # SLAMナビゲーション
│   ├── refined_maps/      # 整形済みマップ
│   ├── labels/            # エリアラベル
│   └── markers/           # マーカー
│
└── slam/                  # SLAMデバッグデータ
```

## 内部サービス

### systemdサービス一覧 (28個)

| サービス | 機能 |
|---|---|
| `master.service` | メインプロセス制御 |
| `app.service` | アプリケーション |
| `ai_brain.service` | AI頭脳 (認識・判断) |
| `rknn_server.service` | ニューラルネットワーク推論 |
| `llm_action.service` | LLMアクション判定 (port 8080) |
| `llm_diary.service` | LLM日記生成 (port 8082) |
| `llm_route.service` | LLMルーター (port 8083) |
| `media.service` | メディア処理 |
| `pet_voice.service` | 音声処理 |
| `petbot_eye.service` | 目のアニメーション |
| `recorder.service` | 録画 |
| `slam.service` | SLAM (自己位置推定・地図生成) |
| `bt_bridge.service` | Bluetooth |
| `network_monitor.service` | ネットワーク監視 |
| `bringup.service` | 起動シーケンス |
| `system_helper.service` | システムヘルパー |
| `update-robotic.service` | OTAアップデート |
| `serial-control.service` | シリアル通信制御 |
| `upload_image.service` | 画像アップロード |
| `upload_video.service` | 動画アップロード |
| `upload_audio.service` | 音声アップロード |
| `upload-recorder.service` | 録画アップロード |
| `debug_log_push.service` | デバッグログ送信 |
| `debuglog_clean.service` | ログクリーンアップ |
| `klog_record.service` | カーネルログ記録 |
| `clean.service` | クリーンアップ |
| `sd-auto-mount.service` | SDカード自動マウント |
| `usb_event_proc.service` | USBイベント処理 |

### 内部HTTPサーバー

| ポート | サービス | 説明 |
|---|---|---|
| 5555 | adbd | ADBデーモン |
| 5558 | master (ZMQ XPUB) | ZMQサブスクライバーポート（内部IPC） |
| 5559 | master (ZMQ XSUB) | ZMQパブリッシャーポート（内部IPC、センサーデータ流通） |
| 8080 | flask_server_action.py | LLMアクション判定。音声テキストを受け取り `mood/instruction` を返す |
| 8082 | flask_server_diary.py | LLM日記生成。イベントリストから日記を生成 |
| 8083 | route.py | 統合ルーター。リクエスト内容に応じて8080/8082に振り分け |
| 27999 | control_center_runner (C++) | ローカルAPI。写真・顔認識・ストレージ等 (auth必要) |
| 50001 | control_center_runner (C++) | 不明（外部アクセス可能） |

### 内部IPC (ZeroMQ)

デバイス内部のプロセス間通信にZeroMQ (ZMQ)を使用。`master`プロセスがXPUB/XSUBプロキシとして動作する。

#### ソケット

| ソケット | アドレス | 役割 |
|---|---|---|
| XPUB | `tcp://127.0.0.1:5558` | サブスクライバー接続先 |
| XSUB | `tcp://127.0.0.1:5559` | パブリッシャー接続先 |
| XPUB (IPC) | `ipc:///dev/shm/ipc.xpub` | IPCサブスクライバーソケット |
| XSUB (IPC) | `ipc:///dev/shm/ipc.xsub` | IPCパブリッシャーソケット |

注意: IPCソケットファイルは存在しない場合がある。TCP経由のほうが確実。

#### メッセージフォーマット

ZMQマルチパートメッセージ（2フレーム構成）:

1. **フレーム1（トピック）**: `#` + トピック名（例: `#/imu`, `#/agent/start_cc_task`）
2. **フレーム2（ペイロード）**: MessagePackエンコード（msgpack str8 `\xd9` + 長さ + JSONペイロード）

#### 観測済みトピック（ポート5559）

| トピック | 内容 | データ |
|---|---|---|
| `/imu` | IMUセンサーデータ | 加速度計・ジャイロスコープ（imu_linkフレーム） |
| `/tf` | 座標変換ツリー | odom → base_footprint変換 |
| `/odom` | オドメトリ | 位置・速度・共分散行列 |
| `/curr_limb_pose` | 手足の姿勢 | 現在のサーボ/手足の状態 |

#### 制御トピック（バイナリ解析から判明）

| トピック | 用途 |
|---|---|
| `/ai/do_action` | アクションのトリガー（ダンス、写真等） |
| `/ai/mood` | 気分・感情の設定 |
| `/ai/sound` | サウンド再生 |
| `/ai/show_eyes` | 目のアニメーション変更 |
| `/agent/start_cc_task` | コントロールセンタータスクの開始 |
| `/agent/stop_cc_task` | コントロールセンタータスクの停止 |

#### メッセージの送信

```bash
# デバイス上でADB経由（pyzmqは未インストール、ctypesまたはスクリプトをpush）
adb shell

# トピック監視（XPUBポート5558にSUBとして接続）
# コマンド送信（XSUBポート5559にPUBとして接続）
```

`/agent/start_cc_task`のメッセージ例:
```
フレーム1: b'#/agent/start_cc_task'
フレーム2: msgpack(json_payload_string)
```

## アクションの実行方法

### フロー: 音声コマンド → アクション実行

```mermaid
flowchart LR
    A["🎤 音声入力"] --> B[pet_voice\nSenseVoice ASR]
    B --> C[テキスト]
    C --> D["LLMアクションサーバー\n:8080 Qwen3+LoRA"]
    D --> E["mood/instruction\n例: happy/dance"]
    E --> F[ZMQパブリッシュ\n/ai/do_action]
    F --> G[masterプロセス\nXPUB/XSUBプロキシ]
    G --> H[control_center\nアクション実行]
    H --> I["🤖 モーター/目/音声"]
```

### フロー: 外部からの制御（Mac側）

```mermaid
flowchart TB
    subgraph Mac
        M1["curl / スクリプト"]
        M2[ble_watcher.py]
        M3[kata_local_api.py]
    end

    subgraph Device["Kata Friends (192.168.x.x)"]
        subgraph LLM["LLMサーバー群 (認証不要)"]
            L1[":8080 アクション"]
            L2[":8082 日記"]
            L3[":8083 ルーター"]
        end
        subgraph ZMQ["ZMQ IPC (localhost限定)"]
            Z1[":5558 XPUB"]
            Z2[":5559 XSUB"]
        end
        subgraph API["ローカルAPI (認証必要)"]
            A1[":27999 thing_model"]
        end
        ADB[":5555 ADB\nrootシェル"]
    end

    M1 -->|"POST /rkllm_action\n(認証不要)"| L1
    M1 -->|"POST /rkllm_diary\n(認証不要)"| L2
    M3 -->|"POST + MD5認証"| A1
    M2 -.->|"BLEパッシブスキャン"| Device
    M1 -->|"adb shell"| ADB
    ADB -->|"ZMQクライアント"| Z2

    L3 --> L1
    L3 --> L2
```

### フロー: 日記生成

```mermaid
flowchart LR
    A[1日のイベントログ] --> B["POST :8082/rkllm_diary\ntask: diary"]
    B --> C["Qwen3-1.7B\nNPU上で実行"]
    C --> D["タイトル / 本文 / 感情\n(Pixar童話風)"]
    D --> E{言語?}
    E -->|中国語| F[完了]
    E -->|その他| G[翻訳\nプロンプト]
    G --> F
```

Kata Friendsにアクションを実行させる方法は3つある:

### 方法1: LLMアクションサーバー（最も簡単）

自然言語テキストを送ると、デバイス上のLLMがアクションを判定する。**認証不要。**

```bash
# 基本例
curl -X POST http://<KATA_IP>:8080/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "踊って"}'
# レスポンス: happy/dance

# 統合ルーター経由（同じ結果）
curl -X POST http://<KATA_IP>:8083/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "写真撮って"}'
# レスポンス: happy/take_photo

# 英語もOK
curl -X POST http://<KATA_IP>:8080/rkllm_action \
  -H "Content-Type: application/json" \
  -d '{"voiceText": "dance please"}'
```

レスポンス形式: `mood/instruction`（例: `happy/dance`, `neutral/no_action`）

**注意**: これはLLMの*判定結果*を返すだけで、アクション自体は実行しない。実際の実行はZMQトピックをサブスクライブする内部プロセスが行う。

### 方法2: ZMQ IPC（直接制御、調査中）

ADB経由でデバイス内部のZMQトピックに直接パブリッシュする。LLMをバイパスできる。

```bash
# デバイスに接続
adb shell

# /ai/do_action トピックにパブリッシュ（ZMQクライアントが必要）
# メッセージ形式: マルチパート [#/ai/do_action, msgpack(payload)]
```

状態: メッセージ形式は部分的に判明。制御トピックのペイロード構造は引き続き調査中。

### 方法3: ローカルAPI（認証必要）

データ取得用（写真、顔認識、ストレージ）。アクションのトリガーには使えない。

```bash
python3 scripts/kata_local_api.py photos    # 写真一覧
python3 scripts/kata_local_api.py faces     # 顔認識データ
python3 scripts/kata_local_api.py storage   # ストレージ情報
```

## LLMアクションサーバー詳細

### 概要

音声認識テキストを受け取り、AIペットの反応（感情＋動作）を返す。Rockchip NPU上でRKLLM（量子化Qwen3 + LoRA SFT）として動作。

### エンドポイント

```
POST http://<KATA_IP>:8080/rkllm_action
Content-Type: application/json

{"voiceText": "踊って"}
```

レスポンス: `happy/dance`

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

### 判定ルール

| 入力 | 結果 |
|---|---|
| ウェイクワードのみ（hello, niko, noa, kata） | `neutral/no_action` |
| 背景ノイズ・口癖 | `neutral/no_action` |
| ウェイクワード + 明確な指令 | ウェイクワード無視して実行 |
| 褒め言葉（見た目） | `happy/show_love` |
| 褒め言葉（行動） | `happy/get_praise` |
| 叱責 | `angry/no_action` or `sad/stop` |

## LLM日記サーバー詳細

### 概要

1日のインタラクションイベントから、AIペット視点の日記を生成する。

### エンドポイント

```
POST http://<KATA_IP>:8082/rkllm_diary
Content-Type: application/json

{
  "task": "diary",
  "prompt": "language:Chinese\nlocal_date:2026-03-05\nevents:\n08:00 - 醒来啦\n19:15 - 被摸了耳朵"
}
```

レスポンス: `タイトル/日記本文/感情`

### 日記の特徴

- **Pixar童話風**: 温かく、友好的で生き生きとした文体
- **パートナー目線**: ユーザーを「主人」ではなく対等な仲間として扱う
- **時間ぼかし**: 具体的な時刻は「朝」「夜」等に変換
- **多言語対応**: 中国語で生成後、指定言語に翻訳（日本語、英語、韓国語等）

### 利用可能な感情:
`Happy`, `Excited`, `Relaxed`, `Curious`, `Loved`, `Sleepy`, `Sad`, `Scared`, `Angry`, `Lonely`

## 顔認識データ

### ディレクトリ構造

```
/data/ai_brain_data/face_metadata/
├── known/                     # 登録済み
│   └── ID_<timestamp>/       # 顔ごとのディレクトリ
│       ├── enrolled_faces/   # 登録時の写真 (.jpg)
│       ├── features/         # 顔特徴量ベクタ (.bin, 各2056B)
│       └── recognized_faces/ # 認識された写真 (.jpg, 大量)
└── unknown/                   # 未登録
    └── <timestamp>/
        ├── enrolled_faces/
        └── features/
```

### アクセス方法

```bash
# 登録済み顔の一覧
adb shell "ls /data/ai_brain_data/face_metadata/known/"

# 特定の顔の登録写真をMacに取得
adb pull /data/ai_brain_data/face_metadata/known/ID_xxx/enrolled_faces/

# 認識された写真を取得
adb pull /data/ai_brain_data/face_metadata/known/ID_xxx/recognized_faces/

# ローカルAPI経由でも取得可能
python3 scripts/kata_local_api.py faces
```

## 写真・動画

### 写真

ローカルAPI経由でサムネイル付きリストを取得:

```bash
python3 scripts/kata_local_api.py photos
```

ADB経由で直接アクセス:

```bash
# 撮影写真（キャッシュ）
adb shell "ls /data/cache/image_recorder_archive/"

# Macにダウンロード
adb pull /data/cache/image_recorder_archive/ ./kata_photos/
```

### 動画

```bash
adb shell "ls /data/cache/video_recorder_archive/"
adb pull /data/cache/video_recorder_archive/ ./kata_videos/
```

## ログ

### ログファイル一覧

`/data/cache/log/` に40以上のログファイル。

| ログ | 内容 |
|---|---|
| `cc_main.*.log` | メインプロセス（認証検証、イベント処理等） |
| `cc_mqtt.*.log` | MQTT通信（トークン配布、プロパティ変更等） |
| `cc_bt.*.log` | Bluetooth通信（BLEアドバタイズデータ等） |
| `rkllm_action_server.log` | LLMアクション推論ログ |
| `rkllm_server.log` | LLMルーターログ |
| `wpa_supplicant.log` | WiFi接続ログ |
| `task_executor_runner.log` | タスク実行ログ |

### リアルタイムログ監視

```bash
# メインプロセスの最新ログを表示
adb shell "tail -f /data/cache/log/cc_main.*.log"

# MQTT通信を監視
adb shell "tail -f /data/cache/log/cc_mqtt.*.log"

# LLM推論を監視
adb shell "tail -f /data/cache/log/rkllm_action_server.log"
```

## リソースファイル

### テーマ

`/data/common/resource/` に3つのカラーテーマ:

| テーマ | パス |
|---|---|
| ピンク | `/data/common/resource/pink/` |
| ブルー | `/data/common/resource/blue/` |
| ブラック | `/data/common/resource/black/` |

各テーマに含まれるもの:
- `actions/` — アクションファイル (169個, `.act`形式)
- `eyes/` — 目のアニメーションフレーム (PNG, 左右別)

### アクション

169個の`.act`ファイル。命名規則:

| プレフィクス | 意味 | 例 |
|---|---|---|
| `RDANCE` | ダンス | `RDANCE008.act` |
| `RKATA` | Kata固有 | `RKATA1.act` ~ `RKATA6.act` |
| `RSING` | 歌 | `RSING001.act` |
| `RSLEEP` | 睡眠 | `RSLEEP000.act` |
| `RMAP` | マップ移動 | `RMAPGO.act`, `RMAPBACK.act` |
| `RPIC` | 写真撮影 | `RPIC001.act` |
| `RGO` | 移動 | `RGO000.act`, `RGO001.act` |
| `RW*` | 歩行 | `RWF001.act`, `RWL.act`, `RWR.act` |

### 目のアニメーション

目の表情はPNGフレームで構成（左目_L、右目_R）:

| アニメーション | 説明 |
|---|---|
| `OPEN_L/R` | 目を開く |
| `CLOSE_L/R` | 目を閉じる |
| `ESleep01_L/R` | 睡眠 |
| `ENAWAKE*_L/R` | 目覚め |
| `ESL0-SL4_L/R` | 瞼の動き |

## SQLiteデータベース

`/data/control_center/db/sqlite.db`

デバイスにsqlite3コマンドがないため、Macにダウンロードして確認:

```bash
adb pull /data/control_center/db/sqlite.db ./
sqlite3 sqlite.db ".tables"
sqlite3 sqlite.db ".schema"
```

## ファイルの取得方法まとめ

| データ | 方法 |
|---|---|
| 写真一覧 | `python3 scripts/kata_local_api.py photos` |
| 顔認識データ | `python3 scripts/kata_local_api.py faces` |
| ストレージ情報 | `python3 scripts/kata_local_api.py storage` |
| 写真ファイル | `adb pull /data/cache/image_recorder_archive/` |
| 顔写真 | `adb pull /data/ai_brain_data/face_metadata/` |
| 動画ファイル | `adb pull /data/cache/video_recorder_archive/` |
| ログ | `adb shell "cat /data/cache/log/cc_main.*.log"` |
| LLMモデル | `adb pull /data/ai_brain/actionmodel.rkllm` |
| アクションファイル | `adb pull /data/common/resource/pink/actions/` |
| 目のアニメーション | `adb pull /data/common/resource/pink/eyes/` |
| SQLiteデータベース | `adb pull /data/control_center/db/sqlite.db` |
| システムプロンプト | `adb shell "cat /app/opt/wlab/sweepbot/share/llm_server/res/*.txt"` |
