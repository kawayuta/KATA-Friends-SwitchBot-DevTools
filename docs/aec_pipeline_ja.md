**[English](aec_pipeline.md)** | 日本語

# AEC パイプライン（エコーキャンセル + バージイン）

TTS 再生中のスピーカーエコーがマイクに回り込み、ウェイクワード検出 (KWS) が誤検知/検知不能になる問題を解決する。
speexdsp AEC で4つのマイクチャネルそれぞれのエコーを除去し、**TTS 再生中でもウェイクワード検出（バージイン）を可能にする。**

## 前提条件

- `snd-aloop` カーネルモジュール (`snd-aloop.ko`) がビルド済み
- デバイスに `libspeexdsp.so.1` がインストール済み
- Docker (arm64 コンテナ実行用、Apple Silicon Mac で動作確認済み)

## アーキテクチャ

```
TTS (mpg123 -a tts_out) → plug → hw:1,0,0 (loopback write)
                                      │
                                      ↓
                            hw:1,1,0 (loopback capture = TTS mono reference)
                                      │
                            ┌─────────┤ aec_daemon (6ch, speexdsp ×4)
                            │         │
                            ↓         ↓
                    softvol_ply     reference buf (ring buffer, mono)
                    → speaker         │
                                      │
    Real mic (cap_dsnoop hw:0,0) ──→ AEC process (4マイクチャネル)
    (6ch: mic0,ref,mic1,mic2,mic3,?)     │
                                    cleaned 6ch audio
                                      │
                                      ↓
                            hw:1,0,1 (loopback write, 6ch)
                                      │
                                      ↓
                            hw:1,1,1 (loopback capture, 6ch)
                                      │
                              record (plug) ← default capture
                                      │
                                      ↓
                              media (RK VQE + KWS) → pet_voice → ZMQ
```

### 信号の流れ

1. **TTS 音声**: mpg123 が `tts_out` (ALSA plug → `hw:1,0,0`) に出力
2. **Reference キャプチャ**: aec_daemon の Reference スレッドが `hw:1,1,0` (loopback 折り返し) から TTS 音声 (モノラル) をキャプチャし、リングバッファに格納すると同時に `softvol_ply` (スピーカー) に再生
3. **AEC 処理**: Main スレッドが `cap_dsnoop` から 6ch を読み込み、4つのマイクチャネル (ch0, ch2, ch3, ch4) それぞれにモノラル TTS reference で `speex_echo_cancellation()` を適用。ch1 (ハードウェア reference) と ch5 はパススルー
4. **クリーン音声出力**: AEC 処理済み 6ch 音声を `hw:1,0,1` (loopback) に書き込み
5. **media 入力**: `record` PCM (default capture) が `hw:1,1,1` にリダイレクトされ、media は AEC 済み 6ch 音声を読む。media 内蔵の RK VQE が追加のビームフォーミングと残留エコー抑制を実施
6. **KWS**: 従来通り media → pet_voice → ZMQ

### チャネルマッピング

| チャネル | 内容 | AEC処理 |
|---------|------|--------|
| 0 | mic0 | 適用 |
| 1 | speaker reference | パススルー |
| 2 | mic1 | 適用 |
| 3 | mic2 | 適用 |
| 4 | mic3 | 適用 |
| 5 | 不明 | パススルー |

## クイックスタート

```bash
# 1. AEC daemon をビルド
./scripts/build-aloop/build_aec.sh

# 2. デバイスにデプロイ (config変更 + daemon起動)
./scripts/deploy_aec.sh

# 3. 元に戻す場合
./scripts/restore_aec.sh
```

## ビルド

```bash
./scripts/build-aloop/build_aec.sh
```

arm64 Docker コンテナ (`ubuntu:22.04`) 内でネイティブコンパイルする。
出力: `scripts/build-aloop/output/aec_daemon` (ELF 64-bit aarch64)

### ビルド関連ファイル

| ファイル | 説明 |
|---------|------|
| `scripts/build-aloop/aec_daemon.c` | AEC daemon ソース (C, speexdsp, 6ch) |
| `scripts/build-aloop/build_aec.sh` | ビルドスクリプト |
| `scripts/build-aloop/Dockerfile` | ビルド環境 (arm64 multiarch 対応) |

### aec_daemon のパラメータ

| パラメータ | 値 |
|-----------|-----|
| サンプルレート | 16kHz |
| フォーマット | S16_LE |
| マイク入力 | 6ch (cap_dsnoop) |
| 出力 | 6ch (loopback hw:1,0,1) |
| Reference | モノラル (loopback hw:1,1,0) |
| フレームサイズ | 160 samples (10ms) |
| フィルタ長 | 2048 (128ms tail) |
| AEC インスタンス数 | 4 (マイクチャネルごとに1つ) |

## デプロイ

```bash
./scripts/deploy_aec.sh [KATA_IP]
```

デプロイスクリプトは以下を行う:

1. 元の `asound.conf` をバックアップ (`/data/devtools/backup_asound.conf`)
2. `aec_daemon` バイナリを `/data/devtools/` に push
3. `asound.conf` に `tts_out` デバイス定義を追加
4. `record` PCM を `hw:1,1,1` (6ch AEC 出力 loopback) にリダイレクト
5. `aec-pipeline.service` に `ExecStartPost` sleep を追加
6. `aec_daemon` を起動し、`master` サービスを再起動

### デバイス上の変更点

**asound.conf** (`/etc/asound.conf`):
```
pcm.tts_out {
    type plug
    slave { pcm "hw:1,0,0"; rate 16000; format S16_LE; channels 1 }
}

pcm.record {
    type plug
    slave { pcm "hw:1,1,1"; rate 16000; format S16_LE; channels 6 }
}
```

**app_flask.py** (mpg123 の出力先変更、3箇所):
```python
["mpg123", "-q", "-a", "tts_out", mp3_path]
```

## 復元

```bash
./scripts/restore_aec.sh [KATA_IP]
```

バックアップから `asound.conf` を復元し、`aec_daemon` を停止、`master` を再起動する。

## systemd サービス (自動起動)

`aec-pipeline.service` が有効化されていれば、デバイス再起動後も自動で:
1. `snd-aloop` モジュールをロード
2. `aec_daemon` を起動
3. 1秒待機 (`ExecStartPost`) で ALSA デバイス準備完了を待つ
4. `master.service` (media + pet_voice) より先に完了

```bash
# サービスの状態確認
adb shell "systemctl status aec-pipeline"

# 手動で停止 (モジュールもアンロード)
adb shell "systemctl stop aec-pipeline"

# 手動で起動
adb shell "systemctl start aec-pipeline"

# 自動起動を無効化
adb shell "systemctl disable aec-pipeline"

# 自動起動を再有効化
adb shell "systemctl enable aec-pipeline"
```

サービスファイル: `/data/overlay_upper/etc/systemd/system/aec-pipeline.service`

## バージイン (対話モードでの TTS 中断)

AEC パイプラインにより、対話モード (`/api/conversation/enable`) で TTS 再生中にウェイクワードを検出すると:

1. TTS 再生を即座に中断 (`mpg123` kill)
2. ウェイクワードと一緒に質問を言った場合は即座に LLM 処理 → 新しい TTS 応答
3. ウェイクワードのみの場合は listening 状態に戻る
4. 連鎖バージインにも対応（新しい TTS 中にさらにバージイン可能）

### 関連関数 (app_flask.py)

| 関数 | 説明 |
|------|------|
| `_tts_play_with_bargein()` | 非ブロッキング TTS + ZMQ 監視。ウェイクワード検出で TTS kill |
| `_conv_respond_with_bargein()` | LLM → TTS → バージインのループ処理 |

## トラブルシューティング

```bash
# aec_daemon が動作中か確認
adb shell "ps aux | grep aec_daemon | grep -v grep"

# Loopback デバイスが認識されているか
adb shell "aplay -l | grep Loopback"

# snd-aloop モジュールがロードされているか
adb shell "lsmod | grep snd_aloop"

# 6ch loopback 出力の確認
adb shell "cat /proc/asound/card1/pcm0p/sub1/hw_params"

# media が loopback から読んでいるか確認
adb shell "cat /proc/asound/card1/pcm1c/sub1/hw_params"

# systemd サービスのログ
adb shell "journalctl -u aec-pipeline --no-pager -n 30"

# aec_daemon を手動で起動してログ確認
adb shell "/data/devtools/aec_daemon"
```

### よくある問題

| 症状 | 原因 | 対策 |
|------|------|------|
| media が起動しない | `hw:1,1,1` が利用不可 (snd-aloop 未ロード or aec_daemon 未起動) | `systemctl start aec-pipeline` |
| TTS 音声がスピーカーから出ない | aec_daemon が停止している | `systemctl start aec-pipeline` |
| ウェイクワードが検出されない | AEC daemon がクラッシュ | `journalctl -u aec-pipeline` でログ確認、サービス再起動 |
| ビルドエラー (glibc mismatch) | x86 クロスコンパイラを使用している | `build_aec.sh` は arm64 Docker コンテナでネイティブコンパイルする |
| レイテンシが大きい | loopback バッファが大きすぎる | aec_daemon の `FRAME_SAMPLES * 8` バッファサイズを調整 |
