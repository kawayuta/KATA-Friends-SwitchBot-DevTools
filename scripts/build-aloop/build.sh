#!/bin/bash
set -euo pipefail

# 1. カーネルソース (Rockchip BSP develop-6.1)
if [ ! -d kernel ]; then
  git clone https://github.com/rockchip-linux/kernel \
    --branch develop-6.1 --depth 1 kernel
fi

cd kernel

# 2. Makefile のバージョンをデバイスに合わせる (6.1.99)
sed -i 's/^SUBLEVEL = .*/SUBLEVEL = 99/' Makefile

# 3. デバイスの config を使用
cp /build/device-config .config

# 4. CONFIG_SND_ALOOP=m に変更
scripts/config --module CONFIG_SND_ALOOP
make ARCH=arm64 CROSS_COMPILE=aarch64-none-linux-gnu- olddefconfig

# 5. ビルド準備
make ARCH=arm64 CROSS_COMPILE=aarch64-none-linux-gnu- prepare
make ARCH=arm64 CROSS_COMPILE=aarch64-none-linux-gnu- modules_prepare

# 6. snd-aloop モジュールのみビルド
make ARCH=arm64 CROSS_COMPILE=aarch64-none-linux-gnu- M=sound/drivers modules

# 6. 出力
cp sound/drivers/snd-aloop.ko /build/output/
echo "=== snd-aloop.ko built ==="
ls -la /build/output/snd-aloop.ko
