FROM --platform=linux/amd64 ubuntu:22.04
RUN apt-get update && apt-get install -y \
  build-essential bc bison flex libssl-dev make libelf-dev git wget xz-utils
RUN dpkg --add-architecture arm64 && \
  sed -i 's|^deb http://archive.ubuntu.com|deb [arch=amd64] http://archive.ubuntu.com|' /etc/apt/sources.list && \
  sed -i 's|^deb http://security.ubuntu.com|deb [arch=amd64] http://security.ubuntu.com|' /etc/apt/sources.list && \
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
  apt-get update && \
  apt-get install -y libasound2-dev:arm64 libspeexdsp-dev:arm64
RUN wget -q https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz \
  && tar xf gcc-arm-*.tar.xz -C /opt && rm gcc-arm-*.tar.xz
ENV PATH="/opt/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH"
WORKDIR /build
