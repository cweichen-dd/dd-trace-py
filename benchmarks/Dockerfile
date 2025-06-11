# ---- Stage 1: Build Python 3.13.4 with frame pointers and USDT ----
FROM debian:bookworm-slim AS python-builder

ARG PYTHON_VERSION=3.13.4
ENV PYTHON_HOME=/opt/python
ENV PYTHON_SRC_DIR=/usr/src/python

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libffi-dev \
    libc6-dbg \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    ca-certificates \
    systemtap \
    systemtap-sdt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $PYTHON_SRC_DIR

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz

WORKDIR $PYTHON_SRC_DIR/Python-${PYTHON_VERSION}

RUN ./configure \
        --prefix=$PYTHON_HOME \
        --with-pydebug \
        --with-dtrace \
        CFLAGS="-fno-omit-frame-pointer -fno-optimize-sibling-calls" \
    && make -j$(nproc) && make install

# ---- Stage 2: Final runtime image ----
FROM debian:bookworm-slim AS final

ARG RUST_VERSION=1.87.0
ARG PYPERF_VERSION=2.9.0
ARG VIZTRACER_VERSION=1.0.4
ARG MEMRAY_VERSION=1.17.2
ARG OBJGRAPH_VERSION=3.6.2
ARG PYSPY_VERSION=0.4.0

ENV PYTHONPERFSUPPORT=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHON_HOME="/opt/python"
ENV PERF_TOOLS_PATH="/opt/perf-tools"
ENV FLAMEGRAPH_PATH="/opt/flamegraph"
ENV PATH="${PYTHON_HOME}/bin:${PERF_TOOLS_PATH}/bin:${FLAMEGRAPH_PATH}:$PATH"


# Install Rust toolchain
RUN curl https://sh.rustup.rs -sSf | \
    sh -s -- --default-toolchain ${RUST_VERSION} -y

RUN apt-get update && apt-get install -y --no-install-recommends \
    binutils \
    bpftrace \
    linux-perf \
    libc6-dbg \
    elfutils \
    gdb \
    git \
    file \
    procps \
    curl \
    ca-certificates \
    systemtap \
    systemtap-sdt-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/brendangregg/perf-tools ${PERF_TOOLS_PATH} \
    && git clone --depth 1 https://github.com/brendangregg/Flamegraph ${FLAMEGRAPH_PATH}

COPY --from=python-builder $PYTHON_HOME $PYTHON_HOME
RUN ln -s $PYTHON_HOME/bin/python3.13 $PYTHON_HOME/bin/python \
    && ln -s $PYTHON_HOME/bin/pip3 $PYTHON_HOME/bin/pip \
    && pip install --no-cache-dir -U \
    pip \
    uv \
    "pyperf==${PYPERF_VERSION}" \
    "viztracer==${VIZTRACER_VERSION}" \
    "py-spy==${PYSPY_VERSION}" \
    "memray==${MEMRAY_VERSION}" \
    "objgraph==${OBJGRAPH_VERSION}" \
    "tuna==0.5.11"

COPY bin/pause-stub /usr/local/bin/pause-stub
COPY bin/benchmark-script /usr/local/bin/benchmark-script
COPY bin/run-benchmark /usr/local/bin/run-benchmark
COPY bin/bpftrace-imports-to-importtime /usr/local/bin/bpftrace-imports-to-importtime
COPY share/bpftrace-imports.bt /usr/local/share/bpftrace-imports.bt

COPY bm/ /packages/bm
RUN pip install /packages/bm

RUN mkdir -p /artifacts

WORKDIR /project
VOLUME ["/artifacts"]

ENTRYPOINT ["run-benchmark"]
