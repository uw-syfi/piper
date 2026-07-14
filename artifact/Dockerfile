FROM rayproject/ray:2.44.1-py310-cu128

SHELL ["/bin/bash", "-lc"]
USER root
WORKDIR /workspace

ARG TORCHTITAN_COMMIT=b01adfb544b4331ecab090ebdb50b2296cd8eb6a
ARG MEGATRON_REF=26.04-alpha.rc2
ARG TORCHTITAN_TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128
ARG APEX_REF=master

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCHTITAN_CONDA_ENV=torchtitan
ENV MEGATRON_CONDA_ENV=megatron
ENV DEEPSPEED_CONDA_ENV=deepspeed
ENV PIPER_CONDA_ENV=piper
ENV MEGATRON_WORKSPACE=/workspace/Megatron-LM
ENV TORCHTITAN_WORKSPACE=/workspace/torchtitan
ENV PIPER_WORKSPACE=/workspace/piper
ENV ARTIFACT_WORKSPACE=/workspace/artifact

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git wget build-essential ninja-build && \
    . /etc/os-release && \
    arch="$(dpkg --print-architecture)" && \
    case "$arch" in \
        amd64) cuda_arch="x86_64" ;; \
        arm64) cuda_arch="sbsa" ;; \
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac && \
    cuda_repo="https://developer.download.nvidia.com/compute/cuda/repos/${ID}${VERSION_ID//./}/${cuda_arch}" && \
    wget -q "${cuda_repo}/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nsight-systems-12-8 \
        cuda-nvcc-12-8 \
        cuda-cudart-dev-12-8 && \
    rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

RUN mkdir -p /workspace/artifact/patches
COPY artifact/patches /workspace/artifact/patches

RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan && \
    cd /workspace/torchtitan && \
    git checkout "${TORCHTITAN_COMMIT}" && \
    git apply /workspace/artifact/patches/torchtitan-qwen-e2e.patch

RUN git clone https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM && \
    cd /workspace/Megatron-LM && \
    git checkout "${MEGATRON_REF}"

RUN git clone https://github.com/NVIDIA/apex.git /workspace/apex && \
    cd /workspace/apex && \
    git checkout "${APEX_REF}"

RUN conda create -y -n torchtitan python=3.10 && \
    conda run -n torchtitan pip install --no-cache-dir --pre torch --index-url "${TORCHTITAN_TORCH_INDEX_URL}" && \
    conda run -n torchtitan pip install --no-cache-dir smart_open requests && \
    conda run -n torchtitan pip install --no-cache-dir --pre --no-deps torchdata --index-url https://download.pytorch.org/whl/nightly/cpu && \
    conda run -n torchtitan pip install --no-cache-dir -e /workspace/torchtitan && \
    conda run -n torchtitan pip install --no-cache-dir --upgrade fsspec && \
    conda clean -afy

RUN conda create -y -n megatron python=3.12 && \
    conda run -n megatron pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n megatron pip install --no-cache-dir \
        numpy packaging pandas pydantic pyyaml einops importlib-metadata \
        nvdlfw-inspect onnx onnxscript tensorboard && \
    conda run -n megatron pip install --no-cache-dir --no-deps \
        transformer-engine==2.13.0 \
        transformer-engine-torch==2.13.0 \
        transformer-engine-cu12==2.13.0 && \
    cd /workspace/apex && conda run -n megatron pip install --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        . && \
    conda run -n megatron pip install --no-cache-dir -e /workspace/Megatron-LM --no-deps && \
    conda clean -afy

RUN conda create -y -n deepspeed python=3.10 && \
    conda run -n deepspeed pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n deepspeed pip install --no-cache-dir deepspeed && \
    conda clean -afy

COPY pyproject.toml requirements.txt README.md /workspace/piper/
COPY src /workspace/piper/src
COPY examples /workspace/piper/examples
COPY test /workspace/piper/test

RUN conda create -y -n piper python=3.10 && \
    conda run -n piper pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 && \
    cd /workspace/piper && conda run -n piper pip install --no-cache-dir -e . \
        --extra-index-url https://download.pytorch.org/whl/cu128 && \
    conda clean -afy

COPY artifact/e2e_eval.py artifact/run_deepspeed.py artifact/run_megatron.py artifact/sitecustomize.py /workspace/artifact/
COPY artifact/scripts /workspace/artifact/scripts
COPY artifact/README.md /workspace/artifact/README.md

RUN chmod +x /workspace/artifact/scripts/*.sh && \
    mkdir -p /workspace/eval-out /tmp/piper/ray_tmp

WORKDIR /workspace/piper
