FROM rayproject/ray:2.44.1-py310-cu128

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates wget && \
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
    apt-get install -y --no-install-recommends cuda-nsight-systems-12-8 && \
    rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

USER ray

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
