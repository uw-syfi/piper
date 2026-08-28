# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

FROM rayproject/ray:2.44.1-py310-cu128

USER root

ARG CUDA_COMPAT_VERSION=570.211.01-0ubuntu1

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
    apt-get install -y --no-install-recommends --allow-downgrades \
        cuda-nsight-systems-12-8 \
        "cuda-compat-12-8=${CUDA_COMPAT_VERSION}" && \
    test -r /usr/local/cuda-12.8/compat/libcuda.so.1 && \
    rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

USER ray

# Prefer CUDA's forward-compatibility user-space driver over the older host
# libcuda injected under /usr/local/nvidia by the container runtime.  The
# launcher repeats this because some NVIDIA runtime configurations prepend
# their own library directories when the container starts.
ENV CUDA_COMPAT_PATH=/usr/local/cuda-12.8/compat
ENV LD_LIBRARY_PATH=${CUDA_COMPAT_PATH}:${LD_LIBRARY_PATH}

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
