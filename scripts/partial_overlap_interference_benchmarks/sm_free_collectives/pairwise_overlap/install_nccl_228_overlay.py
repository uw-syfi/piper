#!/usr/bin/env python3
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

"""Install a pinned, non-system NCCL 2.28/CUDA 12.5 compatibility overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


HERE = Path(__file__).resolve().parent
DEFAULT_OVERLAY = HERE / ".runtime/nccl_2_28_overlay"
CUDA_COMPAT_NAME = "cuda-compat-12-5_555.42.06-1_amd64.deb"
CUDA_COMPAT_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/"
    f"x86_64/{CUDA_COMPAT_NAME}"
)
CUDA_COMPAT_SHA256 = "7fd09e4d7089ed8597d068889f40519a41ce1d67d87b57857e14ce4467c18d47"
NCCL_VERSION = "2.28.3"
PYTHON_PACKAGES = (
    f"nvidia-nccl-cu12=={NCCL_VERSION}",
    "nccl4py==0.2.0",
    "cuda-core==1.1.1",
    "backports.strenum==1.3.1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    os.replace(temporary, destination)


def verify_deb(path: Path) -> None:
    observed = sha256(path)
    if observed != CUDA_COMPAT_SHA256:
        raise RuntimeError(
            f"CUDA compatibility package digest mismatch: {observed} != "
            f"{CUDA_COMPAT_SHA256}"
        )


def query_overlay(python_dir: Path, compat_dir: Path) -> dict[str, object]:
    nccl_lib = python_dir / "nvidia/nccl/lib/libnccl.so.2"
    compat_lib = compat_dir / "usr/local/cuda-12.5/compat"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(python_dir), environment.get("PYTHONPATH", "")))
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(compat_lib),
                str(nccl_lib.parent),
                environment.get("LD_LIBRARY_PATH", ""),
            ),
        )
    )
    program = f"""
import ctypes, json, pathlib
cuda = ctypes.CDLL('libcuda.so.1')
driver = ctypes.c_int()
init_rc = int(cuda.cuInit(0))
driver_rc = int(cuda.cuDriverGetVersion(ctypes.byref(driver)))
ctypes.CDLL({str(nccl_lib)!r}, mode=ctypes.RTLD_GLOBAL)
import nccl.core as nccl
mapped = sorted({{
    row.split()[-1]
    for row in pathlib.Path('/proc/self/maps').read_text().splitlines()
    if 'libnccl.so' in row or 'libcuda.so' in row
}})
print(json.dumps({{
    'cu_init_rc': init_rc,
    'cu_driver_get_version_rc': driver_rc,
    'cuda_driver_api': driver.value,
    'nccl4py_version': str(nccl.get_version().nccl4py_version),
    'nccl_version': str(nccl.get_version().nccl_version),
    'zero_cta_config': repr(nccl.NCCLConfig(cta_policy=nccl.CTAPolicy.Zero)),
    'mapped_libraries': mapped,
}}))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", program], env=environment, text=True
    )
    result = json.loads(output.strip())
    if result["cuda_driver_api"] < 12050:
        raise RuntimeError(f"overlay reports CUDA Driver API {result['cuda_driver_api']}")
    if result["nccl_version"] != NCCL_VERSION:
        raise RuntimeError(f"overlay reports NCCL {result['nccl_version']}")
    return result


def install(overlay: Path, supplied_deb: Path | None) -> dict[str, object]:
    overlay = overlay.resolve()
    python_dir = overlay / "python"
    compat_dir = overlay / "cuda_compat"
    download_dir = overlay / "downloads"
    deb = supplied_deb.resolve() if supplied_deb else download_dir / CUDA_COMPAT_NAME
    if not deb.exists():
        download(CUDA_COMPAT_URL, deb)
    verify_deb(deb)

    python_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(python_dir),
            "--no-deps",
            "--upgrade",
            *PYTHON_PACKAGES,
        ]
    )
    compat_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["dpkg-deb", "-x", str(deb), str(compat_dir)])

    query = query_overlay(python_dir, compat_dir)
    nccl_lib = python_dir / "nvidia/nccl/lib/libnccl.so.2"
    compat_lib = compat_dir / "usr/local/cuda-12.5/compat/libcuda.so.555.42.06"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "overlay": str(overlay),
        "python_dir": str(python_dir),
        "compat_lib_dir": str(compat_lib.parent),
        "packages": list(PYTHON_PACKAGES),
        "cuda_compat": {
            "source": CUDA_COMPAT_URL if supplied_deb is None else str(deb),
            "package_sha256": CUDA_COMPAT_SHA256,
            "libcuda_sha256": sha256(compat_lib),
        },
        "nccl_library_sha256": sha256(nccl_lib),
        "verification": query,
    }
    manifest_path = overlay / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--cuda-compat-deb",
        type=Path,
        help="Use a previously downloaded, digest-checked NVIDIA package.",
    )
    args = parser.parse_args()
    print(json.dumps(install(args.overlay, args.cuda_compat_deb), indent=2))


if __name__ == "__main__":
    main()
