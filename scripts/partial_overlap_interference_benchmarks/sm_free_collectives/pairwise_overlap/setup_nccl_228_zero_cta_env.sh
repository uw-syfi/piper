#!/usr/bin/env bash
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

# Configure the existing NCCL 2.28.3 overlay for Zero-CTA benchmarks.
#
# Source into the current shell:
#   source scripts/partial_overlap_interference_benchmarks/sm_free_collectives/\
#     pairwise_overlap/setup_nccl_228_zero_cta_env.sh
#
# Or run one command in the configured environment:
#   scripts/partial_overlap_interference_benchmarks/sm_free_collectives/\
#     pairwise_overlap/setup_nccl_228_zero_cta_env.sh -- \
#     python scripts/partial_overlap_interference_benchmarks/sm_free_collectives/\
#       pairwise_overlap/run_pairwise_collective_mode.py ...

_piper_zero_cta_prepend_path() {
    local entry="$1"
    local current="$2"
    case ":${current}:" in
        *":${entry}:"*) printf '%s' "${current}" ;;
        *)
            if [[ -n "${current}" ]]; then
                printf '%s:%s' "${entry}" "${current}"
            else
                printf '%s' "${entry}"
            fi
            ;;
    esac
}

_piper_zero_cta_setup() {
    local script_dir repo_root overlay python_dir nccl_lib compat_path python_bin
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return 1
    repo_root="$(cd -- "${script_dir}/../../../.." && pwd -P)" || return 1
    overlay="${PIPER_ZERO_CTA_OVERLAY:-${script_dir}/.runtime/nccl_2_28_overlay}"
    python_dir="${overlay}/python"
    nccl_lib="${python_dir}/nvidia/nccl/lib/libnccl.so.2"

    if [[ -n "${CUDA_COMPAT_PATH:-}" ]]; then
        compat_path="${CUDA_COMPAT_PATH}"
    elif [[ -r /usr/local/cuda-12.8/compat/libcuda.so.1 ]]; then
        compat_path=/usr/local/cuda-12.8/compat
    else
        compat_path="${overlay}/cuda_compat/usr/local/cuda-12.5/compat"
    fi

    if [[ ! -r "${overlay}/manifest.json" ]]; then
        printf 'Missing NCCL 2.28.3 overlay manifest: %s\n' "${overlay}/manifest.json" >&2
        printf 'Install it with install_nccl_228_overlay.py or set PIPER_ZERO_CTA_OVERLAY.\n' >&2
        return 1
    fi
    if [[ ! -r "${nccl_lib}" ]]; then
        printf 'Missing NCCL 2.28.3 library: %s\n' "${nccl_lib}" >&2
        return 1
    fi
    if [[ ! -r "${compat_path}/libcuda.so.1" ]]; then
        printf 'Missing CUDA compatibility library: %s\n' "${compat_path}/libcuda.so.1" >&2
        return 1
    fi

    export PIPER_ZERO_CTA_OVERLAY="${overlay}"
    export CUDA_COMPAT_PATH="${compat_path}"
    export NCCL4PY_OVERLAY="${python_dir}"
    export EXPECTED_NCCL_VERSION=2.28.3
    export PYTHONPATH
    PYTHONPATH="$(_piper_zero_cta_prepend_path "${python_dir}" "${PYTHONPATH:-}")"
    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH="$(_piper_zero_cta_prepend_path "${python_dir}/nvidia/nccl/lib" "${LD_LIBRARY_PATH:-}")"
    LD_LIBRARY_PATH="$(_piper_zero_cta_prepend_path "${compat_path}" "${LD_LIBRARY_PATH}")"

    python_bin="${PYTHON:-$(command -v python)}"
    if [[ -z "${python_bin}" ]]; then
        printf 'No Python interpreter found; set PYTHON to an absolute path.\n' >&2
        return 1
    fi
    "${python_bin}" - "${overlay}" "${compat_path}" <<'PY'
import ctypes
import json
from pathlib import Path
import sys

overlay = Path(sys.argv[1]).resolve()
compat = Path(sys.argv[2]).resolve()
cuda = ctypes.CDLL(str(compat / "libcuda.so.1"), mode=ctypes.RTLD_GLOBAL)
cuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
cuda.cuDriverGetVersion.restype = ctypes.c_int
driver = ctypes.c_int()
status = int(cuda.cuDriverGetVersion(ctypes.byref(driver)))
if status != 0 or driver.value < 12050:
    raise SystemExit(
        f"CUDA Driver API check failed: status={status}, version={driver.value}"
    )

nccl_path = overlay / "python/nvidia/nccl/lib/libnccl.so.2"
ctypes.CDLL(str(nccl_path), mode=ctypes.RTLD_GLOBAL)
import nccl.core as nccl

versions = nccl.get_version()
if str(versions.nccl_version) != "2.28.3":
    raise SystemExit(f"expected NCCL 2.28.3, got {versions.nccl_version}")
zero = nccl.NCCLConfig(cta_policy=nccl.CTAPolicy.Zero)
print(json.dumps({
    "cuda_compat_path": str(compat),
    "cuda_driver_api": driver.value,
    "nccl4py_version": str(versions.nccl4py_version),
    "nccl_library": str(nccl_path),
    "nccl_version": str(versions.nccl_version),
    "zero_cta_config": repr(zero),
}, sort_keys=True), file=sys.stderr)
PY
}

_piper_zero_cta_setup
_piper_zero_cta_status=$?
unset -f _piper_zero_cta_setup _piper_zero_cta_prepend_path

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return "${_piper_zero_cta_status}"
fi
if [[ ${_piper_zero_cta_status} -ne 0 ]]; then
    exit "${_piper_zero_cta_status}"
fi
if [[ "${1:-}" == "--" ]]; then
    shift
fi
if [[ $# -eq 0 ]]; then
    printf 'Environment verified. Source this script or pass a command after --.\n' >&2
    exit 0
fi
exec "$@"
