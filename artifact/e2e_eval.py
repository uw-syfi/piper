#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TT_ITER_RE = re.compile(r"Final \d+ iter times.*?avg:\s*([\d.]+)\s*s,\s*std:\s*([\d.]+)\s*s")
TT_FINAL_BY_RANK_RE = re.compile(
    r"\[rank(\d+)\].*?Final \d+ iter times.*?avg:\s*([\d.]+)\s*s,\s*std:\s*([\d.]+)\s*s"
)
TT_MEM_RE = re.compile(r"\[rank(\d+)\].*?memory:\s*([\d.]+)GiB")
DS_STEP_RE = re.compile(r"\[Step\s+\d+/\d+\].*?step_time=([\d.]+)s")
DS_MEM_RE = re.compile(r"\[rank(\d+)\]\s+peak_memory_allocated_gb=([\d.]+)\s+peak_memory_reserved_gb=([\d.]+)")
MG_ITER_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")
MG_MEM_RE = re.compile(r"\[Rank\s+(\d+)\].*?max allocated:\s*([\d.]+)\s*\|")
OOM_RE = re.compile(r"(?:cuda\s+out\s+of\s+memory|out\s+of\s+memory|std::bad_alloc|\boom\b)", re.IGNORECASE)

DEFAULT_SYSTEMS = ("torchtitan", "megatron", "deepspeed", "piper")
DEFAULT_SCHEDULE_SWEEP = ("1f1b", "interleaved1f1b", "zerobubble", "dualpipe")
DEFAULT_IMAGE = "piper-e2e-artifact:latest"
DEFAULT_CONTAINER = "piper_artifact"
DEFAULT_LOCAL_TORCHTITAN_ASSETS = Path("/m-coriander/coriander/mfris/torchtitan/assets/hf")
CONTAINER_OUT = Path("/workspace/eval-out")
CONTAINER_ARTIFACT = Path("/workspace/artifact")
CONTAINER_TORCHTITAN_HF_ASSETS = Path("/workspace/torchtitan/assets/hf")

RUNTIME_SCHEDULES = {
    "1f1b": "1F1B",
    "interleaved1f1b": "Interleaved1F1B",
    "zerobubble": "InterleavedZeroBubble",
    "dualpipe": "DualPipeV",
}
PIPER_SCHEDULES = {
    "1f1b": "1f1b",
    "interleaved1f1b": "interleaved_1f1b",
    "zerobubble": "zerobubble",
    "dualpipe": "dualpipev",
}
PIPER_ZERO_STAGES = {"zero0": 0, "zero1": 1, "zero2": 2, "zero3": 3}
SYSTEM_COLORS = {
    "torchtitan": "#4C72B0",
    "megatron": "#55A868",
    "deepspeed": "#C44E52",
    "piper": "#8172B3",
}


def remapped_cuda_visible_devices(host_devices: str) -> str:
    devices = [device.strip() for device in host_devices.split(",") if device.strip()]
    return ",".join(str(index) for index, _device in enumerate(devices)) or "0"


def docker_gpus_request(cuda_visible_devices: str | None) -> str:
    if cuda_visible_devices is None:
        return "all"
    return f'"device={cuda_visible_devices}"'


@dataclass(frozen=True)
class Experiment:
    system: str
    sweep: str
    config: str
    pp: int
    dp: int
    ep: int
    zero_level: str
    schedule: str
    mb_size: int
    seq_len: int
    gradient_accumulation: bool = True
    bucket_size_mb: float | None = None
    ar_a2a_same_stream: bool = False
    overlap_chunks: bool = False


@dataclass
class Result:
    system: str
    sweep: str
    config: str
    pp: int
    dp: int
    ep: int
    zero_level: str
    schedule: str
    mb_size: int
    seq_len: int
    local_batch_size: int
    global_batch_size: int
    nnode: int
    ngpu: int
    log_path: str
    metrics_path: str
    returncode: int
    status: str
    iter_time_mean: float | None
    iter_time_stddev: float | None
    peak_memory_gb_by_rank: str
    failure_reason: str


def normalize_schedule(name: str) -> str:
    mapping = {
        "1f1b": "1f1b",
        "interleaved1f1b": "interleaved1f1b",
        "interleaved-1f1b": "interleaved1f1b",
        "interleaved_1f1b": "interleaved1f1b",
        "zerobubble": "zerobubble",
        "zero-bubble": "zerobubble",
        "interleavedzerobubble": "zerobubble",
        "interleaved-zero-bubble": "zerobubble",
        "dualpipe": "dualpipe",
        "dualpipev": "dualpipe",
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported schedule: {name}")
    return mapping[key]


def build_experiments(
    *,
    systems: Iterable[str],
    schedule_sweep: Iterable[str],
    seq_len: int,
    enabled_sweeps: set[str],
    gradient_accumulation: bool,
    ar_a2a_same_stream: bool,
    overlap_chunks: bool,
    bucket_size_mb: float | None,
) -> list[Experiment]:
    experiments: list[Experiment] = []
    scalability_defaults = {
        "config": "qwen3_9b",
        "ep": 1,
        "zero_level": "zero0",
        "schedule": "1f1b",
        "mb_size": 4,
    }
    zero_defaults = {
        "config": "qwen3_1b",
        "pp": 8,
        "dp": 2,
        "ep": 1,
        "schedule": "1f1b",
    }
    zero_levels_by_system = {
        "torchtitan": ("zero1", "zero2", "zero3"),
        "megatron": ("zero1",),
        "deepspeed": ("zero1",),
        "piper": ("zero1", "zero2", "zero3"),
    }
    zero_sweep_values = (16, 24, 32, 34, 36, 38, 40)
    schedule_defaults = {
        "config": "qwen3_9b",
        "pp": 2,
        "dp": 2,
        "ep": 2,
        "zero_level": "zero1",
        "mb_size": 4,
    }
    local_defaults = {
        "config": "qwen3_1b",
        "pp": 1,
        "dp": 1,
        "ep": 1,
        "zero_level": "zero1",
        "schedule": "1f1b",
        "mb_size": 8,
    }
    supported_schedules_by_system = {
        "torchtitan": {"1f1b", "interleaved1f1b", "dualpipe"},
        "megatron": {"1f1b", "interleaved1f1b"},
        "deepspeed": {"1f1b"},
        "piper": {"1f1b", "interleaved1f1b", "dualpipe"},
    }

    for system in systems:
        if "scalability" in enabled_sweeps:
            for pp in (4, 8):
                for dp in (1, 2, 4):
                    experiments.append(
                        Experiment(
                            system=system,
                            sweep="scalability",
                            config=str(scalability_defaults["config"]),
                            pp=pp,
                            dp=dp,
                            ep=int(scalability_defaults["ep"]),
                            zero_level=str(scalability_defaults["zero_level"]),
                            schedule=str(scalability_defaults["schedule"]),
                            mb_size=int(scalability_defaults["mb_size"]),
                            seq_len=seq_len,
                            gradient_accumulation=gradient_accumulation,
                            bucket_size_mb=bucket_size_mb,
                            ar_a2a_same_stream=ar_a2a_same_stream,
                            overlap_chunks=overlap_chunks,
                        )
                    )

        if "zero" in enabled_sweeps:
            for zero_level in zero_levels_by_system[system]:
                for sweep_value in zero_sweep_values:
                    experiments.append(
                        Experiment(
                            system=system,
                            sweep="zero",
                            config=str(zero_defaults["config"]),
                            pp=int(zero_defaults["pp"]),
                            dp=int(zero_defaults["dp"]),
                            ep=int(zero_defaults["ep"]),
                            zero_level=zero_level,
                            schedule=str(zero_defaults["schedule"]),
                            mb_size=sweep_value,
                            seq_len=seq_len,
                            gradient_accumulation=gradient_accumulation,
                            bucket_size_mb=bucket_size_mb,
                            ar_a2a_same_stream=ar_a2a_same_stream,
                            overlap_chunks=overlap_chunks,
                        )
                    )

        if "schedule" in enabled_sweeps:
            for schedule in schedule_sweep:
                if schedule not in supported_schedules_by_system[system]:
                    continue
                experiments.append(
                    Experiment(
                        system=system,
                        sweep="schedule",
                        config=str(schedule_defaults["config"]),
                        pp=int(schedule_defaults["pp"]),
                        dp=int(schedule_defaults["dp"]),
                        ep=int(schedule_defaults["ep"]),
                        zero_level=str(schedule_defaults["zero_level"]),
                        schedule=schedule,
                        mb_size=int(schedule_defaults["mb_size"]),
                        seq_len=seq_len,
                        gradient_accumulation=gradient_accumulation,
                        bucket_size_mb=bucket_size_mb,
                        ar_a2a_same_stream=ar_a2a_same_stream,
                        overlap_chunks=overlap_chunks,
                    )
                )

        if "local" in enabled_sweeps:
            experiments.append(
                Experiment(
                    system=system,
                    sweep="local",
                    config=str(local_defaults["config"]),
                    pp=int(local_defaults["pp"]),
                    dp=int(local_defaults["dp"]),
                    ep=int(local_defaults["ep"]),
                    zero_level=str(local_defaults["zero_level"]),
                    schedule=str(local_defaults["schedule"]),
                    mb_size=int(local_defaults["mb_size"]),
                    seq_len=seq_len,
                    gradient_accumulation=gradient_accumulation,
                    bucket_size_mb=bucket_size_mb,
                    ar_a2a_same_stream=ar_a2a_same_stream,
                    overlap_chunks=overlap_chunks,
                )
            )
    return experiments


def local_batch_size(exp: Experiment) -> int:
    if exp.sweep == "local":
        return exp.mb_size
    return exp.pp * 2 * exp.mb_size


def global_batch_size(exp: Experiment) -> int:
    return local_batch_size(exp) * exp.dp


def experiment_label(exp: Experiment) -> str:
    if exp.sweep == "scalability":
        return f"pp={exp.pp}, dp={exp.dp}"
    if exp.sweep == "zero":
        return f"zero={exp.zero_level}, mb={exp.mb_size}"
    if exp.sweep == "local":
        return "local"
    return exp.schedule


def piper_model_name(config: str) -> str:
    return {"qwen3_1b": "1B", "qwen3_9b": "9B"}[config]


def piper_num_mbs(exp: Experiment) -> int:
    if exp.sweep == "local":
        return 1
    return exp.pp * 2


def slug(exp: Experiment, *, nsight: bool = False) -> str:
    bucket_part = f"-bucket{exp.bucket_size_mb:g}" if exp.bucket_size_mb is not None else ""
    nsight_part = "-nsight1" if nsight else ""
    return (
        f"{exp.system}-{exp.sweep}-qwen{exp.config.removeprefix('qwen3_')}-"
        f"sched_{exp.schedule}-pp{exp.pp}-dp{exp.dp}-ep{exp.ep}-"
        f"{exp.zero_level}{bucket_part}{nsight_part}-bs{exp.mb_size}-"
        f"sl{exp.seq_len}-mbs{piper_num_mbs(exp)}"
    )


def torchtitan_hf_assets_path(config: str) -> str | None:
    return {
        "qwen3_1b": "/workspace/torchtitan/assets/hf/Qwen3-0.6B",
        "qwen3_9b": "/workspace/torchtitan/assets/hf/Qwen3-8B",
    }.get(config)


def backend_layout(exp: Experiment, backend_name: str) -> tuple[int, int]:
    if backend_name == "local":
        return 1, exp.pp * exp.dp
    return exp.dp, exp.pp


def in_container_command(exp: Experiment, args: argparse.Namespace, metrics_container_path: str | None = None) -> list[str]:
    nnode = "{nnode}"
    ngpu = "{ngpu}"
    node_rank = "{node_rank}"
    master_addr = "{master_addr}"

    if exp.system == "torchtitan":
        nnode_value, ngpu_value = backend_layout(exp, args.backend)
        if exp.ep > 1:
            dp_replicate_degree = 1
            dp_shard_degree = exp.ep
        else:
            dp_replicate_degree = exp.dp if exp.zero_level == "zero1" else 1
            dp_shard_degree = 1 if exp.zero_level == "zero1" else exp.dp
        tt_args = [
            "--parallelism.pipeline_parallel_degree", str(exp.pp),
            "--parallelism.expert_parallel_degree", str(exp.ep),
            "--parallelism.pipeline_parallel_schedule", RUNTIME_SCHEDULES[exp.schedule],
            "--parallelism.pipeline_parallel_microbatch_size", str(exp.mb_size),
            "--parallelism.data_parallel_replicate_degree", str(dp_replicate_degree),
            "--parallelism.data_parallel_shard_degree", str(dp_shard_degree),
            "--training.seq_len", str(exp.seq_len),
            "--training.local_batch_size", str(local_batch_size(exp)),
            "--training.global_batch_size", str(global_batch_size(exp)),
        ]
        hf_assets_path = torchtitan_hf_assets_path(exp.config)
        if hf_assets_path is not None:
            tt_args.extend(["--hf_assets_path", hf_assets_path])
        if exp.zero_level == "zero2":
            tt_args.extend(["--parallelism.fsdp_reshard_after_forward", "never"])
        elif exp.zero_level == "zero3":
            tt_args.extend(["--parallelism.fsdp_reshard_after_forward", "always"])
        if exp.schedule == "dualpipe" or not args.torchtitan_compile:
            tt_args.append("--compile.no-enable")
        command = [
            "/workspace/artifact/scripts/run_torchtitan.sh",
            "--nnode", nnode,
            "--ngpu", ngpu,
            "--node-rank", node_rank,
            "--master-addr", master_addr,
            "--master-port", "29500",
            "--module", "qwen3",
            "--config", exp.config,
            "--log-rank", ",".join(str(i) for i in range(min(nnode_value * ngpu_value, int(args.max_log_ranks)))),
        ]
        if args.nsight:
            command.append("--nsight")
        if args.torchtitan_use_bmm_experts:
            command.append("--use-bmm-experts")
        command.extend(["--", *tt_args])
        return command

    if exp.system == "megatron":
        dp_megatron = exp.dp // exp.ep if exp.ep > 1 else exp.dp
        command = [
            "/workspace/artifact/scripts/run_megatron.sh",
            "--nnode", nnode,
            "--ngpu", ngpu,
            "--node-rank", node_rank,
            "--master-addr", master_addr,
            "--master-port", "29500",
            "--model", exp.config,
            "--pp", str(exp.pp),
            "--dp", str(dp_megatron),
            "--ep", str(exp.ep),
        ]
        if args.nsight:
            command.append("--nsight")
        command.extend([
            "--",
            "--micro-bs", str(exp.mb_size),
            "--global-bs", str(global_batch_size(exp)),
            "--seq-length", str(exp.seq_len),
            "--schedule", exp.schedule,
            "--zero-level", exp.zero_level,
            "--train-iters", str(args.baseline_steps),
        ])
        return command

    if exp.system == "deepspeed":
        return [
            "/workspace/artifact/scripts/run_deepspeed.sh",
            "--nnode", nnode,
            "--ngpu", ngpu,
            "--node-rank", node_rank,
            "--master-addr", master_addr,
            "--master-port", "29501",
            "--model", exp.config,
            "--",
            "--pp", str(exp.pp),
            "--dp", str(exp.dp),
            "--ep", str(exp.ep),
            "--micro-bs", str(exp.mb_size),
            "--global-bs", str(global_batch_size(exp)),
            "--seq-len", str(exp.seq_len),
            "--schedule", exp.schedule,
            "--zero-stage", {"zero1": "1", "zero2": "2", "zero3": "3"}[exp.zero_level],
            "--steps", str(args.baseline_steps),
        ]

    command = [
        "/workspace/artifact/scripts/run_piper.sh",
        "--model", piper_model_name(exp.config),
        "--schedule", PIPER_SCHEDULES[exp.schedule],
        "--pp", str(exp.pp),
        "--dp", str(exp.dp),
        "--zero-stage", str(PIPER_ZERO_STAGES[exp.zero_level]),
        "--batch-size", str(exp.mb_size),
        "--seq-len", str(exp.seq_len),
        "--mbs", str(piper_num_mbs(exp)),
        "--warmup", str(args.piper_warmup),
        "--iters", str(args.piper_iters),
        "--iteration-sleep", f"{args.piper_iteration_sleep:g}",
        "--port", str(args.piper_ray_port),
        "--temp-dir", "/tmp/piper/ray_tmp",
        "--use-inductor" if args.piper_use_inductor and exp.sweep != "schedule" else "--no-use-inductor",
    ]
    if exp.ep > 1:
        command.append("--ep")
    if exp.bucket_size_mb is not None:
        command.extend(["--bucket-size", f"{exp.bucket_size_mb:g}"])
    if args.nsight:
        command.append("--nsight")
    if metrics_container_path:
        command.extend(["--metrics-out", metrics_container_path])
    if args.backend == "aws":
        command.extend(["--address", "{master_addr}"])
    return command


def render_command(command: Sequence[str], *, nnode: int, ngpu: int, node_rank: int, master_addr: str) -> list[str]:
    values = {
        "nnode": str(nnode),
        "ngpu": str(ngpu),
        "node_rank": str(node_rank),
        "master_addr": master_addr,
    }
    return [part.format(**values) for part in command]


class Backend:
    name: str

    def layout(self, exp: Experiment) -> tuple[int, int]:
        return backend_layout(exp, self.name)

    def run(self, exp: Experiment, command: Sequence[str], log_path: Path) -> int:
        raise NotImplementedError

    def prepare_piper(self, args: argparse.Namespace) -> None:
        return None

    def cleanup_after_experiment(self, exp: Experiment) -> None:
        return None

    def fetch_file(self, container_path: str, destination: Path) -> bool:
        return Path(container_path).is_file()

    def dry_run_lines(self, exp: Experiment, command: Sequence[str]) -> list[str]:
        raise NotImplementedError


class LocalBackend(Backend):
    name = "local"

    def __init__(
        self,
        image: str,
        out_dir: Path,
        extra_docker_args: Sequence[str],
        cuda_visible_devices: str | None,
        torchtitan_assets_path: Path | None,
    ):
        self.image = image
        self.out_dir = out_dir
        self.extra_docker_args = list(extra_docker_args)
        self.cuda_visible_devices = cuda_visible_devices
        self.container_cuda_visible_devices = (
            remapped_cuda_visible_devices(cuda_visible_devices)
            if cuda_visible_devices is not None
            else None
        )
        self.artifact_src_dir = Path(__file__).resolve().parent
        self.workspace_dir = self.artifact_src_dir.parent
        self.torchtitan_assets_path = torchtitan_assets_path if torchtitan_assets_path and torchtitan_assets_path.is_dir() else None

    def run(self, exp: Experiment, command: Sequence[str], log_path: Path) -> int:
        nnode, ngpu = self.layout(exp)
        rendered = render_command(command, nnode=nnode, ngpu=ngpu, node_rank=0, master_addr="127.0.0.1")
        gpus = docker_gpus_request(self.cuda_visible_devices)
        docker_cmd = [
            "docker", "run", "--rm",
            "--gpus", gpus,
            "--network", "host",
            "--ipc", "host",
            "--shm-size", "32g",
            "-v", f"{self.out_dir}:{CONTAINER_OUT}",
            "-v", f"{self.artifact_src_dir}:{CONTAINER_ARTIFACT}:ro",
            "-v", f"{self.workspace_dir / 'src'}:/workspace/piper/src:ro",
            "-v", f"{self.workspace_dir / 'examples'}:/workspace/piper/examples:ro",
            "-v", f"{self.workspace_dir / 'test'}:/workspace/piper/test:ro",
        ]
        if self.torchtitan_assets_path is not None:
            docker_cmd.extend(["-v", f"{self.torchtitan_assets_path}:{CONTAINER_TORCHTITAN_HF_ASSETS}:ro"])
        if self.cuda_visible_devices is not None:
            docker_cmd.extend([
                "-e", f"NVIDIA_VISIBLE_DEVICES={self.cuda_visible_devices}",
                "-e", f"CUDA_VISIBLE_DEVICES={self.container_cuda_visible_devices}",
            ])
        docker_cmd.extend([
            *self.extra_docker_args,
            self.image,
            "bash", "-lc", shlex.join(rendered),
        ])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(docker_cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        return proc.returncode

    def dry_run_lines(self, exp: Experiment, command: Sequence[str]) -> list[str]:
        nnode, ngpu = self.layout(exp)
        rendered = render_command(command, nnode=nnode, ngpu=ngpu, node_rank=0, master_addr="127.0.0.1")
        cuda_env = (
            f"-e CUDA_VISIBLE_DEVICES={shlex.quote(str(self.container_cuda_visible_devices))} "
            if self.cuda_visible_devices is not None
            else ""
        )
        extra_args = " ".join(shlex.quote(arg) for arg in self.extra_docker_args)
        extra_args = f"{extra_args} " if extra_args else ""
        gpus = docker_gpus_request(self.cuda_visible_devices)
        nvidia_env = (
            f"-e NVIDIA_VISIBLE_DEVICES={shlex.quote(self.cuda_visible_devices)} "
            if self.cuda_visible_devices is not None
            else ""
        )
        return [
            f"docker run --rm --gpus {shlex.quote(gpus)} --network host --ipc host "
            f"{nvidia_env}"
            f"{cuda_env}"
            f"-v {shlex.quote(str(self.out_dir))}:{CONTAINER_OUT} "
            f"-v {shlex.quote(str(self.artifact_src_dir))}:{CONTAINER_ARTIFACT}:ro "
            f"-v {shlex.quote(str(self.workspace_dir / 'src'))}:/workspace/piper/src:ro "
            f"-v {shlex.quote(str(self.workspace_dir / 'examples'))}:/workspace/piper/examples:ro "
            f"-v {shlex.quote(str(self.workspace_dir / 'test'))}:/workspace/piper/test:ro "
            f"{self._torchtitan_assets_mount_for_dry_run()}"
            f"{extra_args}{shlex.quote(self.image)} bash -lc {shlex.quote(shlex.join(rendered))}"
        ]

    def _torchtitan_assets_mount_for_dry_run(self) -> str:
        if self.torchtitan_assets_path is None:
            return ""
        return f"-v {shlex.quote(str(self.torchtitan_assets_path))}:{CONTAINER_TORCHTITAN_HF_ASSETS}:ro "


class AwsSshBackend(Backend):
    name = "aws"

    def __init__(
        self,
        *,
        image: str,
        container: str,
        start_containers: bool,
        ray_port: int,
    ):
        self.image = image
        self.container = container
        self.start_containers = start_containers
        self.ray_port = ray_port
        self.ssh_key = require_env("SSH_KEY")
        self.head_public_ip = require_env("HEAD_PUBLIC_IP")
        self.head_private_ip = require_env("HEAD_PRIVATE_IP")
        self.workers = workers_from_env()
        if self.start_containers:
            self._start_containers()

    def _ssh(self, target: str, remote_command: str, *, worker: bool = False) -> list[str]:
        cmd = [
            "ssh",
            "-i", self.ssh_key,
            "-o", "StrictHostKeyChecking=no",
        ]
        if worker:
            proxy = (
                "ProxyCommand="
                f"ssh -i {self.ssh_key} -o StrictHostKeyChecking=no "
                f"-W %h:%p ubuntu@{self.head_public_ip}"
            )
            cmd.extend(["-o", proxy])
        cmd.extend([f"ubuntu@{target}", remote_command])
        return cmd

    def _node_specs(self, nnode: int) -> list[tuple[str, str, bool]]:
        if nnode > 1 and len(self.workers) < nnode - 1:
            raise RuntimeError(f"Need {nnode - 1} WORKER*_PRIVATE_IP values, found {len(self.workers)}")
        nodes = [("head", self.head_public_ip, False)]
        nodes.extend((f"worker{i}", ip, True) for i, ip in enumerate(self.workers[: nnode - 1], start=1))
        return nodes

    def _docker_exec_command(self, rendered: Sequence[str], *, node_rank: int) -> str:
        inner = shlex.join(rendered)
        docker = [
            "docker", "exec",
            "-e", f"NODE_RANK={node_rank}",
            "-e", "NCCL_SOCKET_IFNAME=ens32",
            "-e", "GLOO_SOCKET_IFNAME=ens32",
            self.container,
            "bash", "-lc", inner,
        ]
        return shlex.join(docker)

    def _start_containers(self) -> None:
        remote = (
            f"docker rm -f {shlex.quote(self.container)} >/dev/null 2>&1 || true; "
            f"docker run -d --name {shlex.quote(self.container)} --gpus all "
            "--network host --ipc host --shm-size 32g "
            f"{shlex.quote(self.image)} sleep infinity"
        )
        for label, target, worker in self._node_specs(len(self.workers) + 1):
            print(f"[aws] starting container on {label}", file=sys.stderr)
            subprocess.run(self._ssh(target, remote, worker=worker), check=True)

    def prepare_piper(self, args: argparse.Namespace) -> None:
        nnode = max(1, len(self.workers) + 1)
        stop = f"docker exec {shlex.quote(self.container)} bash -lc 'conda run -n piper ray stop -f >/dev/null 2>&1 || true'"
        for _label, target, worker in self._node_specs(nnode):
            subprocess.run(self._ssh(target, stop, worker=worker), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        head_cmd = (
            f"docker exec {shlex.quote(self.container)} bash -lc "
            + shlex.quote(
                "conda run -n piper ray start --head "
                f"--node-ip-address={self.head_private_ip} "
                f"--port={self.ray_port} "
                "--disable-usage-stats "
                "--temp-dir=/tmp/piper/ray_tmp"
            )
        )
        subprocess.run(self._ssh(self.head_public_ip, head_cmd), check=True)
        for _label, target, worker in self._node_specs(nnode)[1:]:
            worker_cmd = (
                f"docker exec {shlex.quote(self.container)} bash -lc "
                + shlex.quote(
                    "conda run -n piper ray start "
                    f"--address={self.head_private_ip}:{self.ray_port} "
                    "--disable-usage-stats "
                    "--temp-dir=/tmp/piper/ray_tmp"
                )
            )
            subprocess.run(self._ssh(target, worker_cmd, worker=worker), check=True)

    def run(self, exp: Experiment, command: Sequence[str], log_path: Path) -> int:
        nnode, ngpu = self.layout(exp)
        nodes = self._node_specs(nnode)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if exp.system == "piper":
            rendered = render_command(
                command,
                nnode=nnode,
                ngpu=ngpu,
                node_rank=0,
                master_addr=self.head_private_ip,
            )
            remote = self._docker_exec_command(rendered, node_rank=0)
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(self._ssh(self.head_public_ip, remote), stdout=log_file, stderr=subprocess.STDOUT, text=True)
            return proc.returncode

        with tempfile.TemporaryDirectory(prefix="piper-e2e-node-logs.") as tmp_dir:
            tmp_path = Path(tmp_dir)
            procs: list[tuple[str, subprocess.Popen[str], Path]] = []
            for node_rank, (label, target, worker) in enumerate(nodes):
                rendered = render_command(
                    command,
                    nnode=nnode,
                    ngpu=ngpu,
                    node_rank=node_rank,
                    master_addr=self.head_private_ip,
                )
                remote = self._docker_exec_command(rendered, node_rank=node_rank)
                node_log = tmp_path / f"{log_path.stem}.{label}.log"
                handle = node_log.open("w", encoding="utf-8")
                proc = subprocess.Popen(self._ssh(target, remote, worker=worker), stdout=handle, stderr=subprocess.STDOUT, text=True)
                handle.close()
                procs.append((label, proc, node_log))

            returncodes = []
            for _label, proc, _node_log in procs:
                returncodes.append(proc.wait())
            with log_path.open("w", encoding="utf-8") as combined:
                for label, _proc, node_log in procs:
                    combined.write(f"===== {label} =====\n")
                    if node_log.exists():
                        combined.write(node_log.read_text(encoding="utf-8", errors="replace"))
                        combined.write("\n")
            return 0 if all(code == 0 for code in returncodes) else next(code for code in returncodes if code != 0)

    def fetch_file(self, container_path: str, destination: Path) -> bool:
        remote = f"docker exec {shlex.quote(self.container)} bash -lc {shlex.quote('test -f ' + shlex.quote(container_path) + ' && cat ' + shlex.quote(container_path))}"
        result = subprocess.run(self._ssh(self.head_public_ip, remote), check=False, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.stdout, encoding="utf-8")
        return True

    def cleanup_after_experiment(self, exp: Experiment) -> None:
        if exp.system != "piper":
            return
        stop = f"docker exec {shlex.quote(self.container)} bash -lc 'conda run -n piper ray stop -f >/dev/null 2>&1 || true'"
        for _label, target, worker in self._node_specs(max(1, len(self.workers) + 1)):
            subprocess.run(self._ssh(target, stop, worker=worker), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def dry_run_lines(self, exp: Experiment, command: Sequence[str]) -> list[str]:
        nnode, ngpu = self.layout(exp)
        nodes = self._node_specs(nnode)
        lines = []
        if exp.system == "piper":
            rendered = render_command(command, nnode=nnode, ngpu=ngpu, node_rank=0, master_addr=self.head_private_ip)
            lines.append(self._docker_exec_command(rendered, node_rank=0))
            return lines
        for node_rank, (label, _target, _worker) in enumerate(nodes):
            rendered = render_command(command, nnode=nnode, ngpu=ngpu, node_rank=node_rank, master_addr=self.head_private_ip)
            lines.append(f"{label}: {self._docker_exec_command(rendered, node_rank=node_rank)}")
        return lines


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def workers_from_env() -> list[str]:
    workers: list[str] = []
    index = 1
    while True:
        value = os.environ.get(f"WORKER{index}_PRIVATE_IP")
        if not value:
            break
        workers.append(value)
        index += 1
    return workers


def read_text_lossy(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").replace("\0", "")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_log(system: str, log_path: Path, metrics_path: Path | None = None) -> tuple[float | None, float | None, str, bool, str]:
    text = strip_ansi(read_text_lossy(log_path))
    is_oom = OOM_RE.search(text) is not None
    reason = "oom" if is_oom else ""

    if system == "torchtitan":
        peak: dict[int, float] = {}
        iter_by_rank: dict[int, tuple[float, float]] = {}
        last_iter: tuple[float, float] | None = None
        for line in text.splitlines():
            m = TT_FINAL_BY_RANK_RE.search(line)
            if m:
                iter_by_rank[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
                last_iter = (float(m.group(2)), float(m.group(3)))
            m = TT_ITER_RE.search(line)
            if m:
                last_iter = (float(m.group(1)), float(m.group(2)))
            m = TT_MEM_RE.search(line)
            if m:
                peak[int(m.group(1))] = max(peak.get(int(m.group(1)), 0.0), float(m.group(2)))
        iter_mean, iter_std = iter_by_rank.get(0, last_iter or (None, None))
        peak_str = "/".join(f"{peak[rank]:.2f}" for rank in sorted(peak)) if peak else ""
        return iter_mean, iter_std, peak_str, is_oom, reason

    if system == "deepspeed":
        step_times = [float(m.group(1)) for m in DS_STEP_RE.finditer(text)][-5:]
        peak = {int(m.group(1)): float(m.group(2)) for m in DS_MEM_RE.finditer(text)}
        peak_str = "/".join(f"{peak[rank]:.2f}" for rank in sorted(peak)) if peak else ""
        if not step_times:
            return None, None, peak_str, is_oom, reason
        mean = sum(step_times) / len(step_times)
        return mean, pstdev(step_times, mean), peak_str, is_oom, reason

    if system == "megatron":
        iter_times_s = [float(m.group(1)) / 1000.0 for m in MG_ITER_RE.finditer(text)][-5:]
        peak = {int(m.group(1)): float(m.group(2)) / 1024.0 for m in MG_MEM_RE.finditer(text)}
        peak_str = "/".join(f"{peak[rank]:.2f}" for rank in sorted(peak)) if peak else ""
        if not iter_times_s:
            return None, None, peak_str, is_oom, reason
        mean = sum(iter_times_s) / len(iter_times_s)
        return mean, pstdev(iter_times_s, mean), peak_str, is_oom, reason

    if system == "piper" and metrics_path is not None and metrics_path.is_file():
        rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
        times: list[tuple[float, float, int]] = []
        peak_values: list[str] = []
        for row in rows:
            try:
                mean = float(row.get("iter_time_mean_s") or "")
            except ValueError:
                continue
            try:
                std = float(row.get("iter_time_std_s") or 0.0)
            except ValueError:
                std = 0.0
            try:
                samples = int(float(row.get("samples") or 1))
            except ValueError:
                samples = 1
            times.append((mean, std, samples))
            for key, value in row.items():
                if key.startswith("peak_memory_pp") and value:
                    try:
                        peak_values.append(f"{key.removeprefix('peak_memory_pp').removesuffix('_gb')}:{float(value):.3f}")
                    except ValueError:
                        pass
        total_samples = sum(samples for _mean, _std, samples in times)
        if total_samples:
            mean = sum(mean * samples for mean, _std, samples in times) / total_samples
            std = sum(std * samples for _mean, std, samples in times) / total_samples
            return mean, std, ";".join(peak_values), is_oom, reason
    return None, None, "", is_oom, reason


def pstdev(values: Sequence[float], mean: float) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def write_csv(path: Path, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Result.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def row_tps(row: Result) -> float | None:
    if row.iter_time_mean is None or row.iter_time_mean <= 0:
        return None
    return row.global_batch_size * row.seq_len / row.iter_time_mean


def result_label(row: Result) -> str:
    if row.sweep == "scalability":
        return f"pp={row.pp},dp={row.dp}"
    if row.sweep == "zero":
        return f"{row.zero_level},mb={row.mb_size}"
    if row.sweep == "local":
        return "local"
    return row.schedule


def write_outputs(base_out: Path, rows: list[Result]) -> None:
    write_csv(base_out / "results.csv", rows)
    for sweep in ("scalability", "zero", "schedule", "local"):
        sweep_rows = [row for row in rows if row.sweep == sweep]
        if sweep_rows:
            save_plot(sweep_rows, base_out / f"{sweep}.png", title=f"Qwen3 {sweep} throughput")


def save_plot(rows: list[Result], output_path: Path, *, title: str) -> None:
    ok_rows = [row for row in rows if row.status == "ok" and row_tps(row) is not None]
    if not ok_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional for dry environments
        print(f"Skipping plot {output_path}: {exc}", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row.system}\n{result_label(row)}" for row in ok_rows]
    values = [float(row_tps(row) or 0.0) for row in ok_rows]
    colors = [SYSTEM_COLORS.get(row.system, "#4C72B0") for row in ok_rows]
    fig, ax = plt.subplots(figsize=(max(10, len(ok_rows) * 0.7), 5))
    ax.bar(range(len(ok_rows)), values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Tokens / second")
    ax.set_xticks(range(len(ok_rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Piper paper Qwen e2e artifact.")
    parser.add_argument("--backend", choices=("local", "aws"), default="local")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--aws-start-containers", action="store_true")
    parser.add_argument("--systems", nargs="+", choices=list(DEFAULT_SYSTEMS), default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--sweeps", nargs="+", choices=("scalability", "zero", "schedule", "local"), default=["scalability", "zero", "schedule"])
    parser.add_argument("--schedule-sweep", nargs="+", default=list(DEFAULT_SCHEDULE_SWEEP))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nsight", action="store_true")
    parser.add_argument("--bucket-size-mb", type=float, default=None)
    parser.add_argument("--gradient-accumulation", dest="gradient_accumulation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ar-a2a-same-stream", dest="ar_a2a_same_stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlap-chunks", dest="overlap_chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--piper-warmup", type=int, default=3)
    parser.add_argument("--piper-iters", type=int, default=10)
    parser.add_argument("--piper-iteration-sleep", type=float, default=0.0)
    parser.add_argument("--piper-use-inductor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torchtitan-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torchtitan-use-bmm-experts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--piper-ray-port", type=int, default=6379)
    parser.add_argument("--baseline-steps", type=int, default=8)
    parser.add_argument("--max-log-ranks", type=int, default=8)
    parser.add_argument(
        "--local-cuda-visible-devices",
        "--cuda-visible-devices",
        dest="local_cuda_visible_devices",
        default=None,
        help=(
            "Restrict local Docker runs to these host GPU IDs. Inside the filtered "
            "container, CUDA_VISIBLE_DEVICES is remapped to logical IDs."
        ),
    )
    default_torchtitan_assets = os.environ.get("TORCHTITAN_HF_ASSETS")
    if default_torchtitan_assets is None and DEFAULT_LOCAL_TORCHTITAN_ASSETS.is_dir():
        default_torchtitan_assets = str(DEFAULT_LOCAL_TORCHTITAN_ASSETS)
    parser.add_argument(
        "--local-torchtitan-assets-path",
        default=default_torchtitan_assets,
        help="Host path mounted read-only to /workspace/torchtitan/assets/hf for local TorchTitan runs.",
    )
    parser.add_argument("--docker-arg", action="append", default=[], help="Extra argument passed to docker run in local mode.")
    return parser.parse_args()


def make_backend(args: argparse.Namespace, base_out: Path) -> Backend:
    if args.backend == "local":
        torchtitan_assets_path = Path(args.local_torchtitan_assets_path).resolve() if args.local_torchtitan_assets_path else None
        return LocalBackend(
            args.image,
            base_out,
            args.docker_arg,
            args.local_cuda_visible_devices,
            torchtitan_assets_path,
        )
    return AwsSshBackend(
        image=args.image,
        container=args.container,
        start_containers=args.aws_start_containers,
        ray_port=args.piper_ray_port,
    )


def main() -> int:
    args = parse_args()
    schedule_sweep = [normalize_schedule(item) for item in args.schedule_sweep]
    base_out = Path(args.out_dir).resolve() if args.out_dir else (Path("artifact") / "out" / "e2e-eval" / time.strftime("%Y%m%d_%H%M%S")).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args, base_out)

    experiments = build_experiments(
        systems=args.systems,
        schedule_sweep=schedule_sweep,
        seq_len=args.seq_len,
        enabled_sweeps=set(args.sweeps),
        gradient_accumulation=args.gradient_accumulation,
        ar_a2a_same_stream=args.ar_a2a_same_stream,
        overlap_chunks=args.overlap_chunks,
        bucket_size_mb=args.bucket_size_mb,
    )
    grouped: dict[str, list[Experiment]] = {system: [] for system in args.systems}
    for exp in experiments:
        grouped[exp.system].append(exp)

    print(f"output directory: {base_out}")
    print(f"planned experiments: {len(experiments)}")
    overall_failed = False

    def _terminate(signum: int, _frame) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    results: list[Result] = []
    for system in args.systems:
        system_out = base_out / system
        system_out.mkdir(parents=True, exist_ok=True)
        system_experiments = grouped[system]

        for index, exp in enumerate(system_experiments, start=1):
            exp_slug = slug(exp, nsight=args.nsight)
            log_path = system_out / f"{index:02d}_{exp_slug}.log"
            metrics_name = f".{exp_slug}.metrics.csv"
            host_metrics_path = base_out / metrics_name
            container_metrics_path = str(CONTAINER_OUT / metrics_name)
            command = in_container_command(exp, args, container_metrics_path if system == "piper" else None)
            nnode, ngpu = backend.layout(exp)

            print(f"[{system} {index}/{len(system_experiments)}] {exp.sweep}: {experiment_label(exp)} config={exp.config}")
            for line in backend.dry_run_lines(exp, command):
                print("  " + line)
            if args.dry_run:
                continue

            if system == "piper":
                backend.prepare_piper(args)
            returncode = 1
            try:
                returncode = backend.run(exp, command, log_path)
                if system == "piper" and not host_metrics_path.is_file():
                    backend.fetch_file(container_metrics_path, host_metrics_path)
                if system == "piper":
                    for leaked_json in base_out.glob("*.json"):
                        leaked_json.unlink()
                iter_mean, iter_std, peak_str, is_oom, reason = parse_log(
                    system,
                    log_path,
                    metrics_path=host_metrics_path if system == "piper" else None,
                )
                if returncode == 0 and iter_mean is not None:
                    status = "ok"
                elif is_oom:
                    status = "oom"
                    reason = reason or "oom"
                else:
                    status = "failed"
                    reason = reason or f"exit_code={returncode}"
                if status != "ok":
                    overall_failed = True
                results.append(
                    Result(
                        system=system,
                        sweep=exp.sweep,
                        config=exp.config,
                        pp=exp.pp,
                        dp=exp.dp,
                        ep=exp.ep,
                        zero_level=exp.zero_level,
                        schedule=exp.schedule,
                        mb_size=exp.mb_size,
                        seq_len=exp.seq_len,
                        local_batch_size=local_batch_size(exp),
                        global_batch_size=global_batch_size(exp),
                        nnode=nnode,
                        ngpu=ngpu,
                        log_path=str(log_path),
                        metrics_path="",
                        returncode=returncode,
                        status=status,
                        iter_time_mean=iter_mean,
                        iter_time_stddev=iter_std,
                        peak_memory_gb_by_rank=peak_str,
                        failure_reason=reason,
                    )
                )
                if host_metrics_path.exists():
                    host_metrics_path.unlink()
                write_csv(base_out / "results.csv", results)
            finally:
                if host_metrics_path.exists():
                    host_metrics_path.unlink()
                backend.cleanup_after_experiment(exp)

    if not args.dry_run:
        write_outputs(base_out, results)
        print(f"wrote results under {base_out}")
    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
