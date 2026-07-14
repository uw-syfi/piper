# Piper Qwen E2E Artifact

This directory is the reproducible entrypoint for the paper e2e evals across
Piper, TorchTitan, Megatron, and DeepSpeed.

## Build

From the piper repository root:

```bash
docker build -f artifact/Dockerfile -t piper-e2e-artifact:latest .
```

The image pins TorchTitan to upstream commit
`b01adfb544b4331ecab090ebdb50b2296cd8eb6a` and applies
`artifact/patches/torchtitan-qwen-e2e.patch`. Megatron defaults to the stable
release tag `26.04-alpha.rc2`; pass `--build-arg MEGATRON_REF=<commit-or-tag>`
to override it.

## Local Single-Node Runs

Run the smallest smoke eval on local GPUs:

```bash
python artifact/e2e_eval.py \
  --backend local \
  --systems torchtitan piper \
  --sweeps local \
  --image piper-e2e-artifact:latest
```

Run the full default matrix locally:

```bash
python artifact/e2e_eval.py --backend local --image piper-e2e-artifact:latest
```

Local mode uses one Docker container per experiment and maps
`artifact/out/e2e-eval/<run_id>` into `/workspace/eval-out` inside the
container. For local mode, `nnode=1` and `ngpu=pp*dp`. If
`/m-coriander/coriander/mfris/torchtitan/assets/hf` exists, it is mounted
read-only into `/workspace/torchtitan/assets/hf` so TorchTitan can find the
downloaded Qwen tokenizer assets. Override this with
`--local-torchtitan-assets-path`.

## AWS Existing-Node Runs

AWS mode targets an existing EC2 cluster over SSH. Required environment:

```bash
export SSH_KEY=/path/to/key.pem
export HEAD_PUBLIC_IP=<public head ip>
export HEAD_PRIVATE_IP=<private head ip>
export WORKER1_PRIVATE_IP=<private worker ip>
export WORKER2_PRIVATE_IP=<private worker ip>
export WORKER3_PRIVATE_IP=<private worker ip>
```

Start containers automatically if the image already exists on every node:

```bash
python artifact/e2e_eval.py \
  --backend aws \
  --aws-start-containers \
  --image piper-e2e-artifact:latest \
  --container piper_artifact
```

If containers are already running, omit `--aws-start-containers`. AWS mode uses
`nnode=dp` and `ngpu=pp`, matching the original EC2 experiment layout.

## Useful Options

```bash
python artifact/e2e_eval.py --dry-run
python artifact/e2e_eval.py --backend local --sweeps local --local-cuda-visible-devices 0
python artifact/e2e_eval.py --backend local --sweeps local --local-torchtitan-assets-path /path/to/assets/hf
python artifact/e2e_eval.py --backend local --sweeps local --no-torchtitan-use-bmm-experts
python artifact/e2e_eval.py --backend local --sweeps local --piper-use-inductor --torchtitan-compile
python artifact/e2e_eval.py --systems piper torchtitan --sweeps schedule
python artifact/e2e_eval.py --sweeps zero --bucket-size-mb 25
python artifact/e2e_eval.py --backend aws --nsight --systems piper
```

Outputs are written under `artifact/out/e2e-eval/<run_id>/`:

- `results.csv`
- `<sweep>.png` for combined plots with successful rows
- `<system>/*.log`

## Notes

- Local Docker runs bind-mount this `artifact/` directory plus Piper `src/`,
  `examples/`, and `test/` into the container, so runner/script/source edits do
  not require rebuilding the image. Rebuild only when dependencies or the
  Dockerfile image contents change.
- The TorchTitan patch is intentionally limited to the Qwen e2e runtime needs:
  Qwen3 MoE eval configs, mesh ordering, timing/memory logs, dataset retry
  behavior, and small runtime compatibility fixes.
- The piper runner invokes the current tracked entrypoint:
  `examples/test_harness.py --test-file examples/test_qwen.py`.
- Piper base schedules are generated per experiment, so the artifact is not
  limited to the checked-in `examples/base-schedules/*.json` files.
