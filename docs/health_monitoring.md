# Health Monitoring (Milestone 1)

Detects DP worker failures (crash or hang) during training and fails the job
fast with a classified error, instead of hanging silently until the NCCL
watchdog (~10–30 min). No recovery yet — standby promotion is the next milestone.

## Architecture

```
harness (test_harness.py)
  └─ PiperProgramCoordinator.run_program          [src/coordinator.py]
       async actor; awaits dp_rank tasks (completion order)
       ├─ 1st task failure → log "a dp_rank task failed (k/n)"
       │    └─ no standby configured → raise (fail fast) → job exits
       └─ driver, per dp_rank (test_qwen.main)
            └─ piper_exec_dag(..., step_timeout)  [src/piper.py]
                 ray.get(run_refs, timeout=step_timeout)
                 ├─ GetTimeoutError → warn "step exceeded step_timeout",
                 │                    keep waiting on the SAME step
                 └─ RayTaskError / RayActorError → propagate → task fails
                      └─ PiperActor.run_dag       [src/actor.py, executors.py]
                           actor error or injected fault originates here
```

- Stuck and slow are indistinguishable at the driver (both are `GetTimeoutError`);
  the driver only warns and waits. Loud errors escalate: actor → driver task →
  coordinator → harness.
- A hung *victim* is never detected in M1; the survivor's NCCL watchdog is the
  backstop. Bounded hang detection requires heartbeats (later milestone).

## User-facing knobs

| Knob | Where | Meaning |
|---|---|---|
| `step_timeout` (param, seconds, default `None`) | `piper_exec_dag` | Warn when a step is overdue; `None` disables. `test_qwen.py` sets it automatically: `None` during warmup, then `max(5.0, 5 × last warmup step time)`. |
| `--num-standby` (harness flag, default 0) | coordinator | `0`: fail fast on first dp_rank failure. `>0`: promote a standby instead (see `docs/standby_worker.md`). |
| `PIPER_FAULT` (env, debug only) | `executors.py` BWD dispatch | Fault injection for tests; fires once per matching iteration and prints `PIPER_FAULT firing: <spec>`. Formats below. |

`PIPER_FAULT` formats (`<iter>` is the 0-based `run_dag` counter; warmup counts):

- `bwd:<iter>:<dp_rank>` — raise `RuntimeError` (loud mode)
- `bwd:<iter>:<dp_rank>:sleep:<seconds>` — sleep in BWD (stuck mode)

## Testing the behaviors

Both tests: pure-DP2 schedule, 2 GPUs, default warmup=3 (iters 0–2), so iter 4
is the second timed iteration.

**Loud mode** — injected crash in dp_rank 1:

```bash
CUDA_VISIBLE_DEVICES=1,2 PIPER_FAULT=bwd:4:1 \
python examples/test_harness.py --test-file examples/test_qwen.py \
  --base-schedule examples/base-schedules/dp2.json \
  --schedule 1f1b --ranks 1 --mbs 1
```

Expected: `PIPER_FAULT firing`, coordinator logs `a dp_rank task failed (1/2)`
with the `RuntimeError: injected fault` traceback, job exits nonzero within
seconds. The driver stdout shows only the traceback; the other lines are in
the Ray worker logs (`<temp-dir>/session_*/logs/worker-*.out`).

**Stuck mode** — 60 s hang in dp_rank 1 (self-recovers):

```bash
CUDA_VISIBLE_DEVICES=1,2 PIPER_FAULT=bwd:4:1:sleep:60 \
python examples/test_harness.py --test-file examples/test_qwen.py \
  --base-schedule examples/base-schedules/dp2.json \
  --schedule 1f1b --ranks 1 --mbs 1
```

Expected: both drivers repeat `step exceeded step_timeout=5.0s; still waiting`
for ~60 s (victim and survivor are indistinguishable), then the run completes
normally — exit 0, `results.csv` written, faulted iteration ≈ 60 s.

Regression: `python -m pytest -m "not gpu" -v test` and any normal run without
`PIPER_FAULT` are unaffected (`step_timeout=None` preserves old behavior).

## Limitation: detection only — never which component or why

M1 reports *that* a dp_rank task failed and the error that surfaced — **never
which component is the root cause**. Do not read the coordinator's first
failure report as attribution.

Example (DP=2): GPU 1's NCCL driver wedges mid-all-reduce, process still alive.
Both ranks hang — rank 1's kernel is wedged, rank 0's kernel spins waiting for
it. Both drivers warn `step exceeded step_timeout`. Eventually each rank's own
NCCL watchdog fires; the race is nondeterministic, and if rank 1's host-side
NCCL is also impaired, only rank 0's watchdog fires. The coordinator then logs
**the healthy GPU 0 as the first failure** while the sick GPU 1 dies second or
never.

Root-cause attribution needs evidence from below the training stack (e.g. Xid
scan via `nvidia-smi -q` / `dmesg` per GPU, `NCCL_DEBUG=WARN` logs) and is out
of scope for M1.
