# Standby Worker (Phase 1)

A standby DP rank reserves a GPU, fully initializes, parks before warmup, and
on a trainer failure joins the survivor in a fresh NCCL group ("ready to
receive weights"). Weight transfer and resume are Phase 2.

## Architecture

```
 coordinator            driver dp0            driver dp1           driver dp2 (standby)
 ───────────            ──────────            ──────────           ────────────────────
 spawn drivers ───────► piper_setup           piper_setup          piper_setup (full init)
 cmd channel + registry   register actors ──►   register actors ──►  register actors
      │                 train loop            train loop           PARK on wait_for_cmd
      │                 FWD BWD ═ALL-REDUCE═ FWD BWD ✗ fault              ⋮
 asyncio.wait sees dp1's task fail ◄─────────── task fails                ⋮
      │ set_cmd({promote, failed:1,                                       ⋮
      │          source:0, new_ranks:[0,2]}) ────────────────────► wakes  │
      │ abort_comms ──► fence flag → abort                                │
      │                 dp+ep → abort_done                                │
      │                 step errors → fenced arm                          │
      │                 (signal names dp1, not me)                        │
      │                 join_standby_group ◄═══ new group [0,2] ═══► join_standby_group
      │                 sanity all-reduce = 2.0      │             sanity all-reduce = 2.0
      │                 last_committed=3 →           │             print "ready to
      │                 PiperFencedError →           │             receive weights" →
      │                 return partial metrics       │             return
 all refs resolved → return results → exit 0
```

The coordinator is an asyncio actor: `run_program` awaits the driver tasks,
so `register_actors` / `get_cmd` / `wait_for_cmd` (the promotion command
channel and per-dp_rank actor registry, formerly a separate `PromotionSignal`
named actor) are served concurrently on the same event loop. Drivers reach it
via the handle in `piper_metadata.coordinator`.

No failure: coordinator sends `{op: shutdown}` after trainers finish; the
standby wakes, exits; results.csv identical to a no-standby run.

## Knobs

| Knob | Where | Meaning |
|---|---|---|
| `--num-standby N` (default 0) | `test_harness.py` | Reserve N standby ranks (+N GPU bundles, `world_size = dp*pp + N`). N > 0 enables promotion on trainer failure; 0 keeps fail-fast. pp_degree must be 1. |
| `PIPER_FAULT` (env, debug) | `executors.py` | Fault injection: `bwd:<iter>:<rank>` (crash) or `bwd:<iter>:<rank>:sleep:<s>` (stall). |

## Tests

```bash
# failure -> promotion
CUDA_VISIBLE_DEVICES=0,1,2 PIPER_FAULT=bwd:4:1 python examples/test_harness.py \
  --test-file examples/test_qwen.py --base-schedule examples/base-schedules/dp2.json \
  --schedule 1f1b --ranks 1 --mbs 1 --num-standby 1
# -> promoting standby 2; fenced, optimizer step refused (last_committed=3);
#    sanity_allreduce=2.0; "ready to receive weights"; exit 0 (~2 s fault->joined)
```

Result (2026-08-10, 3x H200, exit 0):

```
10:44:18.771  standby dp_rank 2: initialized and parked; waiting for promotion or shutdown
10:44:31.919  Running 3 timed iterations
              PIPER_FAULT firing: bwd:4:1
10:44:31.979  a dp_rank task failed (1/2)
10:44:31.980  promoting standby 2 for failed rank 1: {'op': 'promote', 'failed': 1,
              'source': 0, 'new_ranks': [0, 2]}
10:44:32.200  fenced by coordinator during promotion (step error: ... optimizer step
              refused (last_committed=3))
10:44:32.502  abort_comms: aborted ['dp_group', 'ep_group']
10:44:34.391  join_standby_group: ranks=[0, 2] sanity_allreduce=2.0   (both actors)
10:44:34.394  survivor: joined standby group [0, 2]; last_committed=3; recovery would
              redo iteration 4
              ready to receive weights
              metrics written to out/20260810_104327/results.csv
```

Fault -> joined: ~2.4 s. End-of-run log lines can be lost to Ray's driver log
forwarding; ground truth is `<ray_tmp>/session_*/logs/worker-*.out`.

## Limitations

- Promotion triggers on loud failures only; a silently hung rank is not
  detected — and standby mode disables the NCCL watchdog backstop
  (`TORCH_NCCL_ASYNC_ERROR_HANDLING=0`, required by `_abort_process_group`).
