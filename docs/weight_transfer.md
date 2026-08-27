# Weight Transfer (Phase 2)

Extends standby promotion (`docs/standby_worker.md`): after the fence and
`join_standby_group`, the standby now receives the survivor's full training
state (params + Adam state) via NIXL RDT, and both ranks resume lockstep
training at `last_committed + 1`. The run finishes with exit 0 instead of
stopping cleanly.

New requirements: Ray >= 2.57 and `pip install nixl`.

## Architecture

New steps, starting where Phase 1 ended (both peers have joined the new
group; `join_standby_group` now also swaps it into `runtime.dp_group/ep_group`
and lifts the fence):

```
 survivor driver              survivor actor          standby actor        standby driver
 ───────────────              ──────────────          ─────────────        ──────────────
                                                      prepare_standby_state (while parked):
                                                      materialize Adam state as zeros,
                                                      register NIXL buffers
 make_state_ref ────────────► ray.put(state, nixl)
 load_state([ref]) ─────────────────────────────────► set_target_for_ref(ref, own buffers)
                              params+Adam ── RDMA ──► lands in place; sets iter counter
 ray.get returns  = resume barrier                    wait_state_loaded returns lc+1 ─► driver
 raise PiperResume(lc+1)
 main loop: redo iteration lc+1  ═══ lockstep ═══     run iterations lc+1..end
```

- State = one flat list, `[p, exp_avg, exp_avg_sq, step]` per bucket param;
  both peers build it with the same function over identical bucket layouts,
  so `set_target_for_ref` matches by position.
- The ref is created with `ray.put(..., _tensor_transport="nixl")` (not a
  decorated method): Ray 2.57 only borrows nested refs created via ray.put.
- The survivor's `ray.get` on `load_state` is the resume barrier: its live
  tensors are not stepped until the transfer has landed.

## Tests and results

E2E (same command as Phase 1's promotion test):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PIPER_FAULT=bwd:4:1 python examples/test_harness.py \
  --test-file examples/test_qwen.py --base-schedule examples/base-schedules/dp2.json \
  --schedule 1f1b --ranks 1 --mbs 1 --num-standby 1
```

Result (2026-08-26, 3x H200 idle, exit 0):

```
04:16:33.319  a dp_rank task failed (1/2)
04:16:33.534  fenced ... optimizer step refused (last_committed=3)
04:16:36.827  standby: state loaded; running iterations 4..5 as replacement
04:16:36.827  survivor: resuming after promotion at iteration 4
              metrics written to out/20260826_041508/results.csv
```

Fault -> resumed training ~3.5 s (transfer ~3 s, incl. one-time NIXL agent
init). The first redone iteration additionally pays the new NCCL group's
lazy communicator init.

## Limitations

- The promoted standby's iterations are not recorded in `results.csv`.
- Recovery granularity is a whole iteration (`last_committed + 1`). No step classifications.
- NCCL hangs not detected.
