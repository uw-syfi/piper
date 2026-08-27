Goal:
Add a standby DP rank. When a rank fails, swap in the DP rank and copy the weights and optimizer state of the survived rank. On which step to redo: failure before UPD → survivors are still at N−1 → copy and redo step N; failure inside UPD → survivor states may differ → copy from one survivor and continue. The standby rank can do initialization at first.

Breakdown:
1. Try to start a rank standby
    - world_size now equals dp_degree * pp_degree + standby ranks
    - Add a CPU+GPU bundle for the standby rank. This means the GPU is reserved.
    - When creating dp groups, do not involve the standby rank.
    - a coordinator change — the parked standby's `run_dp_rank` task never finishes, so M1's `run_program` loop (`while pending: ray.wait…`) must exclude it from its accounting, or a clean 2-DP run never terminates and fail-fast miscounts `failures/n`.

2. But let rank standby wait for running iterations 
    - Effectively start the driver first. 
    - test_qwen.py: if rank standby, pause at `logger.info(f"Running {args.warmup} warmup iterations")`
    - To realise the pause:
        ```python
        @ray.remote
        class PromotionSignal:
            def __init__(self): self.cmd = None
            def set(self, cmd): self.cmd = cmd          # coordinator calls this
            def get(self): return self.cmd              # standby polls this

        # standby's pause, inside test_qwen.main right after piper_setup:
        # the standby can block on a long-poll method (ray.get(sig.wait_for_cmd.remote()) with the coordinator resolving it)
        # cmd tells it: which rank to become, who to copy weights from
        ```
    - The coordinator (which already detects failures in its ray.wait loop — no extra daemon thread needed for this) just calls sig.set.remote({...}) when it decides to promote.
        - Order: set PromotionSignal → then abort.

3. If fail, let rank standby start from the failed iterations
    - The survivor finishes UPD naturally and blocks at N+1's REDUCE
    - Add a commit stage (a which iteration indicator) after UPD's cuda sync
    - Abort the survivor's comm wherever it is → read its commit marker → everyone redoes step last_committed + 1 on the new group.
    - Let the survivor stops at N+1's ALL-REDUCE and in the meantime, the standby copies the survivor's weights. (begins to Phase 2, do not worry now)
    - When fail happens:
        ```python
        _abort_process_group(dp_group) # also ep_group
        new_dp = dist.new_group(ranks=[0, 2], use_local_synchronization=True)
        ```
    - Who executes what: `_abort_process_group` — survivor only (the standby has no dp/ep group to abort); `new_group` — survivor **and** standby, and only them (local-sync rule: members call, nobody else). Both lines run actor-side, triggered after the PromotionSignal is set.
    - Survivor:
        ```python
        except ray.exceptions.RayTaskError as e:
        if ray.get(sig.get.remote()) is not None:      # promotion in progress → I was fenced
            logger.info("fenced by coordinator; waiting for rebuild")
            wait_for_rebuild_and_reissue()             # actor rebuilds groups, then re-fire run_dag
        else:
            raise                                       # genuine failure → M1 behavior unchanged
        ```

Phase 1 implementation: 
- Don't replace the failed replica with rank standby.
    - Do not need to transfer weights through CUDA-IPC.
    - So the standby prints "ready to receive weights". Then it is safe to stop the test.
- Phase 2, do not worry now
    - Standby reads survivor's buffers via CUDA IPC (later RDMA/NIXL), overlapped with detection, no survivor involvement.
    - Survivor's iteration and weight transfer should happen in parallel.
    -  RuntimeState rank remap at promotion, standby's iteration-counter alignment, data-shard adoption.


Testing plan:
- At init, the standby worker should finish init and the regular ranks should work correctly. And results.csv identical to a no-standby run. 
- Same failure mechanism as `/m-coriander/coriander/stfeng/piper/design_docs/health_monitoring.md`. (promotion is enabled by --num-standby > 0; PIPER_STANDBY_ENABLED was removed as redundant)
- At failure, the standby and the survivor should join a NCCL group.
- At failure, the standby should print "ready to receive weights".

> **[Claude]:** Phase 1 implemented and all tests above pass (2026-08-01, 3× H200; plus M1 regressions and the 18 CPU tests). See `docs/standby_worker.md` for knobs, commands, and evidence. Two deviations from this doc, both forced by verified failure modes: (1) the fence needed to be stronger than "abort the comm" — an aborted collective *releases* its kernels with garbage and the pre-enqueued optimizer step would commit poisoned weights (observed empirically: marker advanced past the fault), so in standby mode UPD CPU-syncs all-reduces and refuses the step when fenced; (2) `join_standby_group` must wait for `abort_comms` to fully finish and actors must run with `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` (per `_abort_process_group`'s own docs) — otherwise the fresh survivor+standby comm init deadlocks against the in-flight abort/watchdog. Also note: with the commit marker in place, `last_committed` correctly reads 3 at a fault injected in iteration 4, i.e. "redo iteration 4" — the marker, not the survivor's error text (which is timing-dependent), carries the recovery semantics.
