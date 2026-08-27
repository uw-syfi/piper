# Design

1. Driver — piper_exec_dag (the only real change):

run_refs = [actor.run_dag.remote(loss_fn=loss_fn) for actor in actors.values()]
while True:
    try:
        results = ray.get(run_refs, timeout=step_timeout)   # SAME refs each retry
        break
    except ray.exceptions.GetTimeoutError:
        print("step exceeded timeout; still waiting (peer may be down)")
        # loop — do NOT resubmit run_dag, do NOT return anything
# ... rest of the function unchanged

No PENDING, no signature change, no changes to the results loop. RayActorError/RayTaskError need no except at all — uncaught exceptions already propagate.

Pass the timeout only after warmup (we assume warmup does not fail)

step_timeout: None during warmup; then 5 × measured steady step time.

2. test_qwen.main — no changes. Your "wrapped with try…except…raise" is a no-op: an exception from piper_exec_dag already flies up and fails the task. That's Python doing your escalation for free. (Add a try/except only if you want a log line.)

3. Coordinator — run_program monitors outcomes:

refs = [run_dp_rank.options(...).remote(dp_rank, ...) for dp_rank in range(self.dp_degree)]
pending, results, failures = refs, [], 0
while pending:
    done, pending = ray.wait(pending, num_returns=1)    # whoever finishes/fails FIRST
    try:
        results.append(ray.get(done[0]))
    except Exception as e:
        failures += 1
        print(f"a dp_rank failed ({failures}/{self.dp_degree}): {e}")
        if failures == self.dp_degree:
            raise                                       # everyone's down → job over
return results

One knock-on: run_program now returns results instead of handles, so the harness's double-ray.get becomes a single ray.get(coordinator.run_program.remote(...)) — a one-line change in test_harness.py.

With the coordinator now tolerating the first failure, the loud test's total runtime = the survivor's NCCL watchdog (~10–30 min), because "wait for the standby" waits for something that doesn't exist yet. For milestone 1 I'd make it a policy flag:

if failures and not STANDBY_ENABLED:   # M1: no standby exists
    raise                              # fail fast; waiting buys nothing yet

— fast tests now, and the STANDBY_ENABLED branch is the marked socket where the next milestone plugs in.

# Test

E2E test, in BWD, add a global counter, in the second iteration (past warmup), add an error. in one DP worker. Test with GPU 1 and 2 (cuda_visible_device).

```python
# in case TaskType.BWD, executors.py
if os.environ.get("PIPER_FAULT") == f"bwd:{self._iter_count}:{self.runtime.dp_rank}":
    raise RuntimeError("injected fault")        # loud mode
    # or: time.sleep(x)                    # stuck mode — tests the timeout arm
```

self._iter_count should be a global counter just for debugging this and not in production.

