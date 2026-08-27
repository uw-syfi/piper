# test_qwen.py:

def _run_standby(dp_rank):
    original codes + registers its buffers while parked
    if promote:
        join_standby_group
        I = ray.get(standby_actor.wait_state_loaded.remote())
        for _ in range(args.iters) --> calculate with I:
            piper_exec_dag(xxx)
    else:
        original codes

def main(args, pg):
    turn `for _ in range(args.iters):` into `while True` + iter counting
    warp piper_exec_dag with "try ... except"
        except "Ready to continue" --> modify iter counting

# piper.py:
    
def piper_exec_dag(loss_fn, log_stats: bool = False, step_timeout: float | None = None) -> list:
    original codes until `except (ray.exceptions.RayTaskError, ray.exceptions.RayActorError) as e:`
        join_standby_group
        ref = survivor_actor.get_state.remote()              # produce (nixl); let I = last_committed+1 ride inside get_state's return; get_state must include optimizer state (Adam exp_avg/exp_avg_sq/step), not just weights
        ray.get(standby_actor.load_state.remote([ref]))      # standby receives
        del ref
        raise "Ready to continue"

# actor.py

def join_standby_group(self, new_ranks):
    change `self.runtime.standby_dp_group` to `self.runtime.dp_group`
    set unfenced

Limitations:
1. standby worker not recorded with metrics
2. Can only restart at start of an iteration

Notes
1. Docstrings state only what the API does and what each parameter means. Comments exist only for non-obvious "why"s, at most 1–2 lines each. Never narrate implementation mechanics anywhere.
2. Do not commit anything
3. For NIXL RDT, use the method in https://www.anyscale.com/blog/rdt-ray-direct-transport-fast-easy-weight-syncing-for-rl-reinforcement-learning

Tests
1. Create unit test for NIXL transfer
2. E2E test with the automatic kill
3. Record the precise timing and results correctness
