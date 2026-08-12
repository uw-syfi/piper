# Profile-and-replay tools

This package contains the public offline replay CLI, profile-ratio exporter,
and reusable diagnostics for comparing instrumented traces with simulator
replays.

From the repository root:

```bash
python scripts/profile_and_replay/simulate_step_times.py PROFILE.json
python scripts/profile_and_replay/export_node_profile_ratios.py PROFILE.json
```

The tools import production profile and simulator types from `src/`; they do
not contain a separate simulator implementation.
