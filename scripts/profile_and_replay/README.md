# Profile-and-replay tools

This package contains the public offline replay CLI, profile-ratio exporter,
and reusable diagnostics for comparing instrumented traces with simulator
replays.

From the repository root:

```bash
python scripts/profile_and_replay/simulate_step_times.py PROFILE.json
python scripts/profile_and_replay/export_node_profile_ratios.py PROFILE.json
python scripts/profile_and_replay/attribute_simulation_savings.py PROFILE.json
python scripts/profile_and_replay/visualize_simulation_savings.py \
  NORMAL_INPUT ATTRIBUTION.json --output savings-trace.json
```

The tools import production profile and simulator types from `src/`; they do
not contain a separate simulator implementation.
