# Piper: A Programmable Distributed Training System

[![arXiv](https://img.shields.io/badge/arXiv-2606.11169-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.11169)

New distributed training strategies should not require new distributed runtimes; Piper gives PyTorch users direct control over model placement and GPU scheduling with lightweight annotations and a small scheduling language.

## Updates

* 2026-06 - Blog post: [Introducing Piper: A Programmable Distributed Training System](https://syfi.cs.washington.edu/blog/2026-06-05-piper/).
* 2026-06 - Paper released on arXiv: [2606.11169](https://arxiv.org/abs/2606.11169).

## Introduction

Large training jobs increasingly combine multiple parallelism strategies such as pipeline, data, and expert parallelism with ZeRO-style sharding, creating placement and GPU scheduling choices that current frameworks cannot express cleanly.
Today, ML researchers and practitioners choose between one-off specialized systems that perform well but are hard to adapt, and general-purpose frameworks that are easier to use but expose limited control.

Piper is a user-controllable distributed training system for PyTorch that separates model placement and GPU scheduling from model code and runtime implementation.
With lightweight model annotations and a small scheduling language, Piper lets users express, visualize, profile, and run high-performance training schedules such as DualPipe-style pipeline- and expert-parallel overlap.

## Architecture

![Piper architecture](figs/architecture.jpg)

Piper has two user-facing inputs:

* An annotated PyTorch model: standard model code with lightweight tags for schedulable regions such as pipeline stages and MoE experts.
* A schedule-directive program: instructions that tell the Piper compiler how to shard, replicate, order, and overlap those schedulable regions.

The compiler traces the model with TorchDynamo, splits the graph by Piper annotations, builds a distributed training DAG IR, and applies DAG rewrites according to the schedule directives.
The directive rewrites insert point-to-point pipeline communication, DP collectives, ZeRO gather/scatter collectives, EP all-to-all communication, temporal edges, device assignments, and logical stream assignments.

The runtime decomposes the global DAG into per-device execution plans and runs them on Ray actors.
Each actor manages local CUDA streams, process groups/communicators, model-state buffers, and intermediate tensors.

## Installation

Requires Python 3.10+ on Linux with CUDA GPUs.
The current setup has been tested with Python 3.10 and CUDA 12.x.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you do not already have Python 3.10 locally:

```bash
conda create -n piper python=3.10 -y
conda activate piper
python -m pip install --upgrade pip
python -m pip install -e .
```

## Quickstart

From the repository root, run the Qwen example with a DualPipeV schedule:

```bash
python examples/test_harness.py \
  --test-file examples/test_qwen.py \
  --base-schedule examples/base-schedules/pp4_dp2_ep2_v_placement.json \
  --schedule dualpipev \
  --ranks 2 \
  --mbs 4
```

The base schedule `pp4_dp2_ep2_v_placement.json` describes a PP x DP x EP placement with four annotated pipeline regions mapped onto two physical pipeline ranks in a V layout.
Regions `PP=0` and `PP=3` run on device group `[0, 2]`, while `PP=1` and `PP=2` run on device group `[1, 3]`.
Within each device group, Piper replicates non-expert regions for DP and shards expert regions for EP.
This example expects four visible CUDA devices.
The harness appends DualPipeV `split` and `order` directives for two physical pipeline ranks and four microbatches.

Run the Qwen example with PP x ZeRO-3 1F1B schedule:

```bash
python examples/test_harness.py \
  --test-file examples/test_qwen.py \
  --base-schedule examples/base-schedules/pp2_dp2_ep2_zero3.json \
  --schedule 1f1b \
  --ranks 2 \
  --mbs 4
```

The base schedule `pp2_dp2_ep2_zero3.json` describes a PP x DP x EP placement with two pipeline stages, DP degree two, EP sharding, and ZeRO-3-style gradient and parameter sharding.
Stage `PP=0` runs on device group `[0, 2]`, while `PP=1` runs on device group `[1, 3]`.
This example also expects four visible CUDA devices.
The harness appends 1F1B `split` and `order` directives for two pipeline ranks and four microbatches.
Each run creates `out/<timestamp>/` with the complete generated schedule and metrics.

## Overview of Inputs

### Annotations

`piper.annotate(tag)` is a context manager that attaches Piper metadata to all PyTorch operations traced inside the scope.
The `tag` is a non-empty string naming a schedulable dimension of the model, such as `PP` for pipeline regions or `EP` for expert regions.
For each tag name, Piper assigns integer indices in trace order, so repeated `piper.annotate("PP")` scopes become `PP=0`, `PP=1`, and so on.

The schedule directives program uses these tag names and indices in filters to select regions of the traced model.
For example, a filter can select one concrete region by its tag index, all regions with a tag, or regions that match a combination of tags.
See the blog walkthrough section on [annotating a Qwen3 MoE model](https://syfi.cs.washington.edu/blog/2026-06-05-piper/#annotating-a-qwen3-moe-model) for an example and further details.

### Schedule Directives Program

A schedule directive program is a JSON array of directive objects.
Each directive has an `op` string naming the rewrite to apply to the distributed training DAG IR.
Most directives use a `filter` object to select the model region to apply the directive to.
The `order` directive uses `filters` to describe a sequence of filter groups.
A `filter` is a JSON object whose keys are tag names and whose values are the tag indices to match.

Filter keys can refer to annotation tags, new tags added by previous directives, or the compiler-provided `PASS` tag which supports `F` (forward), `B` (backward), `BI` (backward for inputs), and `BW` (backward for weights) for different training step stages.
Filter values can be tag indices or the special value `"*"` to match any concrete index for the key.

The empty filter `{}` matches the entire DAG IR.
All key/value pairs in a filter are conjunctive, so `{"PP": 1, "EP": "*"}` matches nodes that are in pipeline region `PP=1` and have any `EP` index.

Supported directives:

* `place` assigns matched compute regions to device groups and names the stream used for inserted PP send/recv communication.
  * `op`: `"place"`
  * `filter`: Selects the model regions to place.
  * `devices`: Non-empty list of CUDA device IDs for the placement group.
  * `stream`: Optional logical stream name for inserted point-to-point communication.
* `replicate` replicates matched regions across devices and inserts DP synchronization.
  * `op`: `"replicate"`.
  * `filter`: Selects the model regions to replicate.
  * `devices`: Non-empty list of CUDA device IDs across which the region is replicated.
  * `reduce_stream`: Optional logical stream name for gradient reduction collective communication.
  * `gather_stream`: Optional logical stream name for ZeRO-3 parameter materialization collective communication.
  * `bucket_size`: Optional parameter bucket size in MB for finer-grained synchronization.
  * `shard_grads`: Optional boolean enabling ZeRO-2-style gradient sharding.
  * `shard_params`: Optional boolean enabling ZeRO-3-style parameter sharding.
* `shard` shards matched regions across devices and inserts all-to-all communication, typically for expert regions.
  * `op`: `"shard"`
  * `filter`: Selects the model regions to shard.
  * `devices`: Non-empty list of CUDA device IDs across which the region is sharded.
  * `stream`: Optional logical stream name for inserted all-to-all collective communication.
* `split` duplicates the matched DAG by a named microbatch dimension.
  * `op`: `"split"`
  * `filter`: Selects the sub-DAG to duplicate.
  * `dim_name`: Non-empty string naming the new split dimension.
  * `num_microbatches`: Positive integer number of copies to create.
* `order` adds temporal dependencies between filter groups.
  * `op`: `"order"`
  * `filters`: List of at least two non-empty filter groups.
  * Each filter group is a list of filter objects that occupy the same ordering slot.
  * Consecutive filter groups create temporal dependencies from one slot to the next.
  * Multiple filters in the same group permit Piper to interleave those sub-DAGs.

In practice, base schedules under `examples/base-schedules/` describe model placement and composed parallelism choices, while `examples/test_harness.py` appends generated `split` and `order` directives for schedule families such as `1f1b`, `interleaved_1f1b`, `zerobubble`, and `dualpipev`.
For more detail, see the blog walkthrough sections on [PP x DP x EP placement](https://syfi.cs.washington.edu/blog/2026-06-05-piper/#scheduling-dualpipe-like-pp-x-dp-x-ep-model-placement), [DualPipe-like pipeline scheduling](https://syfi.cs.washington.edu/blog/2026-06-05-piper/#scheduling-a-dualpipe-like-pipeline-schedule), and [schedule builders](https://syfi.cs.washington.edu/blog/2026-06-05-piper/#generating-directives-with-schedule-builders).

## Overview of Outputs

Every harness run creates a timestamped directory under `out/`:

```text
out/<timestamp>/
|-- <schedule>_pp<ranks>_mbs<mbs>.json
`-- results.csv
```

`results.csv` contains a row for each SPMD rank (e.g., each DP rank) reporting the mean/std iteration time, training throughput in tokens/sec, and per-PP-rank peak memory in GB.

Optional artifact generation flags:

* `--viz` renders the generated pipeline schedule and per-PP-rank DAG IRs under the run directory.
* `--pytorch-profiler --pytorch-profiler-iters <n>` runs extra profiled iterations and writes combined Chrome trace files per SPMD rank; Piper annotates GPU events with DAG IR node labels.

## Citation

If you use Piper in your research, please cite:

```bibtex
@misc{frisella2026piperprogrammabledistributedtraining,
      title={Piper: A Programmable Distributed Training System}, 
      author={Megan Frisella and Shubham Tiwari and Andy Ruan and Yi Pan and Parker Gustafson and Mat Jacob and Gilbert Bernstein and Stephanie Wang},
      year={2026},
      eprint={2606.11169},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2606.11169}, 
}
```
