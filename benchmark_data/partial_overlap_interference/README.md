# Pairwise Figure 1 reproducibility bundle

This bundle preserves the exact data and code path for the pairwise benchmark's
Figure 1 realized-overlap dose response. It intentionally excludes unrelated
figures, the multi-aggressor corpus, temporary outputs, and the generic physical
resource suite.

## Reproduce Figure 1 without GPUs

From the repository root:

```bash
python benchmark_data/partial_overlap_interference/validate_bundle.py
python benchmark_data/partial_overlap_interference/render_figure1.py --output /tmp/pairwise-figure1
```

`generate_factor_figures.py` is the exact historical generator (SHA-256 recorded
in `manifest.json`). `render_figure1.py` invokes only its Figure 1 plotting path
against the committed 272-row aggregate source, avoiding dependencies on the
1.1 GB all-figure sample table. The generated PNG and SVG should be compared to
the historical hashes in `manifest.json`; exact raster bytes can vary with the
Matplotlib and font stack.

## Benchmark-to-figure provenance

Figure 1 uses pairwise chunks 1 and 2 from the fixed-20, 12,288-token suite:

- chunk 1 supplies 3,068 estimator-valid communication x compute samples;
- chunk 2 supplies 2,088 estimator-valid communication x communication samples;
- both launch orders and both directed participant orientations are retained;
- the five rendered victim/aggressor kernels are grouped GEMM, P2P send, A2A,
  all-gather, and reduce-scatter.

The authoritative suite manifest identifies the benchmark command and immutable
matrix/profile/trace inputs. The benchmark runner records hashes of its imported
implementation modules. The committed
`scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py`,
`scripts/partial_overlap_interference_benchmarks/run_pairwise_authoritative_12k_coretrend_fixed20.py`,
and their hash-bound support modules in the same package match that recorded
implementation. The preserved historical suite manifest retains the runner path
used during original collection; the reorganized driver maps it to the current
package path when constructing a new run.

The raw chunk corpora are deliberately not duplicated (they total more than 12
GB). Their durable paths and SHA-256 values are recorded in `manifest.json`. The
committed acceptance report and postprocessing receipt establish integrity and
estimator-valid counts. The minimal frozen CPU pipeline under
`provenance/frozen_pipeline/` contains the exact merge, validation, filtering,
conversion, and corpus-validation test sources whose hashes are recorded by the
frozen pipeline map. Starting from the durable raw chunks, the relevant CPU steps
are:

```bash
cd benchmark_data/partial_overlap_interference/provenance/frozen_pipeline
python scripts/merge_pairwise_overlap_results.py CHUNK1.json CHUNK2.json --output MERGED.json
python scripts/validate_pairwise_overlap_corpus.py MERGED.json --output-prefix VALIDATION
python scripts/convert_pairwise_case_results.py MERGED.json TABLES
python -m pytest -q test/test_pairwise_overlap_corpus_validation.py
```

Use the exact validation flags recorded in `provenance/postprocessing_receipt.json`
when rebuilding the authoritative estimator-valid table.

## Figure 1 resource-usage table

The only table-specific processing scripts are
`scripts/partial_overlap_interference_benchmarks/summarize_figure1_isolated_resources.py` and
`scripts/partial_overlap_interference_benchmarks/derive_figure1_peak_sm_footprint.py`. Their
small final tables, per-trial data, geometry evidence, and validation record are
in `resource_usage/`. The GPU collector itself belongs to the broader
physical-resource suite and is outside this bundle.
