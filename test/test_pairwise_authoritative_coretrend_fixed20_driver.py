from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = (
    REPO_ROOT
    / "scripts/partial_overlap_interference_benchmarks/"
    "run_pairwise_authoritative_12k_coretrend_fixed20.py"
)


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "authoritative_coretrend_fixed20_driver", DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_exact_fixed20_coretrend_partition() -> None:
    driver = load_driver()
    manifest = driver.validate_manifest(driver.DEFAULT_MANIFEST)

    assert manifest["schema_version"] == 2
    assert manifest["suite_id"] == driver.SUITE_ID
    assert manifest["core_case_keys"] == driver.CORE_TREND_CASE_KEYS
    assert [row["chunk_id"] for row in manifest["chunks"]] == driver.CHUNK_IDS
    assert [len(row["case_keys"]) for row in manifest["chunks"]] == [10, 10, 5]
    assert [row["nominal_cells"] for row in manifest["chunks"]] == [240, 240, 60]
    assert [
        row["nominal_issued_records"] for row in manifest["chunks"]
    ] == [4800, 4800, 1200]
    assert manifest["totals"] == {
        "core_cases": 25,
        "pairwise_cases": 20,
        "multi_aggressor_cases": 5,
        "coverage_targets": 6,
        "nominal_cells": 540,
        "issued_records_per_nominal_stratum": 20,
        "nominal_issued_records": 10800,
        "maximum_start_anchor_unscored_stages_per_scored_record": 6,
        "maximum_nominal_start_anchor_unscored_stages": 64800,
    }


def test_manifest_pins_fresh_ray_topology_and_absolute_artifacts() -> None:
    driver = load_driver()
    manifest = driver.validate_manifest(driver.DEFAULT_MANIFEST)
    topology = manifest["topology"]

    assert topology["ray_address"] == "10.65.8.91"
    assert topology["ray_port"] == 6379
    assert topology["ray_version"] == "2.44.1"
    assert topology["node_ips"] == ["10.65.8.91", "10.65.10.25"]
    assert topology["placement"] == "pp_outer"
    assert Path(manifest["suite_lock"]) == driver.EXPECTED_SUITE_LOCK
    for chunk in manifest["chunks"]:
        for field in ("output", "checkpoint"):
            path = Path(chunk[field])
            assert path.is_absolute()
            assert path.parent == driver.EXPECTED_ARTIFACT_ROOT


def test_chunk1_command_is_fixed20_and_case_boundary_recoverable() -> None:
    driver = load_driver()
    manifest = driver.validate_manifest(driver.DEFAULT_MANIFEST)
    chunk = manifest["chunks"][0]
    command = driver.runner_command(
        manifest, chunk, dry_run=False, resume=True
    )

    assert command.count("--iterations") == 1
    assert command[command.index("--iterations") + 1] == "20"
    assert command.count("--baseline-repetitions") == 1
    assert command[command.index("--baseline-repetitions") + 1] == "3"
    assert "--adaptive-sampling" not in command
    assert command.count("--resume") == 1
    assert command[command.index("--checkpoint") + 1] == chunk["checkpoint"]
    assert command[command.index("--run-id") + 1] == chunk["run_id"]
    assert command[command.index("--node-ips") + 1] == "10.65.8.91,10.65.10.25"


def test_dry_run_command_never_resumes_or_executes_adaptive_sampling() -> None:
    driver = load_driver()
    manifest = driver.validate_manifest(driver.DEFAULT_MANIFEST)
    command = driver.runner_command(
        manifest, manifest["chunks"][0], dry_run=True, resume=False
    )

    assert command[-1] == "--dry-run"
    assert "--resume" not in command
    assert not any(value.startswith("--adaptive") for value in command)


def test_manifest_rejects_relative_or_wrong_root_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = load_driver()
    original_loader = driver.load_json
    original = original_loader(driver.DEFAULT_MANIFEST)

    relative = copy.deepcopy(original)
    relative["chunks"][0]["output"] = "out/chunk01.json"
    monkeypatch.setattr(
        driver, "load_json",
        lambda path: relative if path == driver.DEFAULT_MANIFEST else original_loader(path),
    )
    with pytest.raises(ValueError, match="absolute shared path"):
        driver.validate_manifest(driver.DEFAULT_MANIFEST)

    wrong_root = copy.deepcopy(original)
    wrong_root["chunks"][0]["output"] = "/tmp/chunk01.json"
    monkeypatch.setattr(
        driver, "load_json",
        lambda path: wrong_root if path == driver.DEFAULT_MANIFEST else original_loader(path),
    )
    with pytest.raises(ValueError, match="live directly under"):
        driver.validate_manifest(driver.DEFAULT_MANIFEST)


def test_manifest_rejects_adaptive_or_wrong_iteration_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = load_driver()
    original_loader = driver.load_json
    altered = copy.deepcopy(original_loader(driver.DEFAULT_MANIFEST))
    altered["experiment"]["fixed_sampling"][
        "records_per_nominal_stratum"
    ] = 30
    monkeypatch.setattr(
        driver, "load_json",
        lambda path: altered if path == driver.DEFAULT_MANIFEST else original_loader(path),
    )
    with pytest.raises(ValueError, match="sampling policy"):
        driver.validate_manifest(driver.DEFAULT_MANIFEST)


def test_selected_chunks_rejects_unknown_identity() -> None:
    driver = load_driver()
    manifest = driver.validate_manifest(driver.DEFAULT_MANIFEST)
    with pytest.raises(ValueError, match="unknown chunk IDs"):
        driver.selected_chunks(manifest, ["chunk-does-not-exist"])


def test_suite_lock_is_nonblocking_and_reentrant_attempt_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = load_driver()
    manifest = copy.deepcopy(driver.validate_manifest(driver.DEFAULT_MANIFEST))
    test_lock = tmp_path / ".fixed20-test.lock"
    manifest["suite_lock"] = str(test_lock)
    monkeypatch.setattr(driver, "EXPECTED_SUITE_LOCK", test_lock)
    with driver.suite_launch_lock(manifest):
        with pytest.raises(RuntimeError, match="another authoritative"):
            with driver.suite_launch_lock(manifest):
                pass
