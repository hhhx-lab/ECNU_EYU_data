"""Deterministic serial and fork-parallel execution for Route A Gate 1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import multiprocessing
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from .met_aug_core import (
    ComponentManifest,
    EventContext,
    JsonlAuditSink,
    MetAugEngine,
    RouteConfig,
    VALID_MASK_MANIFEST_SCHEMA,
    canonical_json_sha256,
    sha256_file,
)


MIN_GATE1_EVENTS = 100000


@dataclass
class Gate1Stats:
    event_count: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)
    stratum_counts: Counter[str] = field(default_factory=Counter)
    same_group_violations: int = 0
    selected: int = 0
    donor_valid: int = 0
    label_valid: int = 0
    placement_valid: int = 0

    def observe(self, result: Any, *, target_group: str) -> None:
        self.event_count += 1
        self.counts[result.state] += 1
        if result.reason:
            self.reasons[result.reason] += 1
        if result.reason != "NOT_SELECTED":
            self.selected += 1
        if result.record is not None:
            self.donor_valid += 1
            if result.record.patient_group == target_group:
                self.same_group_violations += 1
        if result.record is not None and result.reason != "LABEL_INVALID":
            self.label_valid += 1
        if result.state == "PLACEMENT_VALID":
            self.placement_valid += 1
            if result.record is None:
                raise RuntimeError("PLACEMENT_VALID event has no donor record")
            self.stratum_counts["|".join(result.record.stratum)] += 1

    def merge(self, other: "Gate1Stats") -> None:
        self.event_count += other.event_count
        self.counts.update(other.counts)
        self.reasons.update(other.reasons)
        self.stratum_counts.update(other.stratum_counts)
        self.same_group_violations += other.same_group_violations
        self.selected += other.selected
        self.donor_valid += other.donor_valid
        self.label_valid += other.label_valid
        self.placement_valid += other.placement_valid

    def as_mapping(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "counts": dict(self.counts),
            "reasons": dict(self.reasons),
            "stratum_counts": dict(self.stratum_counts),
            "same_group_violations": self.same_group_violations,
            "selected": self.selected,
            "donor_valid": self.donor_valid,
            "label_valid": self.label_valid,
            "placement_valid": self.placement_valid,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Gate1Stats":
        return cls(
            event_count=int(value["event_count"]),
            counts=Counter({str(key): int(count) for key, count in value["counts"].items()}),
            reasons=Counter({str(key): int(count) for key, count in value["reasons"].items()}),
            stratum_counts=Counter(
                {str(key): int(count) for key, count in value["stratum_counts"].items()}
            ),
            same_group_violations=int(value["same_group_violations"]),
            selected=int(value["selected"]),
            donor_valid=int(value["donor_valid"]),
            label_valid=int(value["label_valid"]),
            placement_valid=int(value["placement_valid"]),
        )


Gate1Assets = dict[str, tuple[np.ndarray, np.ndarray]]

_FORK_MANIFEST: ComponentManifest | None = None
_FORK_CONFIG: RouteConfig | None = None
_FORK_ASSETS: Gate1Assets | None = None
_FORK_CASE_IDS: tuple[str, ...] | None = None
_FORK_CASE_SCHEDULE: np.ndarray | None = None


def load_target_assets(path: str | Path, expected_ids: set[str]) -> Gate1Assets:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != VALID_MASK_MANIFEST_SCHEMA:
        raise ValueError("unsupported valid-mask manifest schema")
    if payload.get("manifest_sha256") != canonical_json_sha256(
        payload, exclude=("manifest_sha256",)
    ):
        raise ValueError("valid-mask manifest SHA256 mismatch")
    records_path = manifest_path.parent / payload["records_file"]
    if sha256_file(records_path) != payload.get("records_sha256"):
        raise ValueError("valid-mask records SHA256 mismatch")
    assets: Gate1Assets = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row["case_id"])
        asset_path = (manifest_path.parent / row["mask_path"]).resolve()
        if sha256_file(asset_path) != row["sha256"]:
            raise ValueError(f"valid-mask asset SHA256 mismatch: {case_id}")
        with np.load(asset_path, allow_pickle=False) as asset:
            valid = asset["valid_mask"].astype(bool, copy=True)
            foreground = asset["foreground_mask"].astype(bool, copy=True)
        expected_shape = tuple(int(value) for value in row["shape"])
        if valid.shape != foreground.shape or valid.shape != expected_shape:
            raise ValueError(f"malformed valid-mask asset: {case_id}")
        if not np.any(valid):
            raise ValueError(f"empty valid mask: {case_id}")
        segmentation = foreground.astype(np.int16)
        segmentation *= 3
        valid.setflags(write=False)
        segmentation.setflags(write=False)
        assets[case_id] = (valid, segmentation)
    if set(assets) != expected_ids:
        raise ValueError(
            "Gate 1 valid-mask IDs do not exactly match train target IDs: "
            f"missing={sorted(expected_ids - set(assets))[:10]}, "
            f"extra={sorted(set(assets) - expected_ids)[:10]}"
        )
    return assets


def build_target_case_schedule(*, case_count: int, events: int, seed: int) -> np.ndarray:
    if case_count < 1:
        raise ValueError("Gate 1 target case schedule requires at least one case")
    rng = np.random.default_rng(seed)
    schedule = np.fromiter(
        (int(rng.integers(0, case_count)) for _ in range(events)),
        dtype=np.int32,
        count=events,
    )
    schedule.setflags(write=False)
    return schedule


def split_event_ranges(events: int, workers: int) -> list[tuple[int, int]]:
    if workers < 1:
        raise ValueError("Gate 1 worker count must be positive")
    worker_count = min(int(workers), int(events))
    quotient, remainder = divmod(events, worker_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for worker_index in range(worker_count):
        stop = start + quotient + (1 if worker_index < remainder else 0)
        ranges.append((start, stop))
        start = stop
    if start != events:
        raise RuntimeError("Gate 1 event partition does not cover the frozen event stream")
    return ranges


def _run_shard(*, start: int, stop: int, events_path: Path) -> Gate1Stats:
    if any(
        value is None
        for value in (
            _FORK_MANIFEST,
            _FORK_CONFIG,
            _FORK_ASSETS,
            _FORK_CASE_IDS,
            _FORK_CASE_SCHEDULE,
        )
    ):
        raise RuntimeError("Gate 1 fork state was not initialized")
    manifest = _FORK_MANIFEST
    config = _FORK_CONFIG
    assets = _FORK_ASSETS
    case_ids = _FORK_CASE_IDS
    schedule = _FORK_CASE_SCHEDULE
    assert manifest is not None
    assert config is not None
    assert assets is not None
    assert case_ids is not None
    assert schedule is not None

    engine = MetAugEngine(
        manifest=manifest,
        config=config,
        backend=None,
        audit_sink=JsonlAuditSink(events_path),
    )
    stats = Gate1Stats()
    for patch_index in range(start, stop):
        case_id = case_ids[int(schedule[patch_index])]
        valid, segmentation = assets[case_id]
        result = engine.simulate(
            segmentation=segmentation,
            valid_mask=valid,
            context=EventContext(
                epoch=0,
                rank=0,
                worker=0,
                case_id=case_id,
                patch_index=patch_index,
                full_shape=tuple(int(value) for value in valid.shape),
            ),
            inputs_prevalidated=True,
        )
        stats.observe(result, target_group=manifest.target_groups[case_id])
    return stats


def _run_shard_process(start: int, stop: int, events_path: str, summary_path: str) -> None:
    stats = _run_shard(start=start, stop=stop, events_path=Path(events_path))
    payload = {"start": start, "stop": stop, "stats": stats.as_mapping()}
    Path(summary_path).write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _merge_event_shards(
    *,
    output_path: Path,
    shard_paths: list[Path],
    expected_events: int,
) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    observed_events = 0
    with temporary_path.open("xb") as destination:
        for shard_path in shard_paths:
            with shard_path.open("rb") as source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    observed_events += chunk.count(b"\n")
                    destination.write(chunk)
    if observed_events != expected_events:
        raise RuntimeError(
            f"Gate 1 merged JSONL has {observed_events} events, expected {expected_events}"
        )
    temporary_path.replace(output_path)


def _build_report(
    *,
    stats: Gate1Stats,
    events: int,
    manifest: ComponentManifest,
    config: RouteConfig,
    valid_mask_manifest_path: Path,
    target_seed: int,
) -> dict[str, Any]:
    expected_q = {
        "|".join(stratum.key): stratum.weight / sum(item.weight for item in config.strata)
        for stratum in config.strata
    }
    observed_q = {
        key: (stats.stratum_counts[key] / stats.placement_valid if stats.placement_valid else 0.0)
        for key in expected_q
    }
    q_max_abs_error = max(
        (abs(observed_q[key] - expected_q[key]) for key in expected_q),
        default=0.0,
    )
    selected_rate = stats.selected / events
    violations = []
    if abs(selected_rate - config.p_select) > 0.005:
        violations.append("selected_rate_outside_0.005")
    if stats.same_group_violations:
        violations.append("same_patient_group_donor")
    if q_max_abs_error > 0.01:
        violations.append("effective_strata_distribution_outside_0.01")
    if stats.placement_valid == 0:
        violations.append("no_valid_placements")
    return {
        "schema_version": 1,
        "route_id": config.route_id,
        "status": "pass" if not violations else "fail",
        "event_count": events,
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": sha256_file(config.path),
        "valid_mask_manifest_sha256": sha256_file(valid_mask_manifest_path),
        "target_seed": target_seed,
        "selected_rate": selected_rate,
        "expected_selected_rate": config.p_select,
        "stage_counts": {
            "selected": stats.selected,
            "donor_valid": stats.donor_valid,
            "label_valid": stats.label_valid,
            "placement_valid": stats.placement_valid,
        },
        "states": dict(sorted(stats.counts.items())),
        "reasons": dict(sorted(stats.reasons.items())),
        "same_group_violations": stats.same_group_violations,
        "expected_q": expected_q,
        "observed_q": observed_q,
        "q_max_abs_error": q_max_abs_error,
        "violations": violations,
    }


def run_gate1(
    *,
    component_manifest_path: str | Path,
    route_config_path: str | Path,
    valid_mask_manifest_path: str | Path,
    output_dir: str | Path,
    events: int,
    target_seed: int,
    workers: int,
    minimum_events: int = MIN_GATE1_EVENTS,
    enforce_acceptance: bool = True,
) -> dict[str, Any]:
    if events < minimum_events:
        raise ValueError(f"Gate 1 requires at least {minimum_events:,} events")
    if workers < 1:
        raise ValueError("Gate 1 worker count must be positive")
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if resolved_output_dir.exists():
        raise FileExistsError(
            f"Gate 1 output is immutable and already exists: {resolved_output_dir}"
        )
    resolved_output_dir.mkdir(parents=True, exist_ok=False)

    manifest = ComponentManifest.load(component_manifest_path)
    config = RouteConfig.load(route_config_path, manifest)
    resolved_valid_mask_manifest = Path(valid_mask_manifest_path).expanduser().resolve()
    assets = load_target_assets(resolved_valid_mask_manifest, set(manifest.target_groups))
    case_ids = tuple(sorted(assets))
    schedule = build_target_case_schedule(
        case_count=len(case_ids),
        events=events,
        seed=target_seed,
    )

    global _FORK_MANIFEST, _FORK_CONFIG, _FORK_ASSETS, _FORK_CASE_IDS, _FORK_CASE_SCHEDULE
    _FORK_MANIFEST = manifest
    _FORK_CONFIG = config
    _FORK_ASSETS = assets
    _FORK_CASE_IDS = case_ids
    _FORK_CASE_SCHEDULE = schedule

    ranges = split_event_ranges(events, workers)
    shard_dir = resolved_output_dir / ".gate1_shards"
    shard_dir.mkdir()
    shard_paths = [shard_dir / f"events_{index:04d}.jsonl" for index in range(len(ranges))]
    summary_paths = [shard_dir / f"summary_{index:04d}.json" for index in range(len(ranges))]

    if len(ranges) == 1:
        start, stop = ranges[0]
        stats = _run_shard(start=start, stop=stop, events_path=shard_paths[0])
        summary_paths[0].write_text(
            json.dumps(
                {"start": start, "stop": stop, "stats": stats.as_mapping()},
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel Gate 1 requires the fork multiprocessing start method")
        context = multiprocessing.get_context("fork")
        processes = [
            context.Process(
                target=_run_shard_process,
                args=(start, stop, str(shard_path), str(summary_path)),
                name=f"gate1-{index:02d}",
            )
            for index, ((start, stop), shard_path, summary_path) in enumerate(
                zip(ranges, shard_paths, summary_paths)
            )
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join()
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                if process.is_alive():
                    process.join()
        failures = [process for process in processes if process.exitcode != 0]
        if failures:
            details = ", ".join(
                f"{process.name}:exit={process.exitcode}" for process in failures
            )
            raise RuntimeError(f"parallel Gate 1 worker failure: {details}")

    aggregate = Gate1Stats()
    for (expected_start, expected_stop), summary_path in zip(ranges, summary_paths):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if (int(payload["start"]), int(payload["stop"])) != (expected_start, expected_stop):
            raise RuntimeError(f"Gate 1 shard range mismatch: {summary_path}")
        shard_stats = Gate1Stats.from_mapping(payload["stats"])
        if shard_stats.event_count != expected_stop - expected_start:
            raise RuntimeError(f"Gate 1 shard event count mismatch: {summary_path}")
        aggregate.merge(shard_stats)
    if aggregate.event_count != events:
        raise RuntimeError(
            f"Gate 1 worker summaries have {aggregate.event_count} events, expected {events}"
        )

    events_path = resolved_output_dir / "gate1_events.jsonl"
    _merge_event_shards(
        output_path=events_path,
        shard_paths=shard_paths,
        expected_events=events,
    )
    report = _build_report(
        stats=aggregate,
        events=events,
        manifest=manifest,
        config=config,
        valid_mask_manifest_path=resolved_valid_mask_manifest,
        target_seed=target_seed,
    )
    report_path = resolved_output_dir / "gate1_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(shard_dir)
    if enforce_acceptance and report["violations"]:
        raise RuntimeError(f"Gate 1 violations: {report['violations']}")
    return report
