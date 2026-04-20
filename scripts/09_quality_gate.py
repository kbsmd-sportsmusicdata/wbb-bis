"""
Script 09 — Pipeline Quality Gate
=================================
Runs production-style validation checks after a data refresh.

Covers:
  1) Data contracts (required raw inputs + schema + key integrity)
  2) Script/output quality checks (row counts, null-rates, metric ranges)
  3) Merge integrity checks for player_feature_table_2026.csv
  4) Regression snapshot + drift detection against prior run

Usage:
  python scripts/09_quality_gate.py

Outputs:
  analysis/validation/quality_gate_report_<timestamp>.json
  analysis/validation/kpi_snapshot_<timestamp>.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    PLAYER_BOX_FILE,
    PBP_FILE,
    SCHEDULE_FILE,
    NET_FILE,
    TOURNAMENT_BRACKET,
    PLAYER_BOX_ADVANCED,
    PLAYER_GAME_LOG,
    PBP_PLAYER_METRICS,
    PLAYER_ONOFF,
    POSTSEASON_ONOFF,
    PLAYER_SCOUTING_68,
    PLAYER_SCOUTING_50,
    PLAYER_FEATURE_TABLE,
    REPO_ROOT,
)


@dataclass
class GateTracker:
    passes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.passes.append(msg)
        print(f"  PASS  {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  WARN  {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  FAIL  {msg}")


INPUT_CONTRACTS = [
    {
        "name": "PLAYER_BOX_FILE",
        "path": PLAYER_BOX_FILE,
        "is_parquet": True,
        "required": True,
        "columns": ["athlete_id", "team_id", "game_id", "season", "minutes", "points"],
        "id_columns": ["athlete_id", "team_id", "game_id"],
        "null_threshold": 0.02,
    },
    {
        "name": "PBP_FILE",
        "path": PBP_FILE,
        "is_parquet": True,
        "required": True,
        "columns": ["game_id", "team_id", "athlete_id_1", "season_type", "shooting_play"],
        "id_columns": ["game_id", "team_id"],
        "null_threshold": 0.02,
    },
    {
        "name": "SCHEDULE_FILE",
        "path": SCHEDULE_FILE,
        "is_parquet": False,
        "required": True,
        "columns": ["game_id", "home_team_id", "away_team_id"],
        "id_columns": ["game_id", "home_team_id", "away_team_id"],
        "null_threshold": 0.02,
    },
    {
        "name": "NET_FILE",
        "path": NET_FILE,
        "is_parquet": False,
        "required": True,
        "columns": ["team_id", "net_rank"],
        "id_columns": ["team_id"],
        "null_threshold": 0.02,
    },
    {
        "name": "TOURNAMENT_BRACKET",
        "path": TOURNAMENT_BRACKET,
        "is_parquet": False,
        "required": True,
        "columns": ["seed", "region", "team_location"],
        "id_columns": [],
        "null_threshold": 0.02,
    },
]


OUTPUT_CHECKS = [
    {
        "name": "PLAYER_BOX_ADVANCED",
        "path": PLAYER_BOX_ADVANCED,
        "required": True,
        "rows": (1000, 10000),
        "required_cols": ["athlete_id", "team_id", "ts_pct", "usage_proxy", "pts_per40"],
        "null_checks": {"ts_pct": 0.20, "usage_proxy": 0.20},
    },
    {
        "name": "PLAYER_GAME_LOG",
        "path": PLAYER_GAME_LOG,
        "required": True,
        "rows": (50000, 600000),
        "required_cols": ["athlete_id", "game_id", "game_date", "points"],
        "null_checks": {},
    },
    {
        "name": "PBP_PLAYER_METRICS",
        "path": PBP_PLAYER_METRICS,
        "required": True,
        "rows": (1000, 12000),
        "required_cols": ["athlete_id", "paint_share", "three_share", "assisted_fg_rate"],
        "null_checks": {"paint_share": 0.25, "three_share": 0.25},
    },
    {
        "name": "PLAYER_ONOFF",
        "path": PLAYER_ONOFF,
        "required": True,
        "rows": (350, 500),
        "required_cols": ["athlete_id", "team_id", "on_net_rtg", "off_net_rtg", "net_rtg_diff"],
        "null_checks": {"on_net_rtg": 0.20, "net_rtg_diff": 0.10},
    },
    {
        "name": "POSTSEASON_ONOFF",
        "path": POSTSEASON_ONOFF,
        "required": False,
        "rows": (250, 1200),
        "required_cols": ["athlete_id", "tourney_net_rtg", "tourney_net_rtg_diff"],
        "null_checks": {"tourney_net_rtg_diff": 0.40},
    },
    {
        "name": "PLAYER_SCOUTING_68",
        "path": PLAYER_SCOUTING_68,
        "required": True,
        "rows": (500, 1200),
        "required_cols": ["athlete_id", "team_id", "seed", "region"],
        "null_checks": {"seed": 0.15, "region": 0.15},
    },
    {
        "name": "PLAYER_SCOUTING_50",
        "path": PLAYER_SCOUTING_50,
        "required": True,
        "rows": (250, 900),
        "required_cols": ["athlete_id", "team_id"],
        "null_checks": {},
    },
    {
        "name": "PLAYER_FEATURE_TABLE",
        "path": PLAYER_FEATURE_TABLE,
        "required": True,
        "rows": (350, 500),
        "required_cols": [
            "athlete_id",
            "team_id",
            "seed",
            "region",
            "net_rtg_diff",
            "ts_pct",
            "recruit_rank",
        ],
        "null_checks": {
            "ts_pct": 0.20,
            "on_net_rtg": 0.20,
            "tourney_net_rtg": 0.40,
            "recruit_rank": 0.80,
        },
    },
]


def _read_table(path: Path, is_parquet: Optional[bool] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if is_parquet is None:
        is_parquet = path.suffix.lower() == ".parquet"
    if is_parquet:
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def _check_range(tracker: GateTracker, df: pd.DataFrame, col: str, low: float, high: float, allow_na: bool = True) -> None:
    if col not in df.columns:
        tracker.warn(f"Range check skipped: column '{col}' not found")
        return
    s = pd.to_numeric(df[col], errors="coerce")
    if not allow_na and s.isna().any():
        tracker.fail(f"{col}: contains nulls but nulls are not allowed")
    out = ((s < low) | (s > high)) & s.notna()
    n_out = int(out.sum())
    if n_out > 0:
        tracker.warn(f"{col}: {n_out} values outside [{low}, {high}]")
    else:
        tracker.ok(f"{col}: all values within [{low}, {high}]")


def run_input_contract_checks(tracker: GateTracker) -> None:
    print("\n== 1) Data Contract Checks ==")
    for spec in INPUT_CONTRACTS:
        name = spec["name"]
        path = spec["path"]
        required = spec["required"]
        print(f"\n- {name}: {path}")

        if not path.exists():
            if required:
                tracker.fail(f"{name}: missing required file")
            else:
                tracker.warn(f"{name}: missing optional file")
            continue

        cols = spec["columns"]
        try:
            df = _read_table(path, is_parquet=spec["is_parquet"], columns=cols)
        except Exception as exc:
            tracker.fail(f"{name}: could not read file ({exc})")
            continue

        missing = [c for c in cols if c not in df.columns]
        if missing:
            tracker.fail(f"{name}: missing required columns {missing}")
            continue
        tracker.ok(f"{name}: required columns present")

        for id_col in spec["id_columns"]:
            null_rate = float(df[id_col].isna().mean()) if id_col in df.columns else 1.0
            if null_rate > spec["null_threshold"]:
                tracker.fail(
                    f"{name}: {id_col} null rate {null_rate:.1%} exceeds {spec['null_threshold']:.1%}"
                )
            else:
                tracker.ok(
                    f"{name}: {id_col} null rate {null_rate:.1%} within threshold"
                )


def run_output_checks(tracker: GateTracker) -> Dict[str, int]:
    print("\n== 2) Script-Level Output Checks ==")
    row_counts: Dict[str, int] = {}

    for spec in OUTPUT_CHECKS:
        name = spec["name"]
        path = spec["path"]
        required = spec["required"]
        print(f"\n- {name}: {path}")

        if not path.exists():
            if required:
                tracker.fail(f"{name}: missing required output")
            else:
                tracker.warn(f"{name}: optional output missing")
            continue

        try:
            df = _read_table(path)
        except Exception as exc:
            tracker.fail(f"{name}: could not read output ({exc})")
            continue

        n_rows = len(df)
        row_counts[name] = n_rows
        min_rows, max_rows = spec["rows"]
        if n_rows < min_rows or n_rows > max_rows:
            tracker.fail(f"{name}: row count {n_rows} outside expected [{min_rows}, {max_rows}]")
        else:
            tracker.ok(f"{name}: row count {n_rows} within expected range")

        missing = [c for c in spec["required_cols"] if c not in df.columns]
        if missing:
            tracker.fail(f"{name}: missing required columns {missing}")
            continue
        tracker.ok(f"{name}: required columns present")

        for col, max_null_rate in spec["null_checks"].items():
            if col not in df.columns:
                tracker.warn(f"{name}: null-rate check skipped (missing column '{col}')")
                continue
            null_rate = float(df[col].isna().mean())
            if null_rate > max_null_rate:
                tracker.warn(
                    f"{name}: {col} null rate {null_rate:.1%} above target {max_null_rate:.1%}"
                )
            else:
                tracker.ok(
                    f"{name}: {col} null rate {null_rate:.1%} within target"
                )

        if name == "PLAYER_FEATURE_TABLE":
            _check_range(tracker, df, "fg_pct", 0.0, 1.0)
            _check_range(tracker, df, "efg_pct", 0.0, 1.0)
            _check_range(tracker, df, "ts_pct", 0.0, 1.0)
            _check_range(tracker, df, "three_share", 0.0, 1.0)
            _check_range(tracker, df, "paint_share", 0.0, 1.0)
            _check_range(tracker, df, "pts_per40", 0.0, 60.0)
            _check_range(tracker, df, "ast_per40", 0.0, 20.0)
            _check_range(tracker, df, "reb_per40", 0.0, 30.0)
            if "athlete_id" in df.columns:
                dupes = int(df.duplicated(subset=["athlete_id"]).sum())
                if dupes > 0:
                    tracker.fail(f"PLAYER_FEATURE_TABLE: duplicated athlete_id rows ({dupes})")
                else:
                    tracker.ok("PLAYER_FEATURE_TABLE: one row per athlete_id")

    return row_counts


def run_merge_integrity_checks(tracker: GateTracker) -> Dict[str, float]:
    print("\n== 3) Merge Integrity Checks (Feature Table) ==")
    if not PLAYER_FEATURE_TABLE.exists():
        tracker.fail("PLAYER_FEATURE_TABLE missing; merge integrity checks skipped")
        return {}

    df = pd.read_csv(PLAYER_FEATURE_TABLE)
    coverage: Dict[str, float] = {}

    source_sentinels: List[Tuple[str, List[str], float]] = [
        ("box_advanced", ["minutes_total", "ts_pct", "usage_proxy"], 0.80),
        ("pbp_metrics", ["paint_share", "three_share", "assisted_fg_rate"], 0.70),
        ("regular_onoff", ["on_net_rtg", "off_net_rtg", "net_rtg_diff"], 0.80),
        ("postseason_onoff", ["tourney_net_rtg", "tourney_net_rtg_diff"], 0.55),
        ("recruiting", ["recruit_rank"], 0.20),
        ("bracket", ["seed", "region"], 0.85),
    ]

    for source_name, cols, min_expected in source_sentinels:
        present = [c for c in cols if c in df.columns]
        if not present:
            tracker.warn(f"{source_name}: no sentinel columns present")
            coverage[source_name] = 0.0
            continue
        rates = [float(df[c].notna().mean()) for c in present]
        avg_rate = float(np.mean(rates))
        coverage[source_name] = avg_rate
        if avg_rate >= min_expected:
            tracker.ok(f"{source_name}: coverage {avg_rate:.1%} (target {min_expected:.1%})")
        else:
            tracker.warn(f"{source_name}: coverage {avg_rate:.1%} below target {min_expected:.1%}")

    if "seed" in df.columns:
        unmatched_seed = int(df["seed"].isna().sum())
        if unmatched_seed > 0:
            tracker.warn(f"bracket coverage: {unmatched_seed} players missing seed")
        else:
            tracker.ok("bracket coverage: all players have seed")

    return coverage


def _snapshot_payload(row_counts: Dict[str, int], merge_coverage: Dict[str, float]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "row_counts": row_counts,
        "merge_coverage": {k: round(v, 6) for k, v in merge_coverage.items()},
        "metrics": {},
    }

    if PLAYER_FEATURE_TABLE.exists():
        df = pd.read_csv(PLAYER_FEATURE_TABLE)
        payload["metrics"] = {
            "players": int(len(df)),
            "teams": int(df["team_id"].nunique()) if "team_id" in df.columns else None,
            "mean_net_rtg_diff": float(pd.to_numeric(df.get("net_rtg_diff"), errors="coerce").mean()),
            "mean_ts_pct": float(pd.to_numeric(df.get("ts_pct"), errors="coerce").mean()),
            "recruit_match_rate": float(df["recruit_rank"].notna().mean()) if "recruit_rank" in df.columns else None,
            "postseason_match_rate": float(df["tourney_net_rtg"].notna().mean()) if "tourney_net_rtg" in df.columns else None,
        }

    return payload


def run_regression_snapshot(tracker: GateTracker, row_counts: Dict[str, int], merge_coverage: Dict[str, float]) -> Dict[str, Any]:
    print("\n== 4) Regression Snapshot Checks ==")

    out_dir = REPO_ROOT / "analysis" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _snapshot_payload(row_counts, merge_coverage)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = out_dir / f"kpi_snapshot_{ts}.json"
    report_path = out_dir / f"quality_gate_report_{ts}.json"

    previous = sorted(out_dir.glob("kpi_snapshot_*.json"))
    prev_payload: Optional[Dict[str, Any]] = None
    if previous:
        prev_latest = previous[-1]
        try:
            prev_payload = json.loads(prev_latest.read_text())
            tracker.ok(f"Loaded prior snapshot: {prev_latest.name}")
        except Exception as exc:
            tracker.warn(f"Could not parse prior snapshot {prev_latest.name}: {exc}")

    if prev_payload:
        prev_rows = prev_payload.get("row_counts", {})
        for key, cur in row_counts.items():
            prev = prev_rows.get(key)
            if prev is None or prev == 0:
                continue
            drift = abs(cur - prev) / prev
            if drift > 0.35:
                tracker.warn(f"Row-count drift: {key} changed {drift:.1%} (prev={prev}, cur={cur})")
            else:
                tracker.ok(f"Row-count drift: {key} {drift:.1%} (stable)")

    snapshot_path.write_text(json.dumps(payload, indent=2))
    tracker.ok(f"Wrote KPI snapshot: {snapshot_path}")

    report = {
        "generated_at": payload["run_timestamp"],
        "summary": {
            "pass_count": len(tracker.passes),
            "warning_count": len(tracker.warnings),
            "failure_count": len(tracker.failures),
        },
        "passes": tracker.passes,
        "warnings": tracker.warnings,
        "failures": tracker.failures,
        "snapshot_file": str(snapshot_path),
        "snapshot": payload,
    }
    report_path.write_text(json.dumps(report, indent=2))
    tracker.ok(f"Wrote gate report: {report_path}")

    return report


def main() -> int:
    print("\n" + "=" * 72)
    print("WBB BIS — QUALITY GATE")
    print("Checks: contracts -> outputs -> merge integrity -> regression")
    print("=" * 72)

    tracker = GateTracker()

    run_input_contract_checks(tracker)
    row_counts = run_output_checks(tracker)
    merge_coverage = run_merge_integrity_checks(tracker)
    run_regression_snapshot(tracker, row_counts, merge_coverage)

    print("\n" + "=" * 72)
    print("QUALITY GATE SUMMARY")
    print("=" * 72)
    print(f"Passes  : {len(tracker.passes)}")
    print(f"Warnings: {len(tracker.warnings)}")
    print(f"Failures: {len(tracker.failures)}")

    if tracker.failures:
        print("\nStatus: FAIL (blocking issues found)")
        return 1

    if tracker.warnings:
        print("\nStatus: PASS WITH WARNINGS (review before publish)")
        return 0

    print("\nStatus: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
