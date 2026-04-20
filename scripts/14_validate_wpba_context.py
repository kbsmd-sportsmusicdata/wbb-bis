"""
Script 14 — WPBA Context Quality Gate
=====================================
Validates the WPBA context pipeline outputs with targeted checks:
  1) File contracts (exists, row counts, required columns)
  2) Join coverage checks
  3) Null thresholds and score range checks
  4) Score distribution sanity

Usage:
  python scripts/14_validate_wpba_context.py

Outputs:
  analysis/validation/wpba_quality_gate_report_<timestamp>.json
  analysis/validation/wpba_quality_gate_report_latest.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PROCESSED_DIR, SEASON


TEAM_STYLE_FILE = REPO_ROOT / "analysis" / "conference_efficiency" / "team_style_efficiency_2021_2026.csv"
PROGRAM_TRACK_FILE = REPO_ROOT / "analysis" / "player_development" / "program_fr_sr_track_2021_2026.csv"
WPBA_FILE = REPO_ROOT / "analysis" / "wpba" / f"wpba_fit_context_{SEASON}.csv"
FEATURE_FILE = PROCESSED_DIR / f"player_feature_table_{SEASON}.csv"

VALIDATION_DIR = REPO_ROOT / "analysis" / "validation"


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


def _check_exists(tracker: GateTracker, path: Path, label: str) -> bool:
    if not path.exists():
        tracker.fail(f"{label}: missing file ({path})")
        return False
    tracker.ok(f"{label}: file exists")
    return True


def _check_required_cols(tracker: GateTracker, df: pd.DataFrame, label: str, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        tracker.fail(f"{label}: missing required columns {missing}")
    else:
        tracker.ok(f"{label}: required columns present")


def _check_null_rate(
    tracker: GateTracker,
    df: pd.DataFrame,
    col: str,
    fail_thresh: float,
    warn_thresh: float,
    label: str,
) -> None:
    if col not in df.columns:
        tracker.fail(f"{label}: missing column '{col}'")
        return
    null_rate = float(df[col].isna().mean())
    if null_rate > fail_thresh:
        tracker.fail(f"{label}: {col} null rate {null_rate:.1%} > fail threshold {fail_thresh:.1%}")
    elif null_rate > warn_thresh:
        tracker.warn(f"{label}: {col} null rate {null_rate:.1%} > warn threshold {warn_thresh:.1%}")
    else:
        tracker.ok(f"{label}: {col} null rate {null_rate:.1%} within threshold")


def _check_numeric_range(
    tracker: GateTracker,
    df: pd.DataFrame,
    col: str,
    low: float,
    high: float,
    label: str,
) -> None:
    if col not in df.columns:
        tracker.fail(f"{label}: missing column '{col}'")
        return
    s = pd.to_numeric(df[col], errors="coerce")
    bad = ((s < low) | (s > high)) & s.notna()
    n_bad = int(bad.sum())
    if n_bad > 0:
        tracker.fail(f"{label}: {col} has {n_bad} values outside [{low}, {high}]")
    else:
        tracker.ok(f"{label}: {col} within [{low}, {high}]")


def run_quality_gate() -> int:
    tracker = GateTracker()

    print("\n== WPBA Context Quality Gate ==")
    print(f"Season: {SEASON}")

    # ---------------------------------------------------------------------
    # 1) File contracts
    # ---------------------------------------------------------------------
    print("\n== 1) File Contracts ==")
    ok_team = _check_exists(tracker, TEAM_STYLE_FILE, "team_style")
    ok_prog = _check_exists(tracker, PROGRAM_TRACK_FILE, "program_track")
    ok_wpba = _check_exists(tracker, WPBA_FILE, "wpba_fit")
    ok_feat = _check_exists(tracker, FEATURE_FILE, "feature_table")

    if not all([ok_team, ok_prog, ok_wpba, ok_feat]):
        return _finalize(tracker)

    team = pd.read_csv(TEAM_STYLE_FILE, low_memory=False)
    prog = pd.read_csv(PROGRAM_TRACK_FILE, low_memory=False)
    wpba = pd.read_csv(WPBA_FILE, low_memory=False)
    feat = pd.read_csv(FEATURE_FILE, low_memory=False)

    if len(team) < 3000:
        tracker.fail(f"team_style: rows too low ({len(team):,}); expected >= 3,000")
    else:
        tracker.ok(f"team_style: row count {len(team):,}")

    if len(prog) < 300:
        tracker.fail(f"program_track: rows too low ({len(prog):,}); expected >= 300")
    else:
        tracker.ok(f"program_track: row count {len(prog):,}")

    if len(wpba) != len(feat):
        tracker.fail(f"wpba_fit: row count {len(wpba):,} != feature table {len(feat):,}")
    else:
        tracker.ok(f"wpba_fit: row count matches feature table ({len(wpba):,})")

    # ---------------------------------------------------------------------
    # 2) Schema checks
    # ---------------------------------------------------------------------
    print("\n== 2) Schema Checks ==")
    _check_required_cols(
        tracker,
        team,
        "team_style",
        ["team_id", "season", "team_location", "offensive_eff", "defensive_eff", "pace", "conference"],
    )
    _check_required_cols(
        tracker,
        prog,
        "program_track",
        ["team_location", "program_fr_sr_track_score", "program_fr_sr_track_score_0_100"],
    )
    _check_required_cols(
        tracker,
        wpba,
        "wpba_fit",
        [
            "athlete_id",
            "athlete_display_name",
            "team_location",
            "conference",
            "role_name",
            "archetype_score_0_100",
            "development_trajectory_score_0_100",
            "conference_program_context_score_0_100",
            "wpba_pathway_fit_score_0_100",
            "wpba_fit_band",
        ],
    )

    # ---------------------------------------------------------------------
    # 3) Join-rate and null-threshold checks
    # ---------------------------------------------------------------------
    print("\n== 3) Coverage And Null Checks ==")

    # Feature table context coverage.
    _check_null_rate(tracker, feat, "conference", fail_thresh=0.10, warn_thresh=0.02, label="feature_table")
    _check_null_rate(tracker, feat, "offensive_eff", fail_thresh=0.25, warn_thresh=0.10, label="feature_table")
    _check_null_rate(tracker, feat, "defensive_eff", fail_thresh=0.25, warn_thresh=0.10, label="feature_table")
    _check_null_rate(tracker, feat, "pace", fail_thresh=0.25, warn_thresh=0.10, label="feature_table")

    # Team style conference coverage across full history may be partial.
    _check_null_rate(tracker, team, "conference", fail_thresh=0.90, warn_thresh=0.50, label="team_style")

    # Program track coverage in wpba output.
    _check_null_rate(
        tracker,
        wpba,
        "program_fr_sr_track_score_0_100",
        fail_thresh=0.40,
        warn_thresh=0.20,
        label="wpba_fit",
    )

    # Draft linkage is expected to be partial; warn-only style thresholds.
    if "in_draft_2026" in wpba.columns:
        draft_cov = float(wpba["in_draft_2026"].notna().mean())
        if draft_cov < 0.10:
            tracker.warn(f"wpba_fit: in_draft_2026 coverage low ({draft_cov:.1%})")
        else:
            tracker.ok(f"wpba_fit: in_draft_2026 coverage {draft_cov:.1%}")

    # ---------------------------------------------------------------------
    # 4) Score sanity checks
    # ---------------------------------------------------------------------
    print("\n== 4) Score Sanity Checks ==")
    score_cols = [
        "archetype_score_0_100",
        "development_trajectory_score_0_100",
        "conference_program_context_score_0_100",
        "wpba_pathway_fit_score_0_100",
    ]
    for c in score_cols:
        _check_null_rate(tracker, wpba, c, fail_thresh=0.02, warn_thresh=0.0, label="wpba_fit")
        _check_numeric_range(tracker, wpba, c, 0.0, 100.0, label="wpba_fit")

    # Distribution guardrails.
    if "wpba_pathway_fit_score_0_100" in wpba.columns:
        s = pd.to_numeric(wpba["wpba_pathway_fit_score_0_100"], errors="coerce")
        std = float(s.std(ddof=0)) if s.notna().any() else 0.0
        if std < 3.0:
            tracker.warn(f"wpba_fit: score std looks compressed ({std:.2f})")
        else:
            tracker.ok(f"wpba_fit: score std {std:.2f}")

        q10 = float(s.quantile(0.10))
        q90 = float(s.quantile(0.90))
        if (q90 - q10) < 8.0:
            tracker.warn(f"wpba_fit: score spread q90-q10 is narrow ({q90 - q10:.2f})")
        else:
            tracker.ok(f"wpba_fit: score spread q90-q10 = {q90 - q10:.2f}")

    if "wpba_fit_band" in wpba.columns:
        band_share = wpba["wpba_fit_band"].value_counts(normalize=True, dropna=False)
        if not band_share.empty and float(band_share.iloc[0]) > 0.98:
            tracker.warn("wpba_fit: nearly all rows collapsed into one fit band")
        else:
            tracker.ok("wpba_fit: fit-band distribution is non-collapsed")

    # Identity integrity.
    if "athlete_id" in wpba.columns:
        dupes = int(wpba["athlete_id"].duplicated().sum())
        if dupes > 0:
            tracker.fail(f"wpba_fit: duplicate athlete_id rows detected ({dupes})")
        else:
            tracker.ok("wpba_fit: one-row-per-athlete integrity holds")

    return _finalize(tracker)


def _finalize(tracker: GateTracker) -> int:
    print("\n== Summary ==")
    print(f"  Passed : {len(tracker.passes)}")
    print(f"  Warned : {len(tracker.warnings)}")
    print(f"  Failed : {len(tracker.failures)}")

    report = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "season": SEASON,
        "passed": tracker.passes,
        "warnings": tracker.warnings,
        "failures": tracker.failures,
        "counts": {
            "passed": len(tracker.passes),
            "warnings": len(tracker.warnings),
            "failures": len(tracker.failures),
        },
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = VALIDATION_DIR / f"wpba_quality_gate_report_{ts}.json"
    latest_path = VALIDATION_DIR / "wpba_quality_gate_report_latest.json"

    report_path.write_text(json.dumps(report, indent=2))
    latest_path.write_text(json.dumps(report, indent=2))

    print(f"\nSaved report: {report_path}")
    print(f"Saved latest: {latest_path}")

    return 1 if tracker.failures else 0


if __name__ == "__main__":
    raise SystemExit(run_quality_gate())
