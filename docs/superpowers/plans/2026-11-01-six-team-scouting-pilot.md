# Six-team scouting pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build validated six-team 2026 datasets and a populated interactive scouting dashboard for the user's three opening matchups.
**Architecture:** Independent staged pilot reusing existing adapters, validation, box metrics, and HTML conventions; historical state isolated from recurring production.
**Tech Stack:** Python 3.12, pandas/numpy/pyarrow, pytest, standalone interactive HTML/SVG.
**Spec:** docs/superpowers/specs/2026-11-01-six-team-scouting-pilot.md

## Global Constraints

- Six exact ESPN IDs: 265,25,120,2579,201,153; season 2026 is historical 2025-26.
- Four requested datasets from verified ESPN WBB tags; supporting schedules and memberships are explicit.
- Preserve whole-game validation context and repeated identical shots; no name-only identity join.
- Every rate retains counts, coverage, source lineage; incomplete shot groups are withheld, missing values stay null.
- No changes to production collection/schedule or unrelated original-checkout files.

## Review Focus

1. Identical free throws and source-row reordering must preserve multiplicity and deterministic internal IDs (Task 1).
2. Roster snapshots omit historical box participants; roster views must preserve them with an explicit label (Task 2).
3. A partially missing shot game must not yield biased team/player spatial efficiency (Tasks 1-2).
4. Team metrics need opponent rows even when the opponent is outside the six-team cohort (Task 2).
5. Filters, charts and exports must use the same team/player/window and keep historical labels visible (Task 3).

## Task 1: Source extraction and clean data

**Files:** config/scouting_pilot_2026.json, wbb_scouting/pilot_data.py, tests/scouting/test_pilot_data.py.
**Interfaces:** `load_pilot_sources(raw_dir,config)` verifies exact source metadata; `normalize_shots(frame,season)` preserves multiset identity; `clean_pilot(raw,config,now)` returns cleaned featured tables, validation context and quality records.
- [x] Write tests for repeated FT preservation, reorder-stable row IDs, wrong-season/null-ID rejection, physical coordinates and group shot/box reconciliation; observe RED.
- [x] Implement exact sources/config extraction and clean stages, reusing existing source checks and box validation.
- [x] Run tests GREEN and a real six-team extraction; inspect counts and missing games; commit Task 1.

## Task 2: Processed and feature datasets

**Files:** wbb_scouting/pilot_features.py, wbb_scouting/pilot.py, tests/scouting/test_pilot_features.py, tests/scouting/test_pilot.py; extend cli.py with independent `pilot` command.
**Interfaces:** `process_pilot(clean,config)` produces enriched team/player games and roster union; `build_pilot_features(processed,clean,config)` returns team/player summaries, trends and shot-zone tables; `run_pilot(config,output,now)` writes stages/manifests and report payload.
- [x] Write tests for opponent-denominator preservation, weighted rates, roster union, player attribution, coverage exclusions and unchanged table hashes; observe RED.
- [x] Implement processed/features and independent CLI, with CSV+Parquet outputs and explicit dataset status/grain.
- [x] Run GREEN tests and source-backed build twice; inspect each layer, formulas, six-team counts, row IDs and zero changes; commit Task 2.

## Task 3: Dashboard sections and interactive verification

**Files:** wbb_scouting/pilot_dashboard.py, templates/pilot_dashboard.html, tests/scouting/test_pilot_dashboard.py; docs/recurring_scouting/pilot_2026.md.
**Interfaces:** `pilot_payload(processed,features,quality,config,manifest)` binds actual saved datasets; `render_pilot_dashboard(payload,destination)` embeds safe data and all sections.
- [x] Write safe embedding, historical labels and stage/coverage binding tests; observe RED.
- [x] Implement all three matchup comparisons, team trends, player/roster table, shot map, zones, selected-player log, sources and filtered downloads.
- [x] Verify real browser desktop/mobile interactions, all matchups, filters, downloads, nonempty charts and console; run full tests and diff check; commit Task 3.

## Task 4: Review and concrete delivery

**Files:** verification notes and completed plan.
- [x] Request one fresh whole-branch review after implementation; reproduce and fix Important findings with regressions.
- [x] Rebuild from saved source snapshots; inspect fresh logs, table fingerprints and output links; finish relevant tests.
- [x] Open and attach a scoped reviewable PR, preserve local dashboard/data preview, record exact coverage and limitations. Production remains on the approved live version until concrete-result approval.
