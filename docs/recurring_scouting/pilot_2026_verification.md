# Six-team 2026 pilot verification

Verified November 2, 2026 in an isolated pilot branch using the four requested ESPN WBB releases and supporting schedule/membership snapshots.

- 148 scouting tests passed under Python 3.12, including observed RED-to-GREEN fresh/stale external-cache provenance regressions.
- A real external-cache build into /private/tmp/wbb-pilot-external-pteit9yy produced all 22 tables and a valid portable ZIP; every logical table hash matched the main pilot output.
- Independent review recomputed all 12 team/window eFG%, offensive/defensive ratings, offensive rebound rates, player appearances/production shares and shooting-zone denominators. It confirmed source-row reorder stability and preservation of all 1,047 identical shot rows.
- Source-backed rebuild reported zero changed tables, 207 validated shot groups, six withheld groups and zero box quarantines.
- All 22 CSV and Parquet row counts, table grains and file checksums passed. The 49-member portable archive passed CRC and exact member-byte comparison.
- Browser checks exercised all three matchups, both windows, focus team, player inspection (including California box-only Sakima Walker), made-three filtering, and historical game selection.
- California last-five showed five team games; Lulu Twidale's made-three map showed 11 attempts. Desktop 1440x1000 and mobile 390x844 were inspected. Mobile document client/scroll width were both 375; wide tables scroll within their panels.
- No browser console errors or warnings were observed.
- Browser download-event completion remains unverified: the in-app bridge stalled while awaiting a blob export. Static CSV/Parquet files and the downloadable ZIP are verified and provide a complete delivery path. Do not claim browser-download end-to-end acceptance from this run.

The single whole-branch review found one Important issue: rebuilding from an external cache did not persist source provenance in the destination before packaging. Fixed by writing the verified source record into every destination; regressions cover fresh and stale destination records. No Critical or Minor findings.

Review boundaries were explicitly ruled: browser download completion remains a disclosed QA limitation; current eligibility/availability and 2027 projections are outside this historical pilot; production deployment is outside this request. The pilot retains historical labels and data-quality exclusions.

Generated data, portable bundle and verification screenshots are local ignored artifacts under data/scouting/pilot_2026. The PR contains the reproducible pipeline, dashboard template, tests and operator documentation.
