# Scouting verification

The user approved the concrete preview and activation on November 1, 2026. [PR #3](https://github.com/kbsmd-sportsmusicdata/wbb-bis/pull/3) and the verified release-link fix in [PR #4](https://github.com/kbsmd-sportsmusicdata/wbb-bis/pull/4) are merged. The repository variable `WBB_SCOUTING_ENABLED=true` enables the publisher on the default branch.

## Summary

- Independent review found four Important issues; regression tests reproduce and cover the fixes. A selected postgame CSV issue was regraded Important and fixed in the same pass.
- 127 focused tests pass in the scoped and newly installed Python 3.12 environments.
- Real input now supports 6,025 team-box and 6,020 player-box historical games. Nine player groups are quarantined, including two raw missing-identity groups. Five expected historical team-box games and ten player-box games are missing or withheld. Current season 2027 boxes/PBP and verified NET are pending.
- The final indexed full local rerun took 125.69 seconds and produced zero logical row additions, updates or removals.
- Real original-source versus restored-canonical roster order produced identical report payloads: one edition remained one edition.
- The current preview contains one actual Notre Dame–Villanova pregame report at a frozen November 1, 2026, 16:23 UTC cutoff. Coverage currently includes 18 checked ACC teams; no top-50 NET selection is fabricated.
- The separate historical example uses actual Duke games from 2025–26, including 87–85 versus LSU. Its baseline is explicitly reconstructed.

## Environment and checks

Browser plugin unavailable; used existing Playwright with installed Google Chrome. Local preview: http://127.0.0.1:4197/ . Desktop 1440×1000; phone 390×844. Files also open offline.

| Check | Result |
|---|---|
| Intended URL/title and meaningful first screen | PASS |
| Blank page / framework error overlay | PASS, none |
| Relevant page/console errors | PASS, zero |
| Team, report mode, matchup and edition state | PASS |
| Published roster and season/last-five/game player views | PASS |
| Empty and pending current-season states | PASS |
| Selected JSON identifiers and CSV window/cutoff/source fields | PASS |
| Actual postgame result and baseline sections in CSV | PASS |
| Print PDF rendering | PASS |
| Phone width, controls and table scrolling | PASS, no page overflow |

Interaction loop: current pregame → roster → JSON download → California empty state → postgame empty state → separate historical Duke example → postgame → player windows → second matchup → CSV download → print → phone viewport. Source and coverage labels were retained.

## Persistence and recovery checks

Fake GitHub API tests cover cold-runner restoration, unrelated releases, production 404 exceptions, invalid-latest fallback, all-invalid refusal, missing/corrupt assets, failed upload leaving the prior published release intact, stale-parent rejection, unchanged references, sharded reconstruction, bounded restoration of active raw versions and reuse by published checksum. No credentials or private notes are embedded in the dashboard.

Historical originals remain on immutable remote release links. Routine restore verifies canonical partitions, current report assets and current raw inputs rather than downloading every obsolete source version. An old raw version is checksum-verified when explicitly recovered.

## Activation evidence

- The first real cloud publisher succeeded in [run 41336617069, attempt 1](https://github.com/kbsmd-sportsmusicdata/wbb-bis/actions/runs/41336617069/attempts/1), publishing 40 assets totaling 18,657,824 bytes. Its fresh-runner attempt 2 failed closed because GitHub changed draft `untagged-*` asset URLs on promotion. The original release remains preserved.
- Two production-like regressions reproduced this defect before the fix; all 127 tests then passed. The publisher now saves final-tag URLs while upload verification uses the asset API. Legacy draft references are repaired only against the same release's matching asset name, SHA-256, byte size and canonical URL. A real clean local restore verified 26 canonical partitions, five output artifacts and seven current raw sources.
- The merged fix is commit `430ce00b9aa6a1ad9fdd6317c9ca7bc8a7c4a378`. [Cloud run 41369793756, attempt 1](https://github.com/kbsmd-sportsmusicdata/wbb-bis/actions/runs/41369793756/attempts/1) restored the first release, completed without collection errors, and published [checkpoint 20cdc06d](https://github.com/kbsmd-sportsmusicdata/wbb-bis/releases/tag/scouting-v1-2027-20261102005619229484-20cdc06d). Every collected dataset had inserted=0, updated=0 and removed=0. All seven raw versions and 26 partitions were reused; this release added seven output/manifest assets totaling 2,611,620 bytes.
- Downloaded the real published manifest, HTML, JSON, both CSVs and summary; verified their SHA-256 values and final links. The dashboard's team and pregame/postgame selectors work, with zero browser warnings/errors. At the 2026-11-02 00:56 UTC cutoff it truthfully has 18 ACC teams and zero current report editions. Current 2027 boxes/PBP and verified NET remain pending; historical reference data does not fill those gaps.
- [Cloud run 41369793756, attempt 2](https://github.com/kbsmd-sportsmusicdata/wbb-bis/actions/runs/41369793756/attempts/2) also succeeded from a fresh runner. It restored checkpoint 20cdc06d and published [checkpoint a751d5b8](https://github.com/kbsmd-sportsmusicdata/wbb-bis/releases/tag/scouting-v1-2027-20261102005928603364-a751d5b8), again with zero inserted/updated/removed rows, zero collection errors, seven new output/manifest assets and 2,611,620 bytes. The downloaded latest manifest, all outputs, exact parent link and absence of draft URLs were independently verified. Operational activation acceptance is complete; current data coverage remains partial for the explicit source reasons above.
- The final recovery-fix PR check [41369630480](https://github.com/kbsmd-sportsmusicdata/wbb-bis/actions/runs/41369630480) passed both tests and source-preview jobs. The full repository suite has 127 passing tests; the separate clean environment also verified the recovery regressions.
- Effective workflow state is active on main, with timezone `America/Los_Angeles`: M/W/F 06:17 primary and M/W/F 08:17 conditional retry. Actual GitHub metadata with no Monday primary correctly requires a retry; behavioral tests cover successful-primary skip and failed/pending-primary retry. Next primary is Monday, November 2, 2026 at 06:17 PST (14:17 UTC). This future scheduled invocation has not yet occurred; the completed checks were manual cloud runs.

Public schedules are best effort and can become disabled after prolonged repository inactivity. Current source unavailability remains explicit even after activation. No automatic release deletion is enabled.

The initial delivery is downloadable HTML plus JSON/CSV. A permanently hosted URL, private access, live-game reports, lineups, possessions and RAPM integration remain outside this approved version.
