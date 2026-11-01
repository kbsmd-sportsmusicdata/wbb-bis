# Scouting preview verification

Implementation is proposed in [PR #3](https://github.com/kbsmd-sportsmusicdata/wbb-bis/pull/3). The publishing workflow is disabled pending the approved concrete-preview review. This is not an operational-service completion claim.

## Summary

- Independent review found four Important issues; regression tests reproduce and cover the fixes. A selected postgame CSV issue was regraded Important and fixed in the same pass.
- 125 focused tests pass in the scoped and newly installed Python 3.12 environments.
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

## Remaining operational acceptance

After concrete preview approval: merge the reviewed PR, enable WBB_SCOUTING_ENABLED, verify actual release publication, restore into a clean GitHub runner, verify an unchanged rerun, and confirm effective M/W/F primary and conditional-retry schedule behavior. Public schedules are best effort and can become disabled after prolonged repository inactivity. Current source unavailability remains explicit even after activation.

The initial delivery is downloadable HTML plus JSON/CSV. A permanently hosted URL, private access, live-game reports, lineups, possessions and RAPM integration remain outside this approved version.
