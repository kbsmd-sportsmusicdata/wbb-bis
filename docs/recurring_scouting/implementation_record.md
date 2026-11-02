# Implementation decisions

Decisions recorded during the approved implementation, in chronological order. Activation gates described below were satisfied after the user reviewed the concrete preview; see [verification](verification.md) for current operational evidence.

- Ruling: Use the explicitly approved standalone Python/HTML deliverable and GitHub Actions rather than Data skill defaults for React/Sites/cloud chats — user selected these deliverables and scheduler — cost if wrong: a later hosted URL needs separate delivery work.

- Ruling: Use a scoped Python 3.12 environment for execution and pinned dependencies; system Python is 3.9 — meets approved Python 3.11+ floor — cost if wrong: environment setup must be reproduced in cloud.

- Task 2: Ruling: official NET parsing/retrieval is implemented with Task 4 references, where publication dates/cohort rules are validated — avoids duplicate parsing code — cost if wrong: NET remains visibly pending until its source and reference contracts are verified.

- Task 6: Ruling: embed a 14-day recent report window plus all upcoming editions; retain every edition in full JSON/CSV exports and dated release snapshots — bounds standalone file growth — cost if wrong: older interactive reports require opening their archived dashboard.

- Final: Ruling: restore only active raw versions plus canonical/report assets while keeping all original versions archived remotely — avoids unbounded runner disk/network growth identified in review — cost if wrong: unused historical raw corruption is detected on explicit recovery rather than every refresh.

- Final: Ruling: operational release/schedule acceptance remains gated by the approved concrete-preview review — reviewer correctly set aside unapproved activation and no operational claim is made — cost if wrong: automatic reporting starts only after that approval and verification.

- Final: Ruling: unavailable current boxes/PBP and official NET remain source-pending rather than defects masked by history — reviewer verified fallback handling; no substitute ranking is invented — cost if wrong: reports and top-50 coverage stay limited until authoritative sources are available.

No minor review findings were deferred. Independent whole-branch review and every Important correction are recorded in the verification notes and regression tests.
