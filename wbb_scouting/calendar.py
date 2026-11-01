"""The approved Monday/Wednesday/Friday operating calendar."""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
WEEKDAYS = frozenset((0, 2, 4))
PRIMARY = time(6, 17)
RETRY = time(8, 17)

def aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An explicit timezone is required")
    return value.astimezone(timezone.utc)

def parse_time(value: str) -> datetime:
    return aware_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))

def next_primary_check(now: datetime) -> datetime:
    now = aware_utc(now)
    local = now.astimezone(PACIFIC)
    for offset in range(8):
        day = local.date() + timedelta(days=offset)
        candidate = datetime.combine(day, PRIMARY, tzinfo=PACIFIC)
        if day.weekday() in WEEKDAYS and candidate > now:
            return candidate.astimezone(timezone.utc)
    raise RuntimeError("Unable to resolve next check")

def refresh_window(now: datetime) -> tuple[datetime, datetime]:
    return aware_utc(now), next_primary_check(now)

def retry_required(now: datetime, primary_attempt: dict | None) -> bool:
    local = aware_utc(now).astimezone(PACIFIC)
    if local.weekday() not in WEEKDAYS or local.time().replace(tzinfo=None) < RETRY:
        return False
    if not primary_attempt:
        return True
    try:
        primary_day = parse_time(primary_attempt["attempted_at"]).astimezone(PACIFIC).date()
    except (KeyError, ValueError, TypeError):
        return True
    return (primary_day != local.date()
            or primary_attempt.get("status") != "complete"
            or bool(primary_attempt.get("pending")))

def schedule_health(last_attempt: datetime | None, now: datetime) -> str:
    if last_attempt is None:
        return "never_run"
    deadline = next_primary_check(last_attempt) + timedelta(hours=2, minutes=15)
    return "overdue" if aware_utc(now) > deadline else "on_schedule"
